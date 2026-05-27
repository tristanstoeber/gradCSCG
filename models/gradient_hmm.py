"""
GradientHMM — Action-augmented Cloned HMM (Neuralized CSCG).

This module is the canonical importable home of the ``GradientHMM`` originally
defined inside ``CSCG_Gradient_based_training_(time_effiecent_version).ipynb``.
The class supports two complementary modes:

1. **Legacy single-sequence mode** (notebook-compatible). Pass
   ``obs_sequence`` and ``action_seq`` to ``__init__``; call
   :meth:`hmm_loss` with no arguments. The forward pass is XLA-compiled with
   a fixed maximum-iterations bound. This path is bit-for-bit equivalent to
   the original notebook implementation.

2. **Batched mode** (new). Construct the HMM without sequences, then call
   :meth:`batch_loss` with padded ``(O, A, L)`` tensors. The same parameters
   ``(pi, transition)`` are reused across batches — no rebuild required.
   This is the path used by :class:`models.vqvae_cscg.VQVAEGradientHMM`.

Emissions are deterministic in the cloned-HMM sense: state ``s`` is compatible
with observation ``o`` iff ``time_of_sta[s] == o``, where
``time_of_sta[s] = s // num_clones`` for ``s < num_states - 1`` in the
uniform-clone case. Callers can also pass an explicit ``state_to_obs`` mapping
to support nonuniform clone counts per observation. The final "ancillary"
state acts as a sink and never matches any real observation.
"""

from __future__ import annotations

import time
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

NEG_INF: float = -1.0e10


def configure_tensorflow(mixed_precision: bool = False, verbose: bool = False) -> None:
    """Best-effort TF runtime tuning. Safe to call multiple times.

    Mixed precision is **off by default** because XLA + ``logsumexp`` can
    underflow in fp16. Enable only if you have validated convergence on your
    own data.
    """
    if mixed_precision:
        try:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            if verbose:
                print("[gradient_hmm] mixed precision enabled")
        except Exception as e:  # pragma: no cover
            if verbose:
                print(f"[gradient_hmm] mixed precision unavailable: {e}")

    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if verbose:
            print(f"[gradient_hmm] {len(gpus)} GPU(s); memory growth enabled")
    except RuntimeError as e:  # pragma: no cover
        if verbose:
            print(f"[gradient_hmm] GPU already initialized: {e}")


class GradientHMM(tf.Module):
    """Action-augmented cloned HMM with deterministic emissions.

    Parameters
    ----------
    num_states
        Total number of latent states. Conventionally
        ``num_clones * num_observations + 1`` (the trailing ``+1`` is the
        ancillary sink state).
    num_actions
        Size of the action alphabet ``A``.
    num_clones
        Number of clones per observation symbol in the uniform-clone case.
    state_to_obs
        Optional explicit compatible-observation id per non-sink state. If
        provided, ``num_states`` must equal ``len(state_to_obs) + 1`` or
        ``len(state_to_obs)`` when the sink entry is included. This enables
        fixed nonuniform clone counts while preserving deterministic CSCG
        emissions.
    obs_sequence, action_seq
        Optional. If provided, the HMM stores them and supports the legacy
        single-sequence path (``hmm_loss()`` with no args). If omitted, only
        the batched path (:meth:`batch_loss`) is available.
    seed
        Optional RNG seed for parameter initialization.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        num_clones: int,
        obs_sequence: Optional[Sequence[int]] = None,
        action_seq: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
        state_to_obs: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.num_clones = int(num_clones)

        # state -> compatible observation id (sink state is unreachable by data)
        if state_to_obs is None:
            state_ids = tf.range(self.num_states - 1, dtype=tf.int32)
            time_of_sta = state_ids // self.num_clones
            time_of_sta = tf.concat(
                [time_of_sta, tf.constant([self.num_states], dtype=tf.int32)], axis=0
            )
        else:
            state_to_obs_arr = np.asarray(state_to_obs, dtype=np.int32)
            if state_to_obs_arr.ndim != 1:
                raise ValueError("state_to_obs must be a 1-D sequence")
            if state_to_obs_arr.shape[0] == self.num_states - 1:
                sink_obs = max(
                    self.num_states,
                    int(state_to_obs_arr.max()) + 1 if state_to_obs_arr.size else 0,
                )
                state_to_obs_arr = np.concatenate(
                    [state_to_obs_arr, np.array([sink_obs], dtype=np.int32)]
                )
            elif state_to_obs_arr.shape[0] != self.num_states:
                raise ValueError(
                    "state_to_obs must have length num_states - 1 "
                    f"or num_states; got {state_to_obs_arr.shape[0]} "
                    f"for num_states={self.num_states}"
                )
            if (state_to_obs_arr[:-1] < 0).any():
                raise ValueError("non-sink state_to_obs entries must be non-negative")
            time_of_sta = tf.constant(state_to_obs_arr, dtype=tf.int32)
        self.time_of_sta = time_of_sta

        # ---- parameter init ----
        if seed is not None:
            tf.random.set_seed(seed)

        W = tf.concat(
            [
                tf.random.normal(
                    [self.num_states, self.num_states - 1],
                    mean=1.0,
                    stddev=0.1,
                    dtype=tf.float32,
                ),
                tf.fill([self.num_states, 1], 10.0),
            ],
            axis=1,
        )
        logits_one_action = tf.math.log(W + 1e-9)
        init_logits = tf.repeat(
            logits_one_action[tf.newaxis, ...], repeats=self.num_actions, axis=0 # type: ignore
        )

        self.transition = tf.Variable(init_logits, trainable=True, dtype=tf.float32)
        self.pi = tf.Variable(
            tf.random.normal([self.num_states], dtype=tf.float32), trainable=True
        )

        # ---- legacy single-sequence buffers (optional) ----
        self.O: Optional[tf.Tensor] = None
        self.A: Optional[tf.Tensor] = None
        self.sequence_length: Optional[int] = None
        if obs_sequence is not None and action_seq is not None:
            self.set_sequence(obs_sequence, action_seq)

    # ------------------------------------------------------------------
    # Legacy single-sequence path (notebook-compatible)
    # ------------------------------------------------------------------

    def set_sequence(
        self,
        obs_sequence: Sequence[int],
        action_seq: Sequence[int],
    ) -> None:
        """Bind a fixed sequence to the model for legacy ``hmm_loss()`` use.

        Calling this with a new length will trigger one XLA recompile of
        :meth:`_forward_log` because ``maximum_iterations`` becomes a new
        constant. Prefer :meth:`batch_loss` if you need to evaluate many
        sequence lengths.
        """
        if len(obs_sequence) != len(action_seq):
            raise ValueError(
                f"obs/action length mismatch: {len(obs_sequence)} vs {len(action_seq)}"
            )
        self.sequence_length = int(len(obs_sequence))
        self.O = tf.constant(obs_sequence, dtype=tf.int32)
        self.A = tf.constant(action_seq, dtype=tf.int32)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _forward_log(self) -> tf.Tensor:
        """XLA-compiled forward pass over the bound single sequence.
        
        """
        T = tf.shape(self.O)[0] # type: ignore
        eps = tf.constant(1e-9, dtype=tf.float32)

        log_pi = tf.math.log(tf.nn.softmax(self.pi) + eps)
        log_T = tf.math.log(tf.nn.softmax(self.transition, axis=2) + eps)
        time_of_sta = self.time_of_sta

        mask_0 = tf.equal(time_of_sta[tf.newaxis, :], self.O[0, tf.newaxis]) # type: ignore
        alpha = tf.where(mask_0[0], log_pi, NEG_INF * tf.ones_like(log_pi))

        def cond(t, alpha):
            return t < T - 1

        def body(t, alpha):
            a_t = self.A[t] # type: ignore
            log_T_at = log_T[a_t]
            log_B = tf.where(
                tf.equal(time_of_sta, self.O[t + 1]), # type: ignore
                tf.zeros_like(time_of_sta, dtype=tf.float32),
                NEG_INF * tf.ones_like(time_of_sta, dtype=tf.float32),
            )
            alpha_next = tf.reduce_logsumexp(
                alpha[:, tf.newaxis] + log_T_at + log_B[tf.newaxis, :], axis=0
            )
            return (t + 1, alpha_next)

        _, final_alpha = tf.while_loop(
            cond,
            body,
            (tf.constant(0, dtype=tf.int32), alpha),
            parallel_iterations=1,
            maximum_iterations=self.sequence_length - 1, # type: ignore
        )
        log_likelihood = tf.reduce_logsumexp(final_alpha * 10)
        return tf.cast(-log_likelihood, dtype=tf.float32)

    def hmm_loss(self, _unused: object = None) -> tf.Tensor:
        """Negative log-likelihood of the bound single sequence."""
        if self.O is None or self.A is None:
            raise RuntimeError(
                "No bound sequence; call set_sequence() first or use batch_loss()."
            )
        return self._forward_log()

    # ------------------------------------------------------------------
    # Batched path
    # ------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def _batch_neg_log_likelihood(
        self, O: tf.Tensor, A: tf.Tensor, L: tf.Tensor
    ) -> tf.Tensor:
        """Graph-mode batched forward pass.

        Note
        ----
        This is **not** ``jit_compile=True`` so it remains usable from an
        eager :class:`tf.GradientTape`. When called from inside an outer
        ``@tf.function(jit_compile=True)`` (e.g. the training step in
        :class:`models.vqvae_cscg.VQVAEGradientHMM`), XLA still compiles the
        full forward+backward — so there's no XLA performance loss for the
        production training path. Only direct eager tape usage runs without
        XLA.

        Parameters
        ----------
        O : int32 ``[B, T]``
            Padded observation sequences. Padded entries are ignored via ``L``.
        A : int32 ``[B, T]``
            Padded action sequences (same length as ``O``).
        L : int32 ``[B]``
            True (unpadded) lengths, ``>= 1``.

        Returns
        -------
        scalar float32 mean NLL across the batch.
        """
        eps = tf.constant(1e-9, dtype=tf.float32)
        T = tf.shape(O)[1]

        log_pi = tf.math.log(tf.nn.softmax(self.pi) + eps)  # [N]
        log_T = tf.math.log(tf.nn.softmax(self.transition, axis=2) + eps)  # [A, N, N]

        first_obs = O[:, 0]  # [B]
        compat0 = tf.equal(
            self.time_of_sta[tf.newaxis, :], first_obs[:, tf.newaxis]
        )  # [B, N]
        log_B0 = tf.where(
            compat0,
            tf.zeros_like(compat0, dtype=tf.float32),
            tf.fill(tf.shape(compat0), NEG_INF),
        )
        alpha = log_pi[tf.newaxis, :] + log_B0  # [B, N]

        def cond(t, alpha):
            return t < T - 1

        def body(t, alpha):
            a_t = A[:, t]  # [B]
            log_T_bt = tf.gather(log_T, a_t, axis=0)  # [B, N, N]
            next_obs = O[:, t + 1]  # [B]
            compat_next = tf.equal(
                self.time_of_sta[tf.newaxis, :], next_obs[:, tf.newaxis]
            )
            log_B_next = tf.where(
                compat_next,
                tf.zeros_like(compat_next, dtype=tf.float32),
                tf.fill(tf.shape(compat_next), NEG_INF),
            )
            s = (
                alpha[:, :, tf.newaxis]
                + log_T_bt
                + log_B_next[:, tf.newaxis, :]
            )  # [B, N, N]
            alpha_next = tf.reduce_logsumexp(s, axis=1)  # [B, N]

            active = tf.cast(t < (L - 1), tf.float32)[:, tf.newaxis]  # [B, 1]
            alpha = alpha_next * active + alpha * (1.0 - active)
            return (t + 1, alpha)

        _, alpha_T = tf.while_loop(
            cond,
            body,
            (tf.constant(0, tf.int32), alpha),
            parallel_iterations=1,
            maximum_iterations=T - 1,
        )

        log_liks = tf.reduce_logsumexp(alpha_T, axis=1)  # [B]
        return -tf.reduce_mean(log_liks)

    @tf.function(reduce_retracing=True)
    def _batch_neg_log_likelihood_soft(
        self,
        log_p: tf.Tensor,
        A: tf.Tensor,
        L: tf.Tensor,
        length_normalize: bool = False,
    ) -> tf.Tensor:
        """Soft-emission batched forward pass — Phase 2.

        Replaces the deterministic ``B[s, o] ∈ {0, NEG_INF}`` with a
        per-timestep log-posterior over tokens supplied by the caller (e.g.
        the VQ-VAE encoder). The forward recurrence becomes::

            log α[t+1, j] = logsumexp_i (log α[t, i] + log T[a, i, j])
                            + log p[t+1, time_of_sta[j]]

        This is fully differentiable in ``log_p``, which is what unlocks
        gradients flowing from the HMM into the perceptual front-end. Setting
        ``log_p`` to a one-hot encoding of the hard observations recovers
        :meth:`_batch_neg_log_likelihood` to within ``log(1+eps)``.

        Parameters
        ----------
        log_p : float32 ``[B, T, K]``
            Per-timestep log probabilities over the K-symbol observation
            alphabet. Should be in log-space (i.e. ``log_softmax``-shaped),
            but only relative magnitudes matter for the forward pass.
        A : int32 ``[B, T]``
            Padded action sequences. ``A[b, t]`` is the action taken between
            time ``t`` and ``t+1``.
        L : int32 ``[B]``
            True (unpadded) lengths, ``>= 1``.

        Returns
        -------
        scalar float32 mean NLL across the batch.
        """
        T = tf.shape(log_p)[1]
        K = tf.shape(log_p)[2]
        eps = tf.constant(1e-9, dtype=tf.float32)

        # Pad log_p with one extra column of NEG_INF for the sink state, so
        # we can index by `time_of_sta` directly. Sink's time_of_sta value is
        # `num_states` (>K), which we clip to K to point at the padding.
        pad = tf.fill([tf.shape(log_p)[0], T, 1], NEG_INF)
        log_p_padded = tf.concat([log_p, pad], axis=2)            # [B, T, K+1]
        state_obs_idx = tf.minimum(self.time_of_sta, K)            # [N]
        # Reorder: gather along the K-axis so log_p_em[b, t, j] = soft prob
        # of the observation that state j is compatible with.
        log_p_em = tf.gather(log_p_padded, state_obs_idx, axis=2)  # [B, T, N]

        log_pi = tf.math.log(tf.nn.softmax(self.pi) + eps)                  # [N]
        log_T = tf.math.log(tf.nn.softmax(self.transition, axis=2) + eps)    # [A, N, N]

        alpha = log_pi[tf.newaxis, :] + log_p_em[:, 0, :]          # [B, N]

        def cond(t, alpha):
            return t < T - 1

        def body(t, alpha):
            a_t = A[:, t]                                          # [B]
            log_T_bt = tf.gather(log_T, a_t, axis=0)               # [B, N, N]
            log_p_em_next = log_p_em[:, t + 1, :]                  # [B, N]
            s = (
                alpha[:, :, tf.newaxis]
                + log_T_bt
                + log_p_em_next[:, tf.newaxis, :]
            )                                                      # [B, N, N]
            alpha_next = tf.reduce_logsumexp(s, axis=1)            # [B, N]
            active = tf.cast(t < (L - 1), tf.float32)[:, tf.newaxis]
            alpha = alpha_next * active + alpha * (1.0 - active)
            return (t + 1, alpha)

        _, alpha_T = tf.while_loop(
            cond, body,
            (tf.constant(0, tf.int32), alpha),
            parallel_iterations=1,
            maximum_iterations=T - 1,
        )
        log_liks = tf.reduce_logsumexp(alpha_T, axis=1)            # [B]
        if length_normalize:
            return -tf.reduce_mean(log_liks / tf.cast(L, tf.float32))
        return -tf.reduce_mean(log_liks)

    def batch_loss_soft(
        self,
        log_p: tf.Tensor | np.ndarray,
        A: tf.Tensor | np.ndarray,
        L: Optional[tf.Tensor | np.ndarray] = None,
        length_normalize: bool = False,
    ) -> tf.Tensor:
        """Mean batched negative log-likelihood under soft emissions.

        Parameters
        ----------
        log_p
            ``[B, T, K]`` per-timestep log probabilities over the
            observation alphabet.
        A
            ``[B, T]`` padded actions.
        L
            Optional true lengths ``[B]``.
        length_normalize
            If ``True``, return the mean **per-step** NLL instead of mean
            per-sequence. Useful for joint training where the recon term has
            ``O(1)`` magnitude and the unnormalized HMM NLL has ``O(T)``
            magnitude — without normalization the HMM term dominates by a
            factor of ``T`` and pushes the codebook to collapse. Default
            ``False`` to preserve Phase 1 / Phase 2 behaviour.
        """
        log_p = tf.convert_to_tensor(log_p, dtype=tf.float32)
        A = tf.convert_to_tensor(A, dtype=tf.int32)
        if L is None:
            L = tf.fill([tf.shape(log_p)[0]], tf.shape(log_p)[1])
        else:
            L = tf.convert_to_tensor(L, dtype=tf.int32)
        return self._batch_neg_log_likelihood_soft(log_p, A, L, length_normalize) # type: ignore

    def batch_loss(
        self,
        O: tf.Tensor | np.ndarray,
        A: tf.Tensor | np.ndarray,
        L: Optional[tf.Tensor | np.ndarray] = None,
    ) -> tf.Tensor:
        """Mean batched negative log-likelihood.

        Parameters
        ----------
        O, A
            Padded observation/action tensors of shape ``[B, T]``.
        L
            Optional true lengths ``[B]``. If ``None``, all sequences are
            assumed to be full length ``T``.
        """
        O = tf.convert_to_tensor(O, dtype=tf.int32)
        A = tf.convert_to_tensor(A, dtype=tf.int32)
        if L is None:
            L = tf.fill([tf.shape(O)[0]], tf.shape(O)[1])
        else:
            L = tf.convert_to_tensor(L, dtype=tf.int32)
        return self._batch_neg_log_likelihood(O, A, L) # type: ignore

    # ------------------------------------------------------------------
    # Decoding (Viterbi, eager)
    # ------------------------------------------------------------------

    def decode(
        self,
        x: Sequence[int] | np.ndarray | tf.Tensor,
        a: Sequence[int] | np.ndarray | tf.Tensor,
    ) -> Tuple[float, np.ndarray]:
        """MAP latent-state sequence via Viterbi.

        Returns ``(neg_log_prob_of_path, states_array)``. Eager — uses Python
        loops and is not XLA-compiled. Suitable for evaluation on a single
        sequence at a time.
        """
        x = tf.constant(x, dtype=tf.int32)
        a = tf.constant(a, dtype=tf.int32)

        T_len = int(tf.shape(x)[0])
        eps = tf.constant(1e-9, dtype=tf.float32)

        log_pi = tf.math.log(tf.nn.softmax(self.pi) + eps)
        log_T = tf.math.log(tf.nn.softmax(self.transition, axis=2) + eps)
        time_of_sta = self.time_of_sta

        mask_0 = tf.equal(time_of_sta, x[0])
        delta = tf.where(mask_0, log_pi, NEG_INF * tf.ones_like(log_pi))

        backpointers: List[tf.Tensor] = []
        for t in range(T_len - 1):
            log_T_at = log_T[a[t]]
            log_B = tf.where(
                tf.equal(time_of_sta, x[t + 1]),
                tf.zeros_like(time_of_sta, dtype=tf.float32),
                NEG_INF * tf.ones_like(time_of_sta, dtype=tf.float32),
            )
            scores = delta[:, tf.newaxis] + log_T_at  # [N, N]
            psi = tf.argmax(scores, axis=0, output_type=tf.int32)
            delta = tf.reduce_max(scores, axis=0) + log_B
            backpointers.append(psi)

        states: List[tf.Tensor] = []
        last_state = tf.argmax(delta, output_type=tf.int32)
        states.append(last_state)
        for t in reversed(range(T_len - 1)):
            last_state = backpointers[t][last_state]
            states.append(last_state)
        states_t = tf.stack(states[::-1])
        log_prob = tf.reduce_max(delta)
        return float(-log_prob.numpy()), states_t.numpy()


# ----------------------------------------------------------------------
# Helpers (kept here so the demos and tests stay short)
# ----------------------------------------------------------------------


def pad_and_stack(
    seqs: Sequence[Sequence[int]],
    pad_value: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Right-pad a list of 1-D int sequences to a common length.

    Returns ``(padded[B, T], lengths[B])`` as ``int32`` numpy arrays.
    """
    lengths = np.asarray([len(s) for s in seqs], dtype=np.int32)
    if lengths.size == 0:
        return np.zeros((0, 0), dtype=np.int32), lengths
    T_max = int(lengths.max())
    padded = np.full((len(seqs), T_max), pad_value, dtype=np.int32)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = np.asarray(s, dtype=np.int32)
    return padded, lengths


def chunk_sequences(
    seqs: Sequence[np.ndarray],
    other: Sequence[np.ndarray],
    chunk_size: int,
    min_chunk: int = 2,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Cut paired ``(primary, other)`` sequences into windows of ``chunk_size``.

    Each pair must have the same first dimension. Chunks shorter than
    ``min_chunk`` are dropped (the HMM forward needs ``T >= 1`` and a
    meaningful gradient needs ``T >= 2``).

    Useful for joint training where keeping a whole 10k-step episode in
    GPU memory is impossible (the encoder runs over every frame).
    """
    if len(seqs) != len(other):
        raise ValueError("seqs and other must have equal length")
    out_a: List[np.ndarray] = []
    out_b: List[np.ndarray] = []
    for a, b in zip(seqs, other):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape[0] != b.shape[0]:
            raise ValueError("paired sequences must have matching length on axis 0")
        T = a.shape[0]
        for start in range(0, T, chunk_size):
            stop = min(start + chunk_size, T)
            if stop - start < min_chunk:
                continue
            out_a.append(a[start:stop])
            out_b.append(b[start:stop])
    return out_a, out_b


def make_dataset(
    obs_seqs: Sequence[Sequence[int]],
    act_seqs: Sequence[Sequence[int]],
    batch_size: int = 8,
    shuffle: bool = True,
    pad_obs_value: int = 0,
    pad_act_value: int = 0,
) -> tf.data.Dataset:
    """Build a ``tf.data.Dataset`` yielding ``(O, A, L)`` triples per batch."""
    if len(obs_seqs) != len(act_seqs):
        raise ValueError("obs_seqs and act_seqs must have the same length")
    O_pad, L = pad_and_stack(obs_seqs, pad_obs_value)
    A_pad, _ = pad_and_stack(act_seqs, pad_act_value)
    ds = tf.data.Dataset.from_tensor_slices(
        (
            tf.constant(O_pad, dtype=tf.int32),
            tf.constant(A_pad, dtype=tf.int32),
            tf.constant(L, dtype=tf.int32),
        )
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(obs_seqs), reshuffle_each_iteration=True)
    return ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


def train_hmm(
    hmm: GradientHMM,
    train_step,
    n_iters: int,
    print_every: int = 1000,
    iter_inputs: Optional[Iterable] = None,
) -> Tuple[List[float], List[float]]:
    """Generic training loop.

    ``train_step`` is a user-supplied callable. If ``iter_inputs`` is provided,
    each iteration calls ``train_step(next(iter_inputs))``; otherwise
    ``train_step()`` is called with no arguments (legacy single-sequence
    style).
    """
    history: List[float] = []
    iter_times: List[float] = []
    started = time.perf_counter()
    inputs = iter(iter_inputs) if iter_inputs is not None else None

    for i in range(1, n_iters + 1):
        t0 = time.perf_counter()
        if inputs is None:
            loss = train_step()
        else:
            try:
                batch = next(inputs)
            except StopIteration:
                inputs = iter(iter_inputs)  # type: ignore[arg-type]
                batch = next(inputs)
            loss = train_step(batch)
        dt = time.perf_counter() - t0
        iter_times.append(dt)
        history.append(float(loss.numpy()) if hasattr(loss, "numpy") else float(loss))

        if i == 1 or i % print_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[{i:5d}/{n_iters}] loss={history[-1]:.6f} "
                f"iter={dt*1000:.1f}ms elapsed={elapsed:.1f}s"
            )

    return history, iter_times
