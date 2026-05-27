"""
VQVAEGradientHMM — joint wrapper for VQ-VAE + GradientHMM (Phase 1).

This module ties the perceptual front-end (:class:`models.vqvae.VQVAE`) to the
sequence model (:class:`models.gradient_hmm.GradientHMM`) by treating the
VQ-VAE codebook indices as the observation alphabet of the HMM.

Pipeline (stagewise training, Option 1 — single token per image):

    images ──► VQ-VAE ──► token indices ──► GradientHMM
    (B, H, W, C)            (B,) int32       (B, T) int32

The wrapper does **not** modify either underlying model — it just orchestrates
training and provides utilities (encoding, decoding, reconstruction,
evaluation).

Phase 2 adds :meth:`VQVAEGradientHMM.fit_joint` for end-to-end optimization
of ``L = recon + β·commit + λ·hmm_nll_soft``. The HMM's soft-emission path
allows gradients to flow from the sequence loss back into the encoder, which
pressures the codebook toward stable per-state assignments.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from .gradient_hmm import GradientHMM, chunk_sequences, make_dataset
from .vqvae import VQVAE


_METADATA_FILE = "metadata.json"
_VQVAE_WEIGHTS = "vqvae.weights.h5"
_HMM_CKPT_PREFIX = "hmm.ckpt"
_REMAP_FILE = "remap.npz"


@dataclass
class TrainHistory:
    """Container returned by :meth:`VQVAEGradientHMM.fit` /
    :meth:`VQVAEGradientHMM.fit_joint`."""

    vqvae_history: Dict[str, List[float]] = field(default_factory=dict)
    hmm_loss_history: List[float] = field(default_factory=list)
    # Phase 2 — populated only by ``fit_joint``. Keys: total, recon, commit, hmm.
    joint_history: Dict[str, List[float]] = field(default_factory=dict)
    # Phase 2.5+ — populated by ``finalize_hmm`` (encoder frozen, hard tokens).
    finalize_history: List[float] = field(default_factory=list)
    # Phase 2.5+++ — per-step finalize NLL, comparable to joint's hmm_term.
    finalize_per_step_history: List[float] = field(default_factory=list)
    perplexity: Optional[float] = None
    used_tokens: int = 0


class VQVAEGradientHMM:
    """End-to-end perceptual sequence learner.

    Parameters
    ----------
    image_shape
        Input shape ``(H, W, C)``.
    codebook_size
        Number of discrete tokens ``K``. Becomes the HMM's observation
        alphabet size, after compaction (see ``compact_tokens``).
    embedding_dim
        VQ-VAE embedding dimension ``D``.
    num_clones
        Clones per token in the HMM.
    num_actions
        Size of the action alphabet for the HMM.
    commitment_beta, ema_decay
        VQ hyperparameters.
    base_filters
        Encoder/decoder width.
    compact_tokens
        If ``True`` (default), unused codebook entries are pruned before
        building the HMM. The HMM's observation alphabet is the set of tokens
        actually emitted by the trained VQ-VAE; the wrapper transparently
        remaps integer ids.
    clone_counts
        Optional fixed nonuniform clone counts indexed by HMM observation/token
        id. If omitted, every observation receives ``num_clones`` clones,
        matching the original behavior. The MNIST demo instead passes
        semantic grid-label counts through ``observation_clone_counts`` in the
        fit methods, which resolves labels to arbitrary VQ token ids after
        VQ-VAE warmup.
    seed
        Optional RNG seed.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (28, 28, 1),
        codebook_size: int = 64,
        embedding_dim: int = 32,
        num_clones: int = 10,
        num_actions: int = 4,
        commitment_beta: float = 0.25,
        ema_decay: float = 0.99,
        base_filters: int = 32,
        compact_tokens: bool = True,
        dead_code_threshold: float = 0.0,
        clone_counts: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        self.image_shape = tuple(image_shape)
        self.codebook_size = int(codebook_size)
        self.embedding_dim = int(embedding_dim)
        self.num_clones = int(num_clones)
        self.num_actions = int(num_actions)
        self.compact_tokens = bool(compact_tokens)
        # Stored so save()/load() can faithfully reconstruct the VQ-VAE.
        self.commitment_beta = float(commitment_beta)
        self.ema_decay = float(ema_decay)
        self.base_filters = int(base_filters)
        self.dead_code_threshold = float(dead_code_threshold)
        if clone_counts is None:
            self.clone_counts_config: Optional[List[int]] = None
        else:
            clone_counts_arr = np.asarray(clone_counts, dtype=np.int32)
            if clone_counts_arr.ndim != 1:
                raise ValueError("clone_counts must be a 1-D sequence")
            if clone_counts_arr.size == 0:
                raise ValueError("clone_counts must not be empty")
            if (clone_counts_arr <= 0).any():
                raise ValueError("clone_counts entries must be positive")
            self.clone_counts_config = [int(x) for x in clone_counts_arr.tolist()]

        self.vqvae = VQVAE(
            input_shape=self.image_shape,
            embedding_dim=self.embedding_dim,
            num_embeddings=self.codebook_size,
            commitment_beta=self.commitment_beta,
            decay=self.ema_decay,
            base_filters=self.base_filters,
            dead_code_threshold=self.dead_code_threshold,
        )

        # HMM is built lazily in fit_hmm() once we know the *active*
        # vocabulary size (after VQ-VAE training).
        self.hmm: Optional[GradientHMM] = None
        self._token_remap: Optional[np.ndarray] = None  # K -> compact id (or -1)
        self._token_unmap: Optional[np.ndarray] = None  # compact id -> K
        self.active_vocab_size: int = self.codebook_size
        self.active_clone_counts: np.ndarray = np.full(
            (self.active_vocab_size,), self.num_clones, dtype=np.int32
        )
        self.clone_count_token_labels: Optional[List[Optional[int]]] = None

    # ------------------------------------------------------------------
    # Stage 1: VQ-VAE
    # ------------------------------------------------------------------
    def fit_vqvae(
        self,
        images: np.ndarray,
        labels: Optional[np.ndarray] = None,
        supervision_weight: float = 0.0,
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 3e-4,
        validation_split: float = 0.0,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """Train the VQ-VAE on raw images. Returns a Keras-style history.

        If ``labels`` and a positive ``supervision_weight`` are supplied, a
        temporary classifier head is trained from the encoder latent. This is a
        benchmark-only way to encourage the "eye" to encode visual class
        identity rather than handwriting-instance quirks. The classifier is
        discarded after warmup and is not part of the saved model.
        """
        images = np.asarray(images, dtype=np.float32)
        if images.ndim != 4:
            raise ValueError(
                f"images must be [B, H, W, C], got shape {images.shape}"
            )

        if labels is not None and float(supervision_weight) > 0.0:
            labels = np.asarray(labels, dtype=np.int32)
            if labels.shape[0] != images.shape[0]:
                raise ValueError(
                    "labels must have the same first dimension as images: "
                    f"{labels.shape[0]} vs {images.shape[0]}"
                )
            if not self.vqvae.built:
                _ = self.vqvae(
                    tf.zeros((1,) + self.image_shape, dtype=tf.float32),
                    training=False,
                )

            num_classes = int(labels.max()) + 1
            classifier = tf.keras.Sequential(
                [
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(num_classes),
                ],
                name="temporary_vqvae_supervision_head",
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate)
            ds = tf.data.Dataset.from_tensor_slices(
                (
                    tf.constant(images, dtype=tf.float32),
                    tf.constant(labels, dtype=tf.int32),
                )
            )
            ds = ds.shuffle(
                buffer_size=images.shape[0], reshuffle_each_iteration=True
            ).batch(batch_size, drop_remainder=False)

            history: Dict[str, List[float]] = {
                "total_loss": [],
                "recon_loss": [],
                "vq_loss": [],
                "class_loss": [],
                "perplexity": [],
            }
            for epoch in range(epochs):
                totals: List[float] = []
                recons: List[float] = []
                commits: List[float] = []
                class_losses: List[float] = []
                perps: List[float] = []
                for x_batch, y_batch in ds:
                    with tf.GradientTape() as tape:
                        out = self.vqvae(x_batch, training=True)
                        recon_loss = tf.reduce_mean((x_batch - out["recon"]) ** 2)
                        commit_loss = out["commitment_loss"]
                        logits = classifier(out["z_e"], training=True)
                        class_loss = tf.reduce_mean(
                            tf.keras.losses.sparse_categorical_crossentropy(
                                y_batch, logits, from_logits=True
                            )
                        )
                        total_loss = (
                            recon_loss
                            + commit_loss
                            + float(supervision_weight) * class_loss
                        )

                    trainable = (
                        self.vqvae.encoder.trainable_variables
                        + self.vqvae.decoder.trainable_variables
                        + classifier.trainable_variables
                    )
                    grads = tape.gradient(total_loss, trainable)
                    grad_vars = [
                        (g, v) for g, v in zip(grads, trainable) if g is not None
                    ]
                    optimizer.apply_gradients(grad_vars)

                    totals.append(float(total_loss.numpy()))
                    recons.append(float(recon_loss.numpy()))
                    commits.append(float(commit_loss.numpy()))
                    class_losses.append(float(class_loss.numpy()))
                    perps.append(float(out["perplexity"].numpy()))

                history["total_loss"].append(float(np.mean(totals)))
                history["recon_loss"].append(float(np.mean(recons)))
                history["vq_loss"].append(float(np.mean(commits)))
                history["class_loss"].append(float(np.mean(class_losses)))
                history["perplexity"].append(float(np.mean(perps)))
                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{epochs} "
                        f"- total_loss: {history['total_loss'][-1]:.4f} "
                        f"- recon_loss: {history['recon_loss'][-1]:.4f} "
                        f"- vq_loss: {history['vq_loss'][-1]:.4f} "
                        f"- class_loss: {history['class_loss'][-1]:.4f} "
                        f"- perplexity: {history['perplexity'][-1]:.4f}"
                    )
            return history

        self.vqvae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
        hist = self.vqvae.fit(
            images,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose, # type: ignore
        )
        return {k: list(map(float, v)) for k, v in hist.history.items()}

    # ------------------------------------------------------------------
    # Encoding utilities
    # ------------------------------------------------------------------
    def encode_images(self, images: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Encode raw images to **raw** codebook ids (0..K-1)."""
        return self.vqvae.encode_tokens_numpy(
            np.asarray(images, dtype=np.float32), batch_size=batch_size
        )

    def encode_sequences(
        self,
        image_sequences: Sequence[np.ndarray],
        batch_size: int = 256,
    ) -> List[np.ndarray]:
        """Encode a list of per-episode image arrays.

        Each entry of ``image_sequences`` should be shaped ``[T, H, W, C]``.
        Returns a list of int32 token sequences ``[T]``. The token ids are in
        the *compacted* alphabet if :attr:`compact_tokens` is ``True`` and
        :meth:`fit_hmm` has already been called; otherwise raw codebook ids.
        """
        out: List[np.ndarray] = []
        for ep in image_sequences:
            tokens = self.encode_images(ep, batch_size=batch_size)
            if self.compact_tokens and self._token_remap is not None:
                tokens = self._token_remap[tokens]
            out.append(tokens.astype(np.int32))
        return out

    def _build_token_compaction(self, all_tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute remap arrays from the observed token set."""
        used = np.unique(all_tokens)
        unmap = used.astype(np.int32)  # compact -> original
        remap = -np.ones((self.codebook_size,), dtype=np.int32)  # original -> compact
        remap[used] = np.arange(len(used), dtype=np.int32)
        return remap, unmap

    def _clone_counts_for_vocab(
        self,
        vocab_size: int,
        token_unmap: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return fixed clone counts for the current HMM observation alphabet."""
        if self.clone_counts_config is None:
            return np.full((int(vocab_size),), self.num_clones, dtype=np.int32)

        configured = np.asarray(self.clone_counts_config, dtype=np.int32)
        if configured.shape[0] != self.codebook_size:
            raise ValueError(
                "clone_counts must contain one entry per raw observation/token "
                f"id in the codebook ({self.codebook_size}); got "
                f"{configured.shape[0]}"
            )
        if token_unmap is not None:
            counts = configured[np.asarray(token_unmap, dtype=np.int32)]
        else:
            counts = configured
        if (counts <= 0).any():
            raise ValueError("clone_counts entries must be positive")
        return counts.astype(np.int32)

    def set_clone_counts_from_observation_labels(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        clone_counts: Sequence[int],
        batch_size: int = 256,
        verbose: int = 1,
    ) -> np.ndarray:
        """Resolve semantic observation clone counts onto VQ token ids.

        ``clone_counts[i]`` refers to external observation label ``i`` (for the
        MNIST grid demo, the digit in the physical grid), not the arbitrary VQ
        codebook index. Each VQ token is assigned the majority external label
        among the supplied images, then receives that label's clone count.
        """
        labels = np.asarray(labels, dtype=np.int32)
        images = np.asarray(images, dtype=np.float32)
        counts_by_label = np.asarray(clone_counts, dtype=np.int32)
        if counts_by_label.ndim != 1 or counts_by_label.size == 0:
            raise ValueError("clone_counts must be a non-empty 1-D sequence")
        if (counts_by_label <= 0).any():
            raise ValueError("clone_counts entries must be positive")
        if images.shape[0] != labels.shape[0]:
            raise ValueError(
                "images and labels must align when resolving clone counts: "
                f"{images.shape[0]} vs {labels.shape[0]}"
            )
        if labels.size and (labels < 0).any():
            raise ValueError("observation labels for clone counts must be non-negative")
        if labels.size and int(labels.max()) >= counts_by_label.shape[0]:
            raise ValueError(
                "clone_counts must include every observation label seen in "
                f"the data; max label={int(labels.max())}, "
                f"len={counts_by_label.shape[0]}"
            )

        tokens = self.encode_images(images, batch_size=batch_size)
        token_counts = np.full((self.codebook_size,), self.num_clones, dtype=np.int32)
        token_labels: List[Optional[int]] = [None] * self.codebook_size
        for token in np.unique(tokens):
            mask = tokens == int(token)
            label_hist = np.bincount(labels[mask], minlength=counts_by_label.shape[0])
            majority_label = int(np.argmax(label_hist))
            token_counts[int(token)] = int(counts_by_label[majority_label])
            token_labels[int(token)] = majority_label

        self.clone_counts_config = [int(x) for x in token_counts.tolist()]
        self.clone_count_token_labels = token_labels
        if verbose:
            used = {
                int(tok): {
                    "label": token_labels[int(tok)],
                    "clones": int(token_counts[int(tok)]),
                }
                for tok in np.unique(tokens)
            }
            print(f"[clone-counts] resolved observation-label counts to VQ tokens: {used}")
        return token_counts

    @staticmethod
    def _state_to_obs_from_clone_counts(clone_counts: np.ndarray) -> np.ndarray:
        """Build deterministic CSCG emissions for nonuniform clone counts."""
        clone_counts = np.asarray(clone_counts, dtype=np.int32)
        return np.repeat(np.arange(clone_counts.shape[0], dtype=np.int32), clone_counts)

    def _build_hmm_from_clone_counts(
        self,
        clone_counts: np.ndarray,
        seed: Optional[int] = None,
    ) -> GradientHMM:
        """Build an HMM with explicit state->observation mapping."""
        clone_counts = np.asarray(clone_counts, dtype=np.int32)
        state_to_obs = self._state_to_obs_from_clone_counts(clone_counts)
        return GradientHMM(
            num_states=int(state_to_obs.shape[0]) + 1,
            num_actions=self.num_actions,
            num_clones=self.num_clones,
            seed=seed,
            state_to_obs=state_to_obs,
        )

    def _state_tokens_numpy(self, states: np.ndarray) -> np.ndarray:
        """Map latent state ids to compatible observation ids."""
        hmm = self._require_hmm()
        state_tokens = hmm.time_of_sta.numpy()
        return state_tokens[np.asarray(states, dtype=np.int64)]

    def _state_clone_indices_numpy(self, states: np.ndarray) -> np.ndarray:
        """Map latent state ids to clone index within each observation."""
        states = np.asarray(states, dtype=np.int64)
        clone_counts = np.asarray(self.active_clone_counts, dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(clone_counts[:-1])])
        state_tokens = self._state_tokens_numpy(states)
        out = np.full(states.shape, -1, dtype=np.int32)
        valid = state_tokens < clone_counts.shape[0]
        out[valid] = (states[valid] - starts[state_tokens[valid]]).astype(np.int32)
        return out

    # ------------------------------------------------------------------
    # In-memory state snapshot / restore (used by joint-mode anti-collapse)
    # ------------------------------------------------------------------
    def _snapshot_state(self) -> Dict[str, object]:
        """Capture all trainable state (encoder, decoder, codebook + EMA, HMM)
        as numpy arrays. Used to roll back to a non-collapsed checkpoint if
        joint training degenerates.
        """
        if not self.vqvae.built:
            _ = self.vqvae(tf.zeros((1,) + self.image_shape), training=False)
        snap: Dict[str, object] = {
            "encoder": [v.numpy() for v in self.vqvae.encoder.trainable_variables],
            "decoder": [v.numpy() for v in self.vqvae.decoder.trainable_variables],
            "codebook": self.vqvae.quantizer.embeddings.numpy(),
            "ema_cluster_size": self.vqvae.quantizer.ema_cluster_size.numpy(),
            "ema_w": self.vqvae.quantizer.ema_w.numpy(),
        }
        if self.hmm is not None:
            snap["pi"] = self.hmm.pi.numpy()
            snap["transition"] = self.hmm.transition.numpy()
        return snap

    def _restore_state(self, snap: Dict[str, object]) -> None:
        for var, val in zip(self.vqvae.encoder.trainable_variables, snap["encoder"]):  # type: ignore[arg-type]
            var.assign(val)
        for var, val in zip(self.vqvae.decoder.trainable_variables, snap["decoder"]):  # type: ignore[arg-type]
            var.assign(val)
        self.vqvae.quantizer.embeddings.assign(snap["codebook"])
        self.vqvae.quantizer.ema_cluster_size.assign(snap["ema_cluster_size"])
        self.vqvae.quantizer.ema_w.assign(snap["ema_w"])
        if self.hmm is not None and "pi" in snap and "transition" in snap:
            self.hmm.pi.assign(snap["pi"])
            self.hmm.transition.assign(snap["transition"])

    # ------------------------------------------------------------------
    # Stage 2: HMM
    # ------------------------------------------------------------------
    def fit_hmm(
        self,
        image_sequences: Sequence[np.ndarray],
        action_sequences: Sequence[np.ndarray],
        n_iters: int = 5000,
        learning_rate: float = 1e-3,
        batch_size: int = 8,
        print_every: int = 500,
        seed: Optional[int] = None,
    ) -> List[float]:
        """Train the HMM on token sequences extracted from images.

        Builds the HMM lazily, sized to the **observed** token alphabet (after
        compaction if enabled). Stores the trained HMM on ``self.hmm``.
        """
        if len(image_sequences) != len(action_sequences):
            raise ValueError("image and action sequences must have equal length")

        # First pass: encode everything with raw codebook ids so we can
        # compute the active alphabet.
        raw_token_seqs: List[np.ndarray] = [
            self.encode_images(ep) for ep in image_sequences
        ]
        if self.compact_tokens:
            all_tokens = np.concatenate(raw_token_seqs)
            self._token_remap, self._token_unmap = self._build_token_compaction(
                all_tokens
            )
            token_seqs = [self._token_remap[s].astype(np.int32) for s in raw_token_seqs]
            self.active_vocab_size = int(self._token_unmap.shape[0])
        else:
            self._token_remap = None
            self._token_unmap = None
            token_seqs = [s.astype(np.int32) for s in raw_token_seqs]
            self.active_vocab_size = self.codebook_size

        action_seqs = [np.asarray(a, dtype=np.int32) for a in action_sequences]

        # Build HMM. Default is uniform clones; optional clone_counts keep
        # deterministic CSCG emissions with a nonuniform state allocation.
        self.active_clone_counts = self._clone_counts_for_vocab(
            self.active_vocab_size,
            token_unmap=self._token_unmap,
        )
        self.hmm = self._build_hmm_from_clone_counts(
            self.active_clone_counts,
            seed=seed,
        )

        # Build dataset and train
        ds = make_dataset(token_seqs, action_seqs, batch_size=batch_size, shuffle=True) # type: ignore
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        @tf.function(jit_compile=True, reduce_retracing=True)
        def train_step(batch):
            O, A, L = batch
            with tf.GradientTape() as tape:
                loss = self.hmm.batch_loss(O, A, L) # type: ignore
            grads = tape.gradient(loss, [self.hmm.pi, self.hmm.transition]) # type: ignore
            optimizer.apply_gradients(
                zip(grads, [self.hmm.pi, self.hmm.transition]) # type: ignore
            )
            return loss

        history: List[float] = []
        ds_iter = iter(ds)
        for i in range(1, n_iters + 1):
            try:
                batch = next(ds_iter)
            except StopIteration:
                ds_iter = iter(ds)
                batch = next(ds_iter)
            loss = train_step(batch)
            history.append(float(loss.numpy()))
            if i == 1 or i % print_every == 0:
                print(f"[hmm {i:5d}/{n_iters}] loss={history[-1]:.6f}")

        return history

    # ------------------------------------------------------------------
    # Phase 2 — joint training (encoder + decoder + HMM)
    # ------------------------------------------------------------------
    def fit_joint(
        self,
        image_sequences: Sequence[np.ndarray],
        action_sequences: Sequence[np.ndarray],
        n_iters: int = 5000,
        chunk_size: int = 256,
        batch_size: int = 4,
        learning_rate: float = 3e-4,
        lambda_hmm: float = 1.0,
        temperature: float = 1.0,
        commitment_weight: float = 1.0,
        # ---- Phase 2.5 loss-balancing knobs (defaults preserve Phase 2) ----
        length_normalize_hmm: bool = False,
        lambda_anneal_steps: int = 0,
        diversity_weight: float = 0.0,
        # ---- Phase 2.5+ HMM-side learning-rate boost (default = no boost) --
        hmm_lr_multiplier: float = 1.0,
        freeze_vqvae: bool = False,
        # ---- Phase 2.5++ anti-collapse safeguards (default off) ------------
        check_every: int = 100,
        check_images: Optional[np.ndarray] = None,
        save_best_by_perplexity: bool = False,
        early_stop_min_perplexity: float = 0.0,
        early_stop_patience: int = 3,
        # ---- Phase 2.5+++ active recovery from incipient collapse ----------
        lambda_throttle_floor: float = 0.0,
        lambda_throttle_factor: float = 0.5,
        rollback_min_perplexity: float = 0.0,
        # --------------------------------------------------------------------
        warmup_vqvae_images: Optional[np.ndarray] = None,
        warmup_epochs: int = 0,
        warmup_batch_size: int = 128,
        warmup_lr: float = 3e-4,
        seed: Optional[int] = None,
        print_every: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Joint training of VQ-VAE + HMM with the soft-emission HMM loss.

        Loss: ``L = recon_loss + commitment_weight · commit_loss
                  + lambda_hmm · hmm_nll_soft(log_p, A, L)``.

        Parameters
        ----------
        image_sequences, action_sequences
            Episodes. Each ``image_sequences[i]`` is ``[T_i, H, W, C]`` and
            the matching action sequence is ``[T_i]``. Episodes are split
            into chunks of ``chunk_size`` because the encoder must run over
            every frame in every chunk on every step.
        n_iters
            Number of joint optimisation steps.
        chunk_size
            Episode-chunk length used for the HMM forward pass.
        batch_size
            Number of chunks per step.
        learning_rate
            Adam LR for encoder + decoder + ``pi`` + ``transition``.
        lambda_hmm
            Weight on the soft-emission HMM NLL term. Try ``0.1`` – ``5``.
        temperature
            Softmax temperature for the codebook posterior. Lower ⇒ harder
            assignments. ``1.0`` is a good starting point.
        commitment_weight
            Multiplier on the VQ commitment loss. ``1.0`` keeps the
            commitment-loss magnitude as configured at construction time.
        length_normalize_hmm
            *Phase 2.5.* If ``True``, divide the per-sequence HMM NLL by the
            sequence length before adding to the total loss. Without this,
            the HMM term has magnitude ``O(T)`` while the recon term is
            ``O(1)``, so the HMM term dominates by a factor of ``T`` and
            collapses the codebook. Strongly recommended for joint training.
        lambda_anneal_steps
            *Phase 2.5.* If ``> 0``, the effective ``λ`` is linearly ramped
            from ``0`` at iter 1 up to ``lambda_hmm`` at iter
            ``lambda_anneal_steps``. Lets the codebook stabilize via recon
            before the HMM term turns on. ``0`` disables annealing.
        diversity_weight
            *Phase 2.5.* Coefficient on a usage-entropy penalty
            ``log(K) - H(mean p_t)`` that pushes the codebook toward uniform
            usage. Counteracts joint-training's tendency to collapse the
            codebook. ``0.0`` disables it. Try ``0.05 – 0.5``.
        hmm_lr_multiplier
            *Phase 2.5+.* HMM (``pi``, ``transition``) learning-rate scale
            relative to the encoder/decoder LR. Length-normalization shrinks
            the HMM gradient by a factor of ``T``, so the HMM optimizer sees
            a much weaker signal than the VQ-VAE optimizer. Setting this to
            ``T`` (or ~``100``) restores the HMM-side step size. ``1.0``
            disables the boost.
        check_every
            *Phase 2.5++.* How often (in iterations) to monitor codebook
            perplexity for ``save_best_by_perplexity`` / early-stopping.
        check_images
            *Phase 2.5++.* A small bank of images used to compute codebook
            perplexity at each check. If ``None``, monitoring is skipped.
        save_best_by_perplexity
            *Phase 2.5++.* If ``True``, snapshot all trainable state in
            memory whenever the monitored perplexity exceeds the best seen
            so far, and restore that snapshot when joint training ends. The
            single most important guardrail against late-stage collapse —
            even if training degenerates, the wrapper exits with the best
            non-collapsed model it observed.
        early_stop_min_perplexity
            *Phase 2.5++.* If non-zero, stop the joint phase early once
            perplexity drops below this floor for ``early_stop_patience``
            consecutive checks. ``0.0`` disables early-stopping.
        early_stop_patience
            *Phase 2.5++.* Consecutive bad checks required before
            early-stopping fires.
        lambda_throttle_floor
            *Phase 2.5+++.* If non-zero, every periodic perplexity check that
            comes back below this floor multiplies the effective ``λ`` by
            ``lambda_throttle_factor``. Recon-only pressure dominates while
            the codebook recovers, then the HMM ramps back. Strictly cooler
            than ``early_stop_min_perplexity``: throttling lets training
            continue, where early-stopping ends it.
        lambda_throttle_factor
            *Phase 2.5+++.* Multiplier applied to the effective λ each time
            ``lambda_throttle_floor`` triggers. ``0.5`` (default) halves λ;
            set to ``1.0`` to disable throttling without zeroing the floor.
        rollback_min_perplexity
            *Phase 2.5+++.* If non-zero and a snapshot exists, a perplexity
            check below this floor restores the best snapshot **in place**
            (not just at the end of training). Combined with throttling, this
            converts a collapse into a quick recovery: snap back to a healthy
            codebook and continue at a smaller HMM weight.
        warmup_vqvae_images, warmup_epochs
            If both supplied, the VQ-VAE alone is trained for ``warmup_epochs``
            epochs before joint optimisation begins. Strongly recommended:
            without warmup the codebook starts random and the HMM loss may
            dominate on the first few steps.
        seed
            Optional RNG seed.
        print_every
            Print frequency. Defaults to ``max(1, n_iters // 20)``.

        Returns
        -------
        dict with keys ``"total"``, ``"recon"``, ``"commit"``, ``"hmm"`` —
        one float per training step.

        Notes
        -----
        ``fit_joint`` rebuilds the HMM with the **full** codebook size
        (``num_states = num_clones × codebook_size + 1``); compaction is
        disabled because the active alphabet can shift during joint
        optimisation. Any HMM previously trained by :meth:`fit_hmm` is
        replaced.
        """
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        if len(image_sequences) != len(action_sequences):
            raise ValueError("image_sequences and action_sequences must align")

        if warmup_epochs > 0 and warmup_vqvae_images is not None:
            print(f"[joint] warming up VQ-VAE for {warmup_epochs} epoch(s)")
            self.fit_vqvae(
                warmup_vqvae_images,
                epochs=warmup_epochs,
                batch_size=warmup_batch_size,
                learning_rate=warmup_lr,
                verbose=1,
            )

        # Rebuild HMM at full codebook size; disable compaction.
        self.compact_tokens = False
        self._token_remap = None
        self._token_unmap = None
        self.active_vocab_size = self.codebook_size
        self.active_clone_counts = self._clone_counts_for_vocab(self.codebook_size)
        num_states = int(self.active_clone_counts.sum()) + 1
        print(
            f"[joint] (re)building HMM with num_states={num_states} "
            f"(clone_counts sum over K={self.codebook_size} + sink)"
        )
        self.hmm = self._build_hmm_from_clone_counts(
            self.active_clone_counts,
            seed=seed,
        )

        # Chunk sequences for memory.
        chunked_imgs, chunked_acts = chunk_sequences(
            image_sequences, action_sequences, chunk_size=chunk_size
        )
        if not chunked_imgs:
            raise ValueError(
                f"No usable chunks after chunk_size={chunk_size}; check that "
                "your episode lengths exceed the chunk size."
            )

        # Padded batch builder over chunks.
        def pad_image_chunks(imgs, acts):
            lengths = np.array([len(s) for s in imgs], dtype=np.int32)
            T_max = int(lengths.max())
            B = len(imgs)
            H, W, C = imgs[0].shape[1:]
            images = np.zeros((B, T_max, H, W, C), dtype=np.float32)
            actions = np.zeros((B, T_max), dtype=np.int32)
            for i, (im, ac) in enumerate(zip(imgs, acts)):
                images[i, : len(im)] = im
                actions[i, : len(ac)] = ac
            return images, actions, lengths

        all_imgs, all_acts, all_lens = pad_image_chunks(chunked_imgs, chunked_acts)
        ds = tf.data.Dataset.from_tensor_slices(
            (
                tf.constant(all_imgs, dtype=tf.float32),
                tf.constant(all_acts, dtype=tf.int32),
                tf.constant(all_lens, dtype=tf.int32),
            )
        )
        ds = (
            ds.shuffle(buffer_size=len(chunked_imgs), reshuffle_each_iteration=True)
            .batch(batch_size, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Two separate optimizers so the HMM can take larger steps than the
        # VQ-VAE without amplifying the encoder/decoder gradients (which are
        # already at their natural scale through recon).
        optimizer_vqvae = tf.keras.optimizers.Adam(learning_rate)
        optimizer_hmm = tf.keras.optimizers.Adam(
            learning_rate * float(hmm_lr_multiplier)
        )
        # Need an initial forward pass so vqvae.built becomes True before we
        # enumerate trainable variables.
        if not self.vqvae.built:
            _ = self.vqvae(tf.zeros((1,) + self.image_shape), training=False)

        H, W, C = self.image_shape
        log_K = float(np.log(self.codebook_size))

        # Phase 2.5: λ is varied across iterations, so it lives in a
        # tf.Variable so the @tf.function trace stays stable while we update
        # it from Python.
        lambda_var = tf.Variable(
            float(lambda_hmm) if lambda_anneal_steps <= 0 else 0.0, # type: ignore
            trainable=False,
            dtype=tf.float32,
            name="lambda_hmm",
        )
        diversity_w = tf.constant(float(diversity_weight), dtype=tf.float32)

        @tf.function(jit_compile=False, reduce_retracing=True)
        def joint_step(images, actions, lengths):
            # images: [B, T, H, W, C]
            B = tf.shape(images)[0]
            T_local = tf.shape(images)[1]
            flat = tf.reshape(images, (B * T_local, H, W, C))

            with tf.GradientTape() as tape:
                z_e = self.vqvae.encoder(
                    flat, training=not freeze_vqvae
                )  # [B*T, 1, 1, D]

                # Hard quant + decode (recon path with EMA codebook update)
                quant_st, idx, commit_loss, _perp = self.vqvae.quantizer(
                    z_e, training=not freeze_vqvae
                )
                recon = self.vqvae.decoder(quant_st, training=not freeze_vqvae)
                recon_loss = tf.reduce_mean((flat - recon) ** 2)

                # Soft log-probabilities over the codebook for the HMM
                log_p_flat = self.vqvae.quantizer.log_soft_assignments(
                    z_e, temperature=temperature
                )
                log_p_flat = tf.squeeze(log_p_flat, axis=[1, 2])      # [B*T, K]
                log_p = tf.reshape(log_p_flat, (B, T_local, self.codebook_size))

                # Raw HMM term (always recorded for diagnostic continuity).
                hmm_nll_raw = self.hmm.batch_loss_soft(log_p, actions, lengths) # type: ignore
                hmm_term = self.hmm.batch_loss_soft( # type: ignore
                    log_p, actions, lengths,
                    length_normalize=length_normalize_hmm,
                )

                # Diversity penalty: log K - H(mean p) ≥ 0; pushes codebook
                # toward uniform usage.
                if diversity_weight > 0.0:
                    mean_p = tf.reduce_mean(tf.exp(log_p_flat), axis=0)  # [K]
                    H_usage = -tf.reduce_sum(
                        mean_p * tf.math.log(mean_p + 1e-12)
                    )
                    diversity_pen = log_K - H_usage
                else:
                    diversity_pen = tf.constant(0.0, dtype=tf.float32)

                total = (
                    recon_loss
                    + commitment_weight * commit_loss
                    + lambda_var * hmm_term
                    + diversity_w * diversity_pen
                )

            trainable_vqvae = [] if freeze_vqvae else (
                self.vqvae.encoder.trainable_variables
                + self.vqvae.decoder.trainable_variables
            )
            trainable_hmm = [self.hmm.pi, self.hmm.transition] # type: ignore
            all_trainable = trainable_vqvae + trainable_hmm
            grads = tape.gradient(total, all_trainable)
            n_vqvae = len(trainable_vqvae)
            if n_vqvae:
                optimizer_vqvae.apply_gradients(
                    zip(grads[:n_vqvae], trainable_vqvae)
                )
            optimizer_hmm.apply_gradients(
                zip(grads[n_vqvae:], trainable_hmm)
            )
            return total, recon_loss, commit_loss, hmm_nll_raw, hmm_term, diversity_pen

        history: Dict[str, List[float]] = {
            "total": [], "recon": [], "commit": [],
            "hmm": [],            # raw mean-NLL-per-sequence (legacy meaning)
            "hmm_term": [],       # what actually entered the loss (per-step or raw)
            "diversity": [],      # log K - H(mean p)
            "lambda": [],         # effective λ at this step
            "throttle": [],       # throttle scalar in effect at this step
            "perplexity_iter": [],  # iteration index at which a check was done
            "perplexity": [],       # perplexity measurement at that check
        }
        if print_every is None:
            print_every = max(1, n_iters // 20)

        # Anti-collapse state
        monitoring = check_images is not None and (
            save_best_by_perplexity
            or early_stop_min_perplexity > 0
            or lambda_throttle_floor > 0
            or rollback_min_perplexity > 0
        )
        best_perp = -float("inf")
        best_iter = 0
        best_snapshot: Optional[Dict[str, object]] = None
        bad_perp_count = 0
        stopped_early_at: Optional[int] = None
        # Phase 2.5+++: scalar in [0, 1] that scales λ down each time a
        # perplexity check trips ``lambda_throttle_floor``. Multiplied into
        # the per-iter λ assign so the throttle survives the linear anneal.
        throttle_state = 1.0

        ds_iter = iter(ds)
        for i in range(1, n_iters + 1):
            # Phase 2.5: linear anneal of λ from 0 → lambda_hmm.
            # Phase 2.5+++: also multiply by the perplexity-driven throttle.
            if lambda_anneal_steps > 0:
                progress = min(1.0, (i - 1) / float(lambda_anneal_steps))
                lambda_var.assign(float(lambda_hmm) * progress * throttle_state)
            else:
                lambda_var.assign(float(lambda_hmm) * throttle_state)

            try:
                batch = next(ds_iter)
            except StopIteration:
                ds_iter = iter(ds)
                batch = next(ds_iter)
            total, recon, commit, hmm_nll_raw, hmm_term, diversity_pen = joint_step(*batch)

            history["total"].append(float(total.numpy()))
            history["recon"].append(float(recon.numpy()))
            history["commit"].append(float(commit.numpy()))
            history["hmm"].append(float(hmm_nll_raw.numpy()))
            history["hmm_term"].append(float(hmm_term.numpy()))
            history["diversity"].append(float(diversity_pen.numpy()))
            history["lambda"].append(float(lambda_var.numpy()))
            history["throttle"].append(throttle_state)

            if i == 1 or i % print_every == 0:
                print(
                    f"[joint {i:5d}/{n_iters}] "
                    f"total={history['total'][-1]:.4f} "
                    f"recon={history['recon'][-1]:.4f} "
                    f"commit={history['commit'][-1]:.4f} "
                    f"hmm={history['hmm'][-1]:.4f} "
                    f"hmm_term={history['hmm_term'][-1]:.4f} "
                    f"div={history['diversity'][-1]:.4f} "
                    f"lambda={history['lambda'][-1]:.4f}"
                )

            # Phase 2.5++ — codebook health check
            if monitoring and (i % max(1, check_every) == 0 or i == n_iters):
                usage = self.vqvae.codebook_usage(check_images)  # type: ignore[arg-type]
                p = usage / max(usage.sum(), 1)
                perp = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
                history["perplexity_iter"].append(float(i))
                history["perplexity"].append(perp)

                if save_best_by_perplexity and perp > best_perp:
                    best_perp = perp
                    best_iter = i
                    best_snapshot = self._snapshot_state()
                    print(
                        f"[joint {i:5d}/{n_iters}] best perplexity {perp:.3f} "
                        f"— snapshot saved"
                    )

                # Phase 2.5+++ — active recovery. Order matters: rollback
                # first (restores a known-good state), then throttle (so the
                # restored state isn't immediately re-collapsed by full λ).
                if (
                    rollback_min_perplexity > 0
                    and perp < float(rollback_min_perplexity)
                    and best_snapshot is not None
                ):
                    print(
                        f"[joint {i:5d}/{n_iters}] perplexity {perp:.3f} "
                        f"< rollback floor {rollback_min_perplexity}: "
                        f"restoring best snapshot from iter {best_iter} "
                        f"(perp={best_perp:.3f})"
                    )
                    self._restore_state(best_snapshot)
                    bad_perp_count = 0  # rollback resets the early-stop counter

                if (
                    lambda_throttle_floor > 0
                    and perp < float(lambda_throttle_floor)
                    and float(lambda_throttle_factor) < 1.0
                ):
                    throttle_state *= float(lambda_throttle_factor)
                    print(
                        f"[joint {i:5d}/{n_iters}] perplexity {perp:.3f} "
                        f"< throttle floor {lambda_throttle_floor}: "
                        f"lambda *= {lambda_throttle_factor} "
                        f"(scale now {throttle_state:.3g})"
                    )

                if early_stop_min_perplexity > 0:
                    if perp < float(early_stop_min_perplexity):
                        bad_perp_count += 1
                        print(
                            f"[joint {i:5d}/{n_iters}] perplexity {perp:.3f} "
                            f"< floor {early_stop_min_perplexity} "
                            f"({bad_perp_count}/{early_stop_patience})"
                        )
                        if bad_perp_count >= int(early_stop_patience):
                            stopped_early_at = i
                            print(
                                f"[joint] early-stopping at iter {i}: "
                                f"codebook collapse detected"
                            )
                            break
                    else:
                        bad_perp_count = 0

        # Restore best snapshot if requested
        if save_best_by_perplexity and best_snapshot is not None:
            self._restore_state(best_snapshot)
            print(
                f"[joint] restored best snapshot from iter {best_iter} "
                f"(perplexity={best_perp:.3f})"
            )
        history["best_perplexity"] = [best_perp] if best_snapshot is not None else []  # type: ignore[assignment]
        history["best_iter"] = [float(best_iter)] if best_snapshot is not None else []  # type: ignore[assignment]
        history["early_stopped_at"] = (
            [float(stopped_early_at)] if stopped_early_at is not None else []  # type: ignore[list-item]
        )

        return history

    # ------------------------------------------------------------------
    # Phase 2.5+ — pure-HMM finalization (encoder frozen, hard tokens)
    # ------------------------------------------------------------------
    def finalize_hmm(
        self,
        image_sequences: Sequence[np.ndarray],
        action_sequences: Sequence[np.ndarray],
        n_iters: int = 2000,
        learning_rate: float = 1e-2,
        batch_size: int = 8,
        chunk_size: int = 0,
        reset_pi: bool = False,
        lr_decay_to: float = 0.0,
        transition_entropy_weight: float = 0.0,
        print_every: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Refine the HMM after joint training, with the encoder frozen.

        Encodes every frame through the (now-stable) VQ-VAE to *hard* token
        indices, then runs gradient steps on ``pi`` and ``transition`` only,
        using :meth:`GradientHMM.batch_loss` (no length normalization). This
        recovers the regime where the HMM converges quickly — the slow tail
        of joint training is a side effect of length-normalized gradients
        applied to a moving codebook target.

        The wrapper's existing HMM is reused (no rebuild). Compaction is
        kept off (the wrapper is already in joint mode).

        Phase 2.5+++ extensions
        -----------------------
        chunk_size
            If ``> 0``, cut each episode into windows of this length before
            training (same helper used by joint phase). With long episodes
            (e.g. 10000 steps), training on the full sequence makes every
            gradient update expensive and over-smooths the signal — early
            uncertain steps get drowned out by the easy late-episode tail.
            Smaller chunks (try 512–1024) give more gradient updates per
            iter and reproduce the regime where the integer-only sanity
            demo converges. ``0`` (default) preserves the previous
            full-episode behaviour.
        reset_pi
            If ``True``, zero the initial-state distribution at the start
            of finalize. The joint-phase ``pi`` was optimised for chunk
            starts at random episode positions and is a poor prior for an
            actual episode beginning; a uniform restart converges to a
            sharper solution.
        lr_decay_to
            If ``> 0``, linearly decay the Adam learning rate from
            ``learning_rate`` at iter 1 down to ``lr_decay_to`` at iter
            ``n_iters``. Sharpens the converged transition logits in the
            tail (Adam stalls when probability is near 1; smaller LR lets
            logits keep separating).
        """
        if self.hmm is None:
            raise RuntimeError(
                "finalize_hmm needs an existing HMM (run fit_joint first)."
            )
        if len(image_sequences) != len(action_sequences):
            raise ValueError(
                "image_sequences and action_sequences must align"
            )

        # Phase 2.5+++ — optional pi reset. Done before encoding so the log
        # message lands first; nothing about encoding depends on pi.
        if reset_pi:
            print("[finalize] resetting pi to uniform (zeros)")
            self.hmm.pi.assign(
                tf.zeros([self.hmm.num_states], dtype=tf.float32)
            )

        # Encode at the current codebook (no compaction — joint mode keeps
        # the full alphabet).
        token_seqs: List[np.ndarray] = [
            self.encode_images(ep).astype(np.int32) for ep in image_sequences
        ]
        if self.compact_tokens and self._token_remap is not None:
            token_seqs = [
                self._token_remap[s].astype(np.int32) for s in token_seqs
            ]
        action_seqs = [np.asarray(a, dtype=np.int32) for a in action_sequences]

        # Phase 2.5+++ — optional chunking. The integer-only sanity demo
        # converges on 4 episodes × 2000 steps batched together; long
        # episodes (10000 steps) make each gradient update expensive AND
        # bias the gradient toward easy late-episode steps. Chunking fixes
        # both.
        if chunk_size and int(chunk_size) > 0:
            token_seqs, action_seqs = chunk_sequences(
                token_seqs, action_seqs, chunk_size=int(chunk_size)
            )
            print(
                f"[finalize] chunked into {len(token_seqs)} windows of "
                f"<= {chunk_size} steps each"
            )

        ds = make_dataset(
            token_seqs, action_seqs, # type: ignore
            batch_size=batch_size, shuffle=True,
        )

        # Phase 2.5+++ — Adam's `learning_rate` attribute is a `tf.Variable`
        # in Keras 3; we mutate it per-iter from Python (no retracing).
        optimizer = tf.keras.optimizers.Adam(float(learning_rate))

        # The HMM's `batch_loss` returns mean NLL summed over an entire
        # padded sequence (per-seq). The joint phase reports `hmm_term` =
        # length-normalized soft NLL (per-step). To make the two phases
        # directly comparable, also compute a per-step value here. With
        # near-uniform sequence lengths in a batch (true after chunking,
        # and trivially true for full-episode batches), per-step =
        # per-seq / mean(L) is exact.
        @tf.function(jit_compile=True, reduce_retracing=True)
        def train_step(batch):
            O, A, L = batch
            with tf.GradientTape() as tape:
                raw_loss = self.hmm.batch_loss(O, A, L) # type: ignore
                loss = raw_loss
                if transition_entropy_weight > 0.0:
                    trans = tf.nn.softmax(self.hmm.transition, axis=2) # type: ignore
                    trans_safe = tf.clip_by_value(trans, 1e-8, 1.0)
                    transition_entropy = tf.reduce_mean(
                        -tf.reduce_sum(
                            trans_safe * tf.math.log(trans_safe), axis=2
                        )
                    )
                    loss = loss + float(transition_entropy_weight) * transition_entropy
            grads = tape.gradient(loss, [self.hmm.pi, self.hmm.transition]) # type: ignore
            optimizer.apply_gradients(
                zip(grads, [self.hmm.pi, self.hmm.transition]) # type: ignore
            )
            mean_L = tf.reduce_mean(tf.cast(L, tf.float32))
            per_step = raw_loss / tf.maximum(mean_L, 1.0)
            return raw_loss, per_step

        history: List[float] = []
        per_step_history: List[float] = []
        if print_every is None:
            print_every = max(1, n_iters // 20)

        decay_active = lr_decay_to is not None and float(lr_decay_to) > 0.0

        ds_iter = iter(ds)
        for i in range(1, n_iters + 1):
            # Linear LR decay learning_rate -> lr_decay_to over n_iters.
            if decay_active:
                if n_iters > 1:
                    progress = (i - 1) / float(n_iters - 1)
                else:
                    progress = 1.0
                cur_lr = (
                    float(learning_rate) * (1.0 - progress)
                    + float(lr_decay_to) * progress
                )
                optimizer.learning_rate.assign(cur_lr)

            try:
                batch = next(ds_iter)
            except StopIteration:
                ds_iter = iter(ds)
                batch = next(ds_iter)
            loss, per_step = train_step(batch)
            history.append(float(loss.numpy()))
            per_step_history.append(float(per_step.numpy()))
            if i == 1 or i % print_every == 0:
                cur_lr_log = float(optimizer.learning_rate.numpy()) \
                    if hasattr(optimizer.learning_rate, "numpy") \
                    else float(optimizer.learning_rate)
                print(
                    f"[finalize {i:5d}/{n_iters}] "
                    f"hmm_nll/seq={history[-1]:.4f} "
                    f"hmm_nll/step={per_step_history[-1]:.6f} "
                    f"lr={cur_lr_log:.2e}"
                )

        return {"per_seq": history, "per_step": per_step_history}

    # ------------------------------------------------------------------
    # Convenience: full stagewise pipeline
    # ------------------------------------------------------------------
    def fit(
        self,
        images: np.ndarray,
        image_sequences: Sequence[np.ndarray],
        action_sequences: Sequence[np.ndarray],
        vqvae_epochs: int = 10,
        hmm_iters: int = 5000,
        vqvae_batch_size: int = 128,
        hmm_batch_size: int = 8,
        vqvae_lr: float = 3e-4,
        hmm_lr: float = 1e-3,
        image_labels: Optional[np.ndarray] = None,
        vqvae_supervision_weight: float = 0.0,
        observation_clone_counts: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
        verbose: int = 1,
    ) -> TrainHistory:
        """Run VQ-VAE training, then HMM training, end-to-end."""
        if verbose:
            print("=" * 72)
            print("Phase 1 stage 1/2: training VQ-VAE")
            print("=" * 72)
        vq_hist = self.fit_vqvae(
            images,
            labels=image_labels,
            supervision_weight=vqvae_supervision_weight,
            epochs=vqvae_epochs,
            batch_size=vqvae_batch_size,
            learning_rate=vqvae_lr,
            verbose=verbose,
        )

        if observation_clone_counts is not None:
            if image_labels is None:
                raise ValueError(
                    "observation_clone_counts requires image_labels so clone "
                    "counts can be resolved from observation labels to VQ tokens"
                )
            self.set_clone_counts_from_observation_labels(
                images,
                image_labels,
                observation_clone_counts,
                batch_size=vqvae_batch_size,
                verbose=verbose,
            )

        if verbose:
            print("=" * 72)
            print("Phase 1 stage 2/2: training GradientHMM on VQ tokens")
            print("=" * 72)
        hmm_hist = self.fit_hmm(
            image_sequences,
            action_sequences,
            n_iters=hmm_iters,
            learning_rate=hmm_lr,
            batch_size=hmm_batch_size,
            print_every=max(1, hmm_iters // 20),
            seed=seed,
        )

        # Final perplexity from the union of all images
        all_imgs = np.concatenate([images] + list(image_sequences), axis=0)
        usage = self.vqvae.codebook_usage(all_imgs)
        probs = usage / max(usage.sum(), 1)
        perp = float(np.exp(-(probs * np.log(probs + 1e-12)).sum()))

        return TrainHistory(
            vqvae_history=vq_hist,
            hmm_loss_history=hmm_hist,
            perplexity=perp,
            used_tokens=int((usage > 0).sum()),
        )

    def fit_joint_pipeline(
        self,
        images: np.ndarray,
        image_sequences: Sequence[np.ndarray],
        action_sequences: Sequence[np.ndarray],
        image_labels: Optional[np.ndarray] = None,
        warmup_epochs: int = 4,
        joint_iters: int = 5000,
        warmup_batch_size: int = 128,
        joint_batch_size: int = 4,
        joint_chunk_size: int = 256,
        warmup_lr: float = 3e-4,
        joint_lr: float = 3e-4,
        lambda_hmm: float = 1.0,
        temperature: float = 1.0,
        commitment_weight: float = 1.0,
        # Phase 2.5 — Smart defaults: length-normalize ON, λ anneal over the
        # first ~25 % of joint iters, modest diversity penalty.
        length_normalize_hmm: bool = True,
        lambda_anneal_steps: Optional[int] = None,
        diversity_weight: float = 0.1,
        # Phase 2.5+ — HMM-LR boost during joint phase + optional pure-HMM
        # finalization step at the end (encoder frozen, hard tokens).
        hmm_lr_multiplier: float = 1.0,
        freeze_vqvae_during_joint: bool = False,
        finalize_hmm_iters: int = 0,
        finalize_hmm_lr: float = 1e-2,
        finalize_hmm_batch_size: int = 8,
        finalize_hmm_chunk_size: int = 0,
        finalize_hmm_reset_pi: bool = False,
        finalize_hmm_lr_end: float = 0.0,
        finalize_hmm_transition_entropy: float = 0.0,
        # Phase 2.5++ — anti-collapse safeguards
        check_every: int = 100,
        save_best_by_perplexity: bool = True,
        early_stop_min_perplexity: float = 0.0,
        early_stop_patience: int = 3,
        check_images_max: int = 4096,
        # Phase 2.5+++ — active recovery from incipient collapse
        lambda_throttle_floor: float = 0.0,
        lambda_throttle_factor: float = 0.5,
        rollback_min_perplexity: float = 0.0,
        seed: Optional[int] = None,
        verbose: int = 1,
        vqvae_supervision_weight: float = 0.0,
        observation_clone_counts: Optional[Sequence[int]] = None,
    ) -> TrainHistory:
        """Phase 2 convenience: VQ-VAE warmup → joint encoder/decoder/HMM training.

        Returns the same :class:`TrainHistory` shape as :meth:`fit`, with the
        joint per-step losses living in ``history.joint_history``.
        """
        if verbose:
            print("=" * 72)
            print("Phase 2 stage 1/2: VQ-VAE warmup (no HMM loss yet)")
            print("=" * 72)
        if warmup_epochs > 0:
            vq_hist = self.fit_vqvae(
                images,
                labels=image_labels,
                supervision_weight=vqvae_supervision_weight,
                epochs=warmup_epochs,
                batch_size=warmup_batch_size,
                learning_rate=warmup_lr,
                verbose=verbose,
            )
        else:
            vq_hist = {}

        if observation_clone_counts is not None:
            if image_labels is None:
                raise ValueError(
                    "observation_clone_counts requires image_labels so clone "
                    "counts can be resolved from observation labels to VQ tokens"
                )
            self.set_clone_counts_from_observation_labels(
                images,
                image_labels,
                observation_clone_counts,
                batch_size=warmup_batch_size,
                verbose=verbose,
            )

        if lambda_anneal_steps is None:
            # Default: ramp over the first 25 % of joint iters.
            lambda_anneal_steps = max(1, joint_iters // 4)

        if verbose:
            print("=" * 72)
            stage_label = (
                "stage 2/3" if finalize_hmm_iters > 0 else "stage 2/2"
            )
            print(
                f"Phase 2 {stage_label}: joint training "
                f"(lambda={lambda_hmm}, anneal={lambda_anneal_steps}, "
                f"tau={temperature}, beta={commitment_weight}, "
                f"diversity={diversity_weight}, "
                f"normalize={length_normalize_hmm}, "
                f"hmm_lr_mult={hmm_lr_multiplier})"
            )
            print("=" * 72)
        # Sub-sample images for fast perplexity checks during joint training.
        check_images = None
        if (
            save_best_by_perplexity
            or early_stop_min_perplexity > 0
            or lambda_throttle_floor > 0
            or rollback_min_perplexity > 0
        ):
            n = min(int(check_images_max), images.shape[0])
            if n > 0:
                idx = np.random.default_rng(0).choice(
                    images.shape[0], size=n, replace=False
                )
                check_images = images[idx]

        joint_hist = self.fit_joint(
            image_sequences,
            action_sequences,
            n_iters=joint_iters,
            chunk_size=joint_chunk_size,
            batch_size=joint_batch_size,
            learning_rate=joint_lr,
            lambda_hmm=lambda_hmm,
            temperature=temperature,
            commitment_weight=commitment_weight,
            length_normalize_hmm=length_normalize_hmm,
            lambda_anneal_steps=lambda_anneal_steps,
            diversity_weight=diversity_weight,
            hmm_lr_multiplier=hmm_lr_multiplier,
            freeze_vqvae=freeze_vqvae_during_joint,
            check_every=check_every,
            check_images=check_images,
            save_best_by_perplexity=save_best_by_perplexity,
            early_stop_min_perplexity=early_stop_min_perplexity,
            early_stop_patience=early_stop_patience,
            lambda_throttle_floor=lambda_throttle_floor,
            lambda_throttle_factor=lambda_throttle_factor,
            rollback_min_perplexity=rollback_min_perplexity,
            warmup_epochs=0,  # already done above
            seed=seed,
        )

        finalize_hist: Dict[str, List[float]] = {"per_seq": [], "per_step": []}
        if finalize_hmm_iters > 0:
            if verbose:
                print("=" * 72)
                print(
                    f"Phase 2.5+ stage 3/3: pure-HMM finalization "
                    f"({finalize_hmm_iters} iters, encoder frozen, hard tokens, "
                    f"lr={finalize_hmm_lr})"
                )
                print("=" * 72)
            finalize_hist = self.finalize_hmm(
                image_sequences,
                action_sequences,
                n_iters=finalize_hmm_iters,
                learning_rate=finalize_hmm_lr,
                batch_size=finalize_hmm_batch_size,
                chunk_size=finalize_hmm_chunk_size,
                reset_pi=finalize_hmm_reset_pi,
                lr_decay_to=finalize_hmm_lr_end,
                transition_entropy_weight=finalize_hmm_transition_entropy,
            )

        all_imgs = np.concatenate([images] + list(image_sequences), axis=0)
        usage = self.vqvae.codebook_usage(all_imgs)
        probs = usage / max(usage.sum(), 1)
        perp = float(np.exp(-(probs * np.log(probs + 1e-12)).sum()))

        return TrainHistory(
            vqvae_history=vq_hist,
            hmm_loss_history=joint_hist["hmm"],
            joint_history=joint_hist,
            finalize_history=finalize_hist["per_seq"],
            finalize_per_step_history=finalize_hist["per_step"],
            perplexity=perp,
            used_tokens=int((usage > 0).sum()),
        )

    # ------------------------------------------------------------------
    # Decoding & evaluation
    # ------------------------------------------------------------------
    def _require_hmm(self) -> GradientHMM:
        if self.hmm is None:
            raise RuntimeError("HMM is not yet trained; call fit_hmm() / fit() first.")
        return self.hmm

    def decode_sequence(
        self,
        images_or_tokens: np.ndarray,
        actions: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Run Viterbi on a single episode.

        Accepts either raw images ``[T, H, W, C]`` or pre-encoded tokens
        ``[T]``. Returns ``(neg_log_prob, latent_states, used_tokens)``.
        """
        hmm = self._require_hmm()
        if images_or_tokens.ndim == 4:
            tokens_raw = self.encode_images(images_or_tokens)
        else:
            tokens_raw = np.asarray(images_or_tokens, dtype=np.int32)
        if self.compact_tokens and self._token_remap is not None:
            tokens = self._token_remap[tokens_raw].astype(np.int32)
            if (tokens < 0).any():
                # token id wasn't in training; fall back to nearest used code
                # by remapping unknowns to the most common token in training
                unknown = tokens < 0
                # default fallback: code 0 of the active alphabet
                tokens[unknown] = 0
        else:
            tokens = tokens_raw.astype(np.int32)
        actions = np.asarray(actions, dtype=np.int32)
        nll, states = hmm.decode(tokens, actions)
        return nll, states, tokens

    def reconstruct(self, images: np.ndarray) -> np.ndarray:
        """Encode → quantize → decode round-trip."""
        out = self.vqvae(tf.constant(images, dtype=tf.float32), training=False)
        return out["recon"].numpy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, save_dir: str) -> None:
        """Persist the wrapper to ``save_dir``.

        Layout:

        * ``metadata.json``     — hyperparameters + active vocab size.
        * ``vqvae.weights.h5``  — Keras weights (encoder, decoder, codebook).
        * ``hmm.ckpt.*``        — ``tf.train.Checkpoint`` of ``pi`` and
          ``transition`` (omitted if the HMM hasn't been built yet).
        * ``remap.npz``         — token compaction tables (omitted if
          ``compact_tokens=False`` or ``fit_hmm`` hasn't run).

        Re-create with :meth:`load`.
        """
        os.makedirs(save_dir, exist_ok=True)

        meta = {
            "image_shape": list(self.image_shape),
            "codebook_size": self.codebook_size,
            "embedding_dim": self.embedding_dim,
            "num_clones": self.num_clones,
            "num_actions": self.num_actions,
            "commitment_beta": self.commitment_beta,
            "ema_decay": self.ema_decay,
            "base_filters": self.base_filters,
            "compact_tokens": self.compact_tokens,
            "dead_code_threshold": self.dead_code_threshold,
            "clone_counts": self.clone_counts_config,
            "active_clone_counts": self.active_clone_counts.tolist(),
            "clone_count_token_labels": self.clone_count_token_labels,
            "active_vocab_size": int(self.active_vocab_size),
            "has_hmm": self.hmm is not None,
            "format_version": 1,
        }
        with open(os.path.join(save_dir, _METADATA_FILE), "w") as f:
            json.dump(meta, f, indent=2)

        # Materialize VQ-VAE variables before save_weights, in case the
        # caller hasn't run a forward pass yet.
        if not self.vqvae.built:
            dummy = tf.zeros((1,) + self.image_shape, dtype=tf.float32)
            _ = self.vqvae(dummy, training=False)
        self.vqvae.save_weights(os.path.join(save_dir, _VQVAE_WEIGHTS))

        if self.hmm is not None:
            ckpt = tf.train.Checkpoint(
                pi=self.hmm.pi, transition=self.hmm.transition
            )
            ckpt.write(os.path.join(save_dir, _HMM_CKPT_PREFIX))

        if self._token_remap is not None and self._token_unmap is not None:
            np.savez(
                os.path.join(save_dir, _REMAP_FILE),
                token_remap=self._token_remap,
                token_unmap=self._token_unmap,
            )

    @classmethod
    def load(cls, save_dir: str) -> "VQVAEGradientHMM":
        """Reload a wrapper previously written by :meth:`save`."""
        meta_path = os.path.join(save_dir, _METADATA_FILE)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No metadata.json under {save_dir}")
        with open(meta_path) as f:
            meta = json.load(f)

        instance = cls(
            image_shape=tuple(meta["image_shape"]),
            codebook_size=meta["codebook_size"],
            embedding_dim=meta["embedding_dim"],
            num_clones=meta["num_clones"],
            num_actions=meta["num_actions"],
            commitment_beta=meta.get("commitment_beta", 0.25),
            ema_decay=meta.get("ema_decay", 0.99),
            base_filters=meta.get("base_filters", 32),
            compact_tokens=meta.get("compact_tokens", True),
            dead_code_threshold=meta.get("dead_code_threshold", 0.0),
            clone_counts=meta.get("clone_counts", None),
        )

        # Build VQ-VAE weights before loading (Keras lazy build).
        dummy = tf.zeros((1,) + instance.image_shape, dtype=tf.float32)
        _ = instance.vqvae(dummy, training=False)
        instance.vqvae.load_weights(os.path.join(save_dir, _VQVAE_WEIGHTS))

        # Token remap (optional)
        remap_path = os.path.join(save_dir, _REMAP_FILE)
        if os.path.exists(remap_path):
            with np.load(remap_path) as data:
                instance._token_remap = data["token_remap"].astype(np.int32)
                instance._token_unmap = data["token_unmap"].astype(np.int32)
            instance.active_vocab_size = int(meta["active_vocab_size"])
        else:
            instance.active_vocab_size = int(meta.get("active_vocab_size",
                                                      instance.codebook_size))

        # HMM (optional — present iff fit_hmm was called before save)
        if "active_clone_counts" in meta:
            instance.active_clone_counts = np.asarray(
                meta["active_clone_counts"], dtype=np.int32
            )
        else:
            instance.active_clone_counts = instance._clone_counts_for_vocab(
                instance.active_vocab_size,
                token_unmap=instance._token_unmap,
            )
        instance.clone_count_token_labels = meta.get("clone_count_token_labels")

        ckpt_index = os.path.join(save_dir, _HMM_CKPT_PREFIX + ".index")
        if meta.get("has_hmm") and os.path.exists(ckpt_index):
            instance.hmm = instance._build_hmm_from_clone_counts(
                instance.active_clone_counts,
            )
            ckpt = tf.train.Checkpoint(
                pi=instance.hmm.pi, transition=instance.hmm.transition
            )
            ckpt.read(os.path.join(save_dir, _HMM_CKPT_PREFIX)).expect_partial()

        return instance

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------
    @property
    def num_states(self) -> int:
        """Total HMM latent states (clones * active_vocab + 1 sink)."""
        if self.hmm is not None:
            return int(self.hmm.num_states)
        return int(np.asarray(self.active_clone_counts, dtype=np.int32).sum()) + 1

    def codebook_usage(self, images: np.ndarray) -> np.ndarray:
        """Per-token usage counts over a batch of images."""
        return self.vqvae.codebook_usage(np.asarray(images, dtype=np.float32))

    def clone_assignments(
        self,
        image_sequence: np.ndarray,
        action_sequence: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Decode and report per-step token + clone-state assignments."""
        nll, states, tokens = self.decode_sequence(image_sequence, action_sequence)
        # state -> (token, clone within token)
        # state == num_states - 1 is the ancillary sink (shouldn't be MAP)
        if self.compact_tokens and self._token_unmap is not None:
            raw_tokens = self._token_unmap[tokens]
        else:
            raw_tokens = tokens
        clones = self._state_clone_indices_numpy(states)
        token_of_state = self._state_tokens_numpy(states)
        return {
            "neg_log_prob": np.array(nll, dtype=np.float32),
            "tokens": tokens.astype(np.int32),
            "raw_tokens": raw_tokens.astype(np.int32),
            "states": states.astype(np.int32),
            "clones": clones.astype(np.int32),
            "token_of_state": token_of_state.astype(np.int32),
        }
