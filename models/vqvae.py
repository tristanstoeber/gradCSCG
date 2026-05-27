"""
TensorFlow 2 VQ-VAE for the Phase-1 perceptual front-end.

Architecture
------------
* **Encoder**: small CNN that reduces ``(H, W, C_in)`` → ``(1, 1, D)`` so each
  image becomes a single ``D``-dim latent (Option 1: one token per
  observation). Suitable for MNIST-scale inputs.
* **VectorQuantizerEMA**: codebook of ``K`` vectors of dim ``D``; nearest
  neighbour assignment with a straight-through estimator on the encoder, and
  exponential-moving-average codebook updates (no gradient through the
  codebook). Tracks usage perplexity.
* **Decoder**: mirror of the encoder; reconstructs the input.

Loss
----
``L = recon_loss + commitment_beta * commitment_loss``
where ``recon_loss`` is mean-squared error and ``commitment_loss`` is
``mean((z_e - sg(e))^2)``. The codebook is updated by EMA, not gradients.

Usage
-----

    vqvae = VQVAE(input_shape=(28, 28, 1), embedding_dim=32, num_embeddings=64)
    out = vqvae(images, training=True)         # dict with 'recon', 'tokens', 'vq_loss', ...
    tokens = vqvae.encode_tokens(images)       # int32 [B]
    recon = vqvae.decode_tokens(tokens)        # float32 [B, H, W, C]

The model is a standard ``tf.keras.Model``, so ``compile`` + ``fit`` works,
but for tighter control we expose :meth:`train_step` for use inside custom
loops as well.

Reference: Oord et al., 2017 (VQ-VAE); Razavi et al., 2019 (EMA codebook).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf


class VectorQuantizerEMA(tf.keras.layers.Layer):
    """Vector-quantization layer with EMA codebook updates.

    The layer expects continuous inputs of shape ``[..., D]`` and returns
    quantized vectors of the same shape, plus the discrete code indices and
    the (commitment) loss term.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_beta: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        dead_code_threshold: float = 0.0,
        name: str = "vector_quantizer_ema",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_beta = float(commitment_beta)
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        # Phase 2.5++ — dead-code revival. Codes whose EMA cluster size falls
        # below this threshold are reseeded from random current-batch encoder
        # outputs at the next training-mode call. ``0.0`` disables revival
        # (preserves Phase 1/2/2.5 behaviour and exact seed reproducibility);
        # any positive value enables it. The default is ``0.0`` so users with
        # a known-good seeded config keep getting bit-identical results.
        # Opt in (e.g. ``1.0``) for collapse protection on harder problems.
        self.dead_code_threshold = float(dead_code_threshold)
        # Revival samples from the current batch via a *separate* random
        # generator so that calls to it do NOT consume the global RNG state.
        # This keeps seeded reproducibility intact even when revival is on.
        if self.dead_code_threshold > 0.0:
            self._revival_rng = tf.random.Generator.from_seed(
                42, alg="philox"
            )
        else:
            self._revival_rng = None

        initializer = tf.keras.initializers.RandomUniform(
            minval=-1.0 / self.num_embeddings, maxval=1.0 / self.num_embeddings
        )
        self.embeddings = self.add_weight(
            name="codebook",
            shape=(self.embedding_dim, self.num_embeddings),
            initializer=initializer,
            trainable=False,
            dtype=tf.float32,
        )
        # EMA accumulators
        self.ema_cluster_size = self.add_weight(
            name="ema_cluster_size",
            shape=(self.num_embeddings,),
            initializer="zeros",
            trainable=False,
            dtype=tf.float32,
        )
        self.ema_w = self.add_weight(
            name="ema_w",
            shape=(self.embedding_dim, self.num_embeddings),
            initializer=initializer,
            trainable=False,
            dtype=tf.float32,
        )

    def call(
        self, inputs: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Quantize ``inputs`` (``[..., D]``).

        Returns
        -------
        quantized
            Same shape as ``inputs``, equal to the closest codebook vectors
            with a straight-through gradient back to ``inputs``.
        encoding_indices
            ``int32`` shape ``inputs.shape[:-1]``; the assigned code index per
            element.
        commitment_loss
            Scalar scalar loss (already weighted by ``commitment_beta``).
        perplexity
            Scalar tensor; soft measure of codebook utilization.
        """
        input_shape = tf.shape(inputs)
        flat = tf.reshape(inputs, (-1, self.embedding_dim))  # [N, D]

        # squared L2 distance to each codebook vector (broadcast)
        # ||x - e||^2 = ||x||^2 - 2 x.e + ||e||^2
        dist = (
            tf.reduce_sum(flat ** 2, axis=1, keepdims=True)
            - 2.0 * tf.matmul(flat, self.embeddings)
            + tf.reduce_sum(self.embeddings ** 2, axis=0, keepdims=True)
        )  # [N, K]
        encoding_indices_flat = tf.argmax(-dist, axis=1, output_type=tf.int32)  # [N]
        encodings = tf.one_hot(encoding_indices_flat, self.num_embeddings, dtype=tf.float32)

        # Quantize then reshape back
        quantized_flat = tf.matmul(encodings, self.embeddings, transpose_b=True)  # [N, D]
        quantized = tf.reshape(quantized_flat, input_shape)
        encoding_indices = tf.reshape(encoding_indices_flat, input_shape[:-1])

        # ---- losses ----
        # encoder must commit to codebook
        commitment_loss = self.commitment_beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - inputs) ** 2
        )
        self.add_loss(commitment_loss)

        # ---- EMA codebook update (no gradients) ----
        if training:
            cluster_size_t = tf.reduce_sum(encodings, axis=0)  # [K]
            updated_cluster = (
                self.decay * self.ema_cluster_size + (1.0 - self.decay) * cluster_size_t
            )
            dw = tf.matmul(flat, encodings, transpose_a=True)  # [D, K]
            updated_ema_w = self.decay * self.ema_w + (1.0 - self.decay) * dw

            n = tf.reduce_sum(updated_cluster)
            stabilized = (
                (updated_cluster + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            new_embeddings = updated_ema_w / tf.reshape(stabilized, (1, -1))

            # Phase 2.5++ — dead-code revival. Any code whose updated cluster
            # size sits below `dead_code_threshold` is reseeded from a random
            # encoder output in this batch. This is the standard production
            # remedy for VQ-VAE codebook collapse: dead codes never come back
            # under EMA alone (cluster size only decays), so without this hook
            # a code that loses traction is gone forever.
            if self.dead_code_threshold > 0.0:
                dead_mask = tf.cast(
                    updated_cluster < self.dead_code_threshold, tf.float32
                )  # [K]
                n_flat = tf.shape(flat)[0]
                # Use the dedicated generator so global-seed reproducibility
                # is unaffected by whether revival is on or off.
                rand_idx = self._revival_rng.uniform(
                    [self.num_embeddings], 0, n_flat, dtype=tf.int32
                )
                cands = tf.gather(flat, rand_idx)         # [K, D]
                cands_T = tf.transpose(cands)             # [D, K]
                # Replace dead columns of the codebook with the candidates.
                new_embeddings = (
                    cands_T * dead_mask[None, :]
                    + new_embeddings * (1.0 - dead_mask[None, :])
                )
                # Reset EMA accumulators for revived codes so they get a fair
                # share of future updates instead of being dragged back to
                # zero by the still-near-zero history.
                updated_cluster = tf.where(
                    dead_mask > 0.0,
                    tf.ones_like(updated_cluster),
                    updated_cluster,
                )
                updated_ema_w = (
                    cands_T * dead_mask[None, :]
                    + updated_ema_w * (1.0 - dead_mask[None, :])
                )

            self.ema_cluster_size.assign(updated_cluster)
            self.ema_w.assign(updated_ema_w)
            self.embeddings.assign(new_embeddings)

        # straight-through estimator: gradient flows around the quantization
        quantized_st = inputs + tf.stop_gradient(quantized - inputs)

        # perplexity (utilization summary)
        avg_probs = tf.reduce_mean(encodings, axis=0)
        perplexity = tf.exp(
            -tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10))
        )
        return quantized_st, encoding_indices, commitment_loss, perplexity

    def lookup(self, indices: tf.Tensor) -> tf.Tensor:
        """Look up code vectors from indices ``[...]`` -> ``[..., D]``."""
        indices = tf.cast(indices, tf.int32)
        flat = tf.reshape(indices, (-1,))
        gathered = tf.gather(tf.transpose(self.embeddings), flat)  # [N, D]
        out_shape = tf.concat([tf.shape(indices), [self.embedding_dim]], axis=0)
        return tf.reshape(gathered, out_shape)

    def log_soft_assignments(
        self, inputs: tf.Tensor, temperature: float = 1.0
    ) -> tf.Tensor:
        """Differentiable per-token log probabilities — Phase 2.

        Returns ``log_softmax(-||z - e||² / τ)`` with shape
        ``inputs.shape[:-1] + (num_embeddings,)``. As ``temperature -> 0``
        this collapses to the hard assignment used by :meth:`call`. Unlike
        :meth:`call`, this does **not** update the codebook EMA; pass the
        encoder output through :meth:`call` for that.
        """
        input_shape = tf.shape(inputs)
        flat = tf.reshape(inputs, (-1, self.embedding_dim))
        dist = (
            tf.reduce_sum(flat ** 2, axis=1, keepdims=True)
            - 2.0 * tf.matmul(flat, self.embeddings)
            + tf.reduce_sum(self.embeddings ** 2, axis=0, keepdims=True)
        )  # [N, K]
        log_p_flat = tf.nn.log_softmax(-dist / float(temperature), axis=1)
        out_shape = tf.concat([input_shape[:-1], [self.num_embeddings]], axis=0)
        return tf.reshape(log_p_flat, out_shape)


def _build_encoder(
    input_shape: Tuple[int, int, int], embedding_dim: int, base_filters: int
) -> tf.keras.Model:
    """A 28x28-friendly encoder that ends with a single 1x1xD latent."""
    inp = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.layers.Conv2D(base_filters, 4, strides=2, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Conv2D(base_filters * 2, 4, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(base_filters * 2, 3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(embedding_dim, name="pre_quant")(x)
    x = tf.keras.layers.Reshape((1, 1, embedding_dim))(x)
    return tf.keras.Model(inp, x, name="encoder")


def _build_decoder(
    input_shape: Tuple[int, int, int], embedding_dim: int, base_filters: int
) -> tf.keras.Model:
    """Mirror of the encoder; reconstructs ``input_shape`` from a 1x1xD latent."""
    H, W, C = input_shape
    H4, W4 = H // 4, W // 4

    inp = tf.keras.Input(shape=(1, 1, embedding_dim), name="quantized")
    x = tf.keras.layers.Flatten()(inp)
    x = tf.keras.layers.Dense(H4 * W4 * base_filters * 2, activation="relu")(x)
    x = tf.keras.layers.Reshape((H4, W4, base_filters * 2))(x)
    x = tf.keras.layers.Conv2DTranspose(
        base_filters * 2, 3, strides=1, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.Conv2DTranspose(
        base_filters, 4, strides=2, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.Conv2DTranspose(C, 4, strides=2, padding="same", activation="sigmoid")(x)
    return tf.keras.Model(inp, x, name="decoder")


class VQVAE(tf.keras.Model):
    """End-to-end VQ-VAE producing one discrete token per input image.

    Parameters
    ----------
    input_shape
        ``(H, W, C)``.
    embedding_dim
        Latent / codebook vector size ``D``.
    num_embeddings
        Codebook size ``K`` — this is the **vocabulary size of the HMM
        observations**.
    commitment_beta, decay, epsilon
        VQ hyperparameters; see :class:`VectorQuantizerEMA`.
    base_filters
        Width of the smallest Conv layer.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (28, 28, 1),
        embedding_dim: int = 32,
        num_embeddings: int = 64,
        commitment_beta: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        base_filters: int = 32,
        dead_code_threshold: float = 0.0,
        name: str = "vqvae",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.image_shape = tuple(input_shape)
        self.embedding_dim = int(embedding_dim)
        self.num_embeddings = int(num_embeddings)

        self.encoder = _build_encoder(self.image_shape, self.embedding_dim, base_filters) # type: ignore
        self.quantizer = VectorQuantizerEMA(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            commitment_beta=commitment_beta,
            decay=decay,
            epsilon=epsilon,
            dead_code_threshold=dead_code_threshold,
        )
        self.decoder = _build_decoder(self.image_shape, self.embedding_dim, base_filters) # type: ignore

        # Bookkeeping metrics
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.vq_loss_tracker = tf.keras.metrics.Mean(name="vq_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.perplexity_tracker = tf.keras.metrics.Mean(name="perplexity")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.vq_loss_tracker,
            self.perplexity_tracker,
        ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def call(self, x: tf.Tensor, training: bool = False) -> Dict[str, tf.Tensor]: # type: ignore
        z_e = self.encoder(x, training=training)  # [B, 1, 1, D]
        quant, idx, commit_loss, perp = self.quantizer(z_e, training=training)
        recon = self.decoder(quant, training=training)  # [B, H, W, C]
        # idx shape is [B, 1, 1]; squeeze to [B] for the single-token mode
        tokens = tf.squeeze(idx, axis=[1, 2])
        return {
            "recon": recon,
            "tokens": tokens,
            "z_e": z_e,
            "quantized": quant,
            "commitment_loss": commit_loss,
            "perplexity": perp,
        }

    def train_step(self, data):
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            out = self(x, training=True)
            recon_loss = tf.reduce_mean((x - out["recon"]) ** 2)
            commit_loss = out["commitment_loss"]
            total_loss = recon_loss + commit_loss

        # Only encoder + decoder are trainable; quantizer codebook is EMA.
        trainable = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(total_loss, trainable)
        self.optimizer.apply_gradients(zip(grads, trainable)) # type: ignore

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.vq_loss_tracker.update_state(commit_loss)
        self.perplexity_tracker.update_state(out["perplexity"])
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        out = self(x, training=False)
        recon_loss = tf.reduce_mean((x - out["recon"]) ** 2)
        commit_loss = out["commitment_loss"]
        total_loss = recon_loss + commit_loss
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.vq_loss_tracker.update_state(commit_loss)
        self.perplexity_tracker.update_state(out["perplexity"])
        return {m.name: m.result() for m in self.metrics}

    # ------------------------------------------------------------------
    # Convenience for downstream HMM training
    # ------------------------------------------------------------------
    @tf.function
    def encode_tokens(self, x: tf.Tensor) -> tf.Tensor:
        """Encode images to integer code indices ``[B]``."""
        z_e = self.encoder(x, training=False)
        flat = tf.reshape(z_e, (-1, self.embedding_dim))
        dist = (
            tf.reduce_sum(flat ** 2, axis=1, keepdims=True)
            - 2.0 * tf.matmul(flat, self.quantizer.embeddings)
            + tf.reduce_sum(self.quantizer.embeddings ** 2, axis=0, keepdims=True)
        )
        idx = tf.argmax(-dist, axis=1, output_type=tf.int32)
        return idx

    def encode_tokens_numpy(
        self, images: np.ndarray, batch_size: int = 256
    ) -> np.ndarray:
        """Eager numpy helper. Encodes a stream of images in mini-batches."""
        out = np.empty((images.shape[0],), dtype=np.int32)
        for start in range(0, images.shape[0], batch_size):
            stop = min(start + batch_size, images.shape[0])
            batch = tf.constant(images[start:stop], dtype=tf.float32)
            out[start:stop] = self.encode_tokens(batch).numpy()
        return out

    def soft_tokens(
        self, x: tf.Tensor, temperature: float = 1.0, training: bool = False
    ) -> tf.Tensor:
        """Per-image soft log probabilities over the codebook — Phase 2.

        Returns shape ``[B, K]`` for the single-token-per-image setup
        (encoder ends in 1×1×D so spatial dims squeeze out cleanly).
        """
        z_e = self.encoder(x, training=training)              # [B, 1, 1, D]
        log_p = self.quantizer.log_soft_assignments(z_e, temperature)  # [B,1,1,K]
        return tf.squeeze(log_p, axis=[1, 2])

    @tf.function
    def decode_tokens(self, indices: tf.Tensor) -> tf.Tensor:
        """Decode integer indices ``[B]`` back to reconstructed images."""
        codes = self.quantizer.lookup(indices)  # [B, D]
        codes = tf.reshape(codes, (-1, 1, 1, self.embedding_dim))
        return self.decoder(codes, training=False)

    def codebook_usage(self, images: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Return histogram of code usage over a sample of images."""
        idx = self.encode_tokens_numpy(images, batch_size=batch_size)
        counts = np.bincount(idx, minlength=self.num_embeddings)
        return counts.astype(np.int64)
