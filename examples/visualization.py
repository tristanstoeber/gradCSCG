"""Visualization helpers for the neuralCSCG demo.

Exactly four public functions, one per plot used in the notebook:

* :func:`plot_reconstructions` — VQ-VAE input vs. reconstruction.
* :func:`plot_codebook_usage`  — histogram of token utilization.
* :func:`plot_clone_purity`    — clone-state vs. ground-truth-cell confusion.
* :func:`render_viterbi_graph` — Viterbi-path graph with MNIST-image nodes.

All four save a PNG to ``output_file`` and return summary statistics.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402


__all__ = [
    "plot_reconstructions",
    "plot_codebook_usage",
    "plot_clone_purity",
    "render_viterbi_graph",
]


# ---------------------------------------------------------------------------
# VQ-VAE diagnostic plots
# ---------------------------------------------------------------------------
def plot_reconstructions(
    model,
    images: np.ndarray,
    output_file: str,
    n: int = 12,
) -> None:
    """Plot ``n`` inputs (top) against their VQ-VAE reconstructions (bottom)."""
    n = min(n, images.shape[0])
    sample = images[:n]
    recon = model.reconstruct(sample)
    tokens = model.encode_images(sample)

    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3.5))
    for i in range(n):
        axes[0, i].imshow(sample[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"t={int(tokens[i])}", fontsize=8)
        axes[1, i].imshow(np.clip(recon[i].squeeze(), 0, 1), cmap="gray")
        axes[1, i].axis("off")
    fig.suptitle("VQ-VAE reconstructions  (top: input, bottom: reconstruction)")
    fig.tight_layout()
    fig.savefig(output_file, dpi=120)
    plt.close(fig)


def plot_codebook_usage(model, images: np.ndarray, output_file: str) -> Dict[str, int]:
    """Histogram of token utilization over ``images``.

    Returns a dict with ``codebook_size`` and ``active`` (codes used at least
    once).
    """
    counts = model.codebook_usage(images)
    used = int((counts > 0).sum())

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(np.arange(counts.shape[0]), counts)
    ax.set_xlabel("code id")
    ax.set_ylabel("count")
    ax.set_title(f"Codebook usage ({used}/{counts.shape[0]} active)")
    fig.tight_layout()
    fig.savefig(output_file, dpi=120)
    plt.close(fig)
    return {"codebook_size": int(counts.shape[0]), "active": used}


# ---------------------------------------------------------------------------
# Clone-state diagnostic
# ---------------------------------------------------------------------------
def plot_clone_purity(
    model,
    image_seq: np.ndarray,
    action_seq: np.ndarray,
    positions: np.ndarray,
    output_file: str,
) -> Dict[str, float]:
    """Plot a clone-state vs. ground-truth-cell confusion matrix.

    Decodes the episode with Viterbi, builds the (clone state, flat cell id)
    confusion matrix, saves the heatmap and returns purity statistics.
    """
    info = model.clone_assignments(image_seq, action_seq)
    states = info["states"]
    flat_cell = positions[:, 0] * 100 + positions[:, 1]
    unique_cells, cell_ids = np.unique(flat_cell, return_inverse=True)
    unique_states = np.unique(states)

    state_to_row = {int(s): i for i, s in enumerate(unique_states)}
    confusion = np.zeros((unique_states.size, unique_cells.size), dtype=np.int64)
    for s, c in zip(states, cell_ids):
        confusion[state_to_row[int(s)], c] += 1

    fig, ax = plt.subplots(
        figsize=(0.4 * unique_cells.size + 2, 0.18 * unique_states.size + 2)
    )
    im = ax.imshow(confusion, aspect="auto", cmap="viridis")
    ax.set_xlabel("grid cell (linear id)")
    ax.set_ylabel("clone state")
    ax.set_title("clone state vs. ground-truth cell")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_file, dpi=120)
    plt.close(fig)

    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    purity = float((confusion.max(axis=1) / row_sums.squeeze()).mean())
    return {
        "purity": purity,
        "n_states_used": int(unique_states.size),
        "n_cells_visited": int(unique_cells.size),
    }


# ---------------------------------------------------------------------------
# Viterbi-path graph with MNIST-image nodes
# ---------------------------------------------------------------------------
_DIGIT_CMAP = plt.get_cmap("tab10")
_Edge = Tuple[int, int]


def _to_gray_stack(images: np.ndarray) -> np.ndarray:
    imgs = np.asarray(images, dtype=np.float32)
    if imgs.ndim == 4 and imgs.shape[-1] == 1:
        imgs = imgs[..., 0]
    return np.clip(imgs, 0.0, 1.0)


def _digit_pools(images: np.ndarray, digits: np.ndarray) -> Dict[int, np.ndarray]:
    """digit -> stack of images of that digit, drawn from the episode itself."""
    digits = np.asarray(digits).reshape(-1)
    stack = _to_gray_stack(images)
    return {int(d): stack[np.flatnonzero(digits == d)] for d in np.unique(digits)}


def _state_to_digit(states: np.ndarray, digits: np.ndarray) -> Dict[int, int]:
    """Majority ground-truth digit observed at each clone state."""
    counts: Dict[int, Counter] = defaultdict(Counter)
    for s, d in zip(np.asarray(states).tolist(), np.asarray(digits).tolist()):
        counts[int(s)][int(d)] += 1
    return {s: cc.most_common(1)[0][0] for s, cc in counts.items()}


def _viterbi_edge_counts(
    states: np.ndarray, count_threshold: int
) -> Tuple[List[int], List[_Edge], List[int]]:
    """Count directed clone-state transitions in a Viterbi path, then filter."""
    edge_counts: Counter = Counter()
    arr = np.asarray(states)
    for src, dst in zip(arr[:-1].tolist(), arr[1:].tolist()):
        si, di = int(src), int(dst)
        if si != di:
            edge_counts[(si, di)] += 1
    kept = {e: c for e, c in edge_counts.items() if c > int(count_threshold)}
    active = sorted({s for edge in kept for s in edge})
    idx = {s: i for i, s in enumerate(active)}
    ordered = sorted(kept)
    edges = [(idx[a], idx[b]) for a, b in ordered]
    counts = [kept[e] for e in ordered]
    return active, edges, counts


def _kamada_kawai_layout(n: int, edges: Sequence[_Edge]) -> np.ndarray:
    import igraph

    g = igraph.Graph(n=n, edges=list(edges), directed=True)
    coords = np.asarray(g.layout("kamada_kawai").coords, dtype=np.float64)
    return coords if coords.shape[0] == n else np.zeros((n, 2), dtype=np.float64)


def render_viterbi_graph(
    model,
    images: np.ndarray,
    actions: np.ndarray,
    digits: np.ndarray,
    output_file: str,
    count_threshold: int = 20,
    seed: int = 0,
    title: Optional[str] = None,
) -> Dict[str, int]:
    """Render the Viterbi-path graph with each node drawn as an MNIST digit.

    Decodes one episode with Viterbi, counts directed clone-to-clone
    transitions, keeps the ones used more than ``count_threshold`` times, and
    draws the resulting directed graph. Every node is drawn as an actual MNIST
    sample of the digit it represents (its majority ground-truth observation),
    with a digit-coloured border. Bidirectional edges curve apart so both
    directions remain visible.

    Returns ``{"nodes": n, "edges": e}``.
    """
    _nll, states, _ = model.decode_sequence(np.asarray(images), np.asarray(actions))
    states = np.asarray(states, dtype=np.int64)
    digits = np.asarray(digits).reshape(-1)

    nodes, edges, counts = _viterbi_edge_counts(states, count_threshold)
    if not nodes:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(
            0.5, 0.5,
            f"no edges with count > {count_threshold}",
            ha="center", va="center", fontsize=11, color="#555",
        )
        ax.axis("off")
        fig.savefig(output_file, dpi=120)
        plt.close(fig)
        return {"nodes": 0, "edges": 0}

    state_to_digit = _state_to_digit(states, digits)
    pools = _digit_pools(images, digits)
    coords = _kamada_kawai_layout(len(nodes), edges)

    coords -= coords.min(axis=0)
    span = coords.max(axis=0)
    span[span == 0] = 1.0
    coords /= span.max()

    if len(nodes) > 1:
        diff = coords[:, None, :] - coords[None, :, :]
        dmat = np.sqrt((diff ** 2).sum(axis=-1))
        np.fill_diagonal(dmat, np.inf)
        d_nn = float(np.median(dmat.min(axis=1)))
    else:
        d_nn = 0.5

    node_scale = 0.75       # nodes a bit smaller than max packing
    edge_width_scale = 2.5  # thicker edges so every connection reads clearly
    r = max(0.014, 0.42 * d_nn * node_scale)

    bbox = coords.max(axis=0) - coords.min(axis=0)
    side = float(np.clip(1.5 * np.sqrt(max(len(nodes), 1)), 8.0, 28.0))
    fig_w = max(side * (bbox[0] + 4 * r), 6.0)
    fig_h = max(side * (bbox[1] + 5 * r), 6.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_aspect("equal")
    ax.axis("off")

    arrow_ms = 8.0 + 2.2 * edge_width_scale
    edge_alpha = float(min(0.9, 0.55 * edge_width_scale ** 0.4))
    max_count = max(counts) or 1
    for k, (i, j) in enumerate(edges):
        p0, p1 = coords[i], coords[j]
        delta = p1 - p0
        length = float(np.hypot(*delta))
        if length < 1e-9:
            continue
        u = delta / length
        start = p0 + u * (r * 1.08)
        end = p1 - u * (r * 1.08)
        width = (0.5 + 3.0 * (counts[k] / max_count)) * edge_width_scale
        ax.annotate(
            "",
            xy=tuple(end),
            xytext=tuple(start),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#555555",
                lw=width,
                alpha=edge_alpha,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=arrow_ms,
                connectionstyle="arc3,rad=0.22",
            ),
            zorder=1,
        )

    rng = np.random.default_rng(seed)
    label_fs = float(np.clip(70.0 / np.sqrt(max(len(nodes), 1)), 4.0, 9.0))
    digits_present = sorted({state_to_digit.get(int(s), 0) for s in nodes})
    for idx, state in enumerate(nodes):
        x, y = coords[idx]
        digit = int(state_to_digit.get(int(state), 0))
        pool = pools.get(digit)
        if pool is not None and pool.shape[0] > 0:
            img = pool[int(rng.integers(0, pool.shape[0]))]
            ax.imshow(
                img, cmap="gray", vmin=0.0, vmax=1.0,
                extent=(x - r, x + r, y - r, y + r),
                interpolation="nearest", aspect="auto", zorder=3,
            )
        ax.add_patch(
            Rectangle(
                (x - r, y - r), 2 * r, 2 * r,
                fill=False,
                edgecolor=_DIGIT_CMAP(digit % 10),
                linewidth=2.0,
                zorder=4,
            )
        )
        ax.text(
            x, y - r - 0.45 * r, f"s{int(state)}",
            ha="center", va="top", fontsize=label_fs,
            color="#333333", zorder=5,
        )

    ax.set_xlim(coords[:, 0].min() - 3 * r, coords[:, 0].max() + 3 * r)
    ax.set_ylim(coords[:, 1].min() - 4 * r, coords[:, 1].max() + 3 * r)
    ax.set_title(title or "Viterbi path graph (MNIST nodes)")

    if digits_present:
        handles = [
            Patch(
                facecolor=_DIGIT_CMAP(d % 10),
                edgecolor="black",
                label=f"digit {d}",
            )
            for d in digits_present
        ]
        ax.legend(
            handles=handles, loc="upper left", fontsize=8,
            title="node border = observation digit", framealpha=0.9,
        )

    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"nodes": len(nodes), "edges": len(edges)}
