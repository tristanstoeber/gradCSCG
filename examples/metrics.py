"""Topology-recovery metrics for the cloned-HMM pipeline.

Given a decoded episode of clone states and the ground-truth positions, these
helpers project the latent transition graph onto a physical place graph and
score it against the environment's true adjacency.

Public surface (the only things the notebook imports):

* :func:`transition_probs`        — extract ``[A, N, N]`` from a trained model.
* :func:`state_cell_assignment`   — map each clone state to its majority cell.
* :func:`true_grid_edges`         — physical adjacency from the grid.
* :func:`projected_cell_edges`    — learned latent transitions projected to
                                    physical-cell edges, with a threshold.
* :func:`edge_prf`                — precision / recall / F1 of an edge set.
* :func:`action_next_cell_accuracy` — top-scored predicted next cell vs. truth.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


__all__ = [
    "ACTION_DELTAS",
    "transition_probs",
    "true_next_cell",
    "true_grid_edges",
    "state_cell_assignment",
    "projected_cell_edges",
    "edge_prf",
    "action_next_cell_accuracy",
]


ACTION_DELTAS: Tuple[Tuple[int, int], ...] = (
    (-1, 0),  # up
    (1, 0),   # down
    (0, -1),  # left
    (0, 1),   # right
)

Cell = Tuple[int, int]
Edge = Tuple[Cell, Cell]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _as_cell(p: Sequence[int]) -> Cell:
    return int(p[0]), int(p[1])


def _is_walkable(grid: np.ndarray, cell: Cell) -> bool:
    r, c = cell
    return (
        0 <= r < grid.shape[0]
        and 0 <= c < grid.shape[1]
        and int(grid[r, c]) != -1
    )


def _state_cell_confusion(
    states: np.ndarray, positions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[Cell]]:
    states = np.asarray(states, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    cells = [_as_cell(p) for p in positions]
    unique_states = np.unique(states)
    unique_cells = sorted(set(cells))
    s2r = {int(s): i for i, s in enumerate(unique_states)}
    c2c = {c: i for i, c in enumerate(unique_cells)}
    confusion = np.zeros((unique_states.size, len(unique_cells)), dtype=np.int64)
    for s, c in zip(states, cells):
        confusion[s2r[int(s)], c2c[c]] += 1
    return confusion, unique_states.astype(np.int64), unique_cells


# ---------------------------------------------------------------------------
# Environment dynamics & adjacency
# ---------------------------------------------------------------------------
def true_next_cell(grid: np.ndarray, cell: Cell, action: int) -> Cell:
    """Environment's true next cell under sticky-wall dynamics."""
    grid = np.asarray(grid)
    dr, dc = ACTION_DELTAS[int(action)]
    nxt = (cell[0] + dr, cell[1] + dc)
    return nxt if _is_walkable(grid, nxt) else cell


def true_grid_edges(
    grid: np.ndarray,
    cells: Iterable[Cell] | None = None,
    include_self_loops: bool = False,
    undirected: bool = True,
) -> set:
    """Physical adjacency edges induced by the grid and action dynamics."""
    grid = np.asarray(grid)
    if cells is None:
        cell_list = [tuple(map(int, rc)) for rc in np.argwhere(grid != -1)]
    else:
        cell_list = sorted({_as_cell(c) for c in cells})
    cell_set = set(cell_list)

    edges: set = set()
    for cell in cell_list:
        for action in range(len(ACTION_DELTAS)):
            nxt = true_next_cell(grid, cell, action)
            if nxt not in cell_set:
                continue
            if cell == nxt and not include_self_loops:
                continue
            edge = (cell, nxt)
            if undirected and edge[0] > edge[1]:
                edge = (edge[1], edge[0])
            edges.add(edge)
    return edges


# ---------------------------------------------------------------------------
# Model -> transition probabilities
# ---------------------------------------------------------------------------
def transition_probs(model) -> np.ndarray:
    """Action-conditioned transition probabilities ``[A, N, N]``."""
    import tensorflow as tf

    if getattr(model, "hmm", None) is None:
        raise ValueError("model does not expose a trained HMM (model.hmm).")
    return tf.nn.softmax(model.hmm.transition, axis=2).numpy()


# ---------------------------------------------------------------------------
# State -> place assignment
# ---------------------------------------------------------------------------
def state_cell_assignment(
    states: np.ndarray, positions: np.ndarray
) -> Tuple[Dict[int, Cell], Dict[str, float]]:
    """Assign every visited latent state to its majority physical cell.

    Returns ``(assignment, metrics)`` where ``assignment`` maps
    ``state_id -> (row, col)`` and ``metrics`` carries clone purity.
    """
    confusion, unique_states, unique_cells = _state_cell_confusion(states, positions)
    row_sums = confusion.sum(axis=1)
    assignment: Dict[int, Cell] = {}
    purities: List[float] = []
    weighted_hits = 0
    for row, state in enumerate(unique_states):
        if row_sums[row] <= 0:
            continue
        col = int(np.argmax(confusion[row]))
        assignment[int(state)] = unique_cells[col]
        hits = int(confusion[row, col])
        weighted_hits += hits
        purities.append(hits / float(row_sums[row]))

    cell_counts: Dict[Cell, int] = defaultdict(int)
    for cell in assignment.values():
        cell_counts[cell] += 1

    metrics = {
        "state_cell_purity": float(np.mean(purities)) if purities else 0.0,
        "state_cell_purity_weighted": float(
            weighted_hits / max(int(row_sums.sum()), 1)
        ),
        "represented_cell_fraction": float(
            len(cell_counts) / max(len(unique_cells), 1)
        ),
        "n_decoded_states": float(len(assignment)),
        "n_visited_cells": float(len(unique_cells)),
    }
    return assignment, metrics


# ---------------------------------------------------------------------------
# Projection and scoring
# ---------------------------------------------------------------------------
def projected_cell_edges(
    trans: np.ndarray,
    visited_states: np.ndarray,
    state_to_cell: Mapping[int, Cell],
    threshold: float = 0.01,
    include_self_loops: bool = False,
    undirected: bool = True,
) -> set:
    """Project high-probability latent transitions to physical-cell edges."""
    visited = np.asarray(sorted(set(map(int, visited_states))), dtype=np.int64)
    edges: set = set()
    for action in range(trans.shape[0]):
        sub = trans[action][np.ix_(visited, visited)]
        rows, cols = np.where(sub > float(threshold))
        for r, c in zip(rows, cols):
            src_state = int(visited[int(r)])
            dst_state = int(visited[int(c)])
            if src_state not in state_to_cell or dst_state not in state_to_cell:
                continue
            src = state_to_cell[src_state]
            dst = state_to_cell[dst_state]
            if src == dst and not include_self_loops:
                continue
            edge = (src, dst)
            if undirected and edge[0] > edge[1]:
                edge = (edge[1], edge[0])
            edges.add(edge)
    return edges


def edge_prf(learned_edges: set, true_edges: set) -> Dict[str, float]:
    """Precision / recall / F1 of a learned edge set vs. ground truth."""
    tp = len(learned_edges & true_edges)
    fp = len(learned_edges - true_edges)
    fn = len(true_edges - learned_edges)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "map_edge_precision": float(precision),
        "map_edge_recall": float(recall),
        "map_edge_f1": float(f1),
        "map_edge_tp": float(tp),
        "map_edge_fp": float(fp),
        "map_edge_fn": float(fn),
    }


def action_next_cell_accuracy(
    trans: np.ndarray,
    visited_states: np.ndarray,
    state_to_cell: Mapping[int, Cell],
    grid: np.ndarray,
) -> Dict[str, float]:
    """Score whether projected actions point to the true next cell.

    For each represented source cell and each action, the projected next cell
    is the one with the highest transition mass; this is compared with the
    environment's true sticky-wall successor.
    """
    grid = np.asarray(grid)
    visited = np.asarray(sorted(set(map(int, visited_states))), dtype=np.int64)
    state_index = {int(s): i for i, s in enumerate(visited)}
    cell_to_states: Dict[Cell, List[int]] = defaultdict(list)
    for state in visited:
        si = int(state)
        if si in state_to_cell:
            cell_to_states[state_to_cell[si]].append(si)

    represented_cells = set(cell_to_states)
    correct = scored = total = covered = 0
    for src_cell, src_states in cell_to_states.items():
        for action in range(trans.shape[0]):
            true_dst = true_next_cell(grid, src_cell, action)
            total += 1
            if true_dst in represented_cells:
                covered += 1
            scores: Counter = Counter()
            for dst_cell, dst_states in cell_to_states.items():
                mass = trans[action][np.ix_(src_states, dst_states)]
                scores[dst_cell] = float(np.max(mass)) if mass.size else 0.0
            if not scores:
                continue
            pred_dst, _ = max(scores.items(), key=lambda kv: kv[1])
            scored += 1
            if pred_dst == true_dst:
                correct += 1

    return {
        "action_next_cell_accuracy": float(correct / max(scored, 1)),
        "action_next_cell_coverage": float(covered / max(total, 1)),
    }
