from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def bh_adjust_1d(p_values):
    out = np.full(p_values.size, np.nan)
    n_valid = 0
    for i in range(p_values.size):
        value = p_values[i]
        if np.isfinite(value):
            n_valid += 1

    if n_valid == 0:
        return out

    valid_p = np.empty(n_valid, np.float64)
    valid_idx = np.empty(n_valid, np.int64)
    write = 0
    for i in range(p_values.size):
        value = p_values[i]
        if np.isfinite(value):
            valid_p[write] = value
            valid_idx[write] = i
            write += 1

    order = np.argsort(valid_p)
    previous = 1.0
    for rank_pos in range(n_valid - 1, -1, -1):
        sorted_pos = order[rank_pos]
        adjusted = valid_p[sorted_pos] * n_valid / (rank_pos + 1.0)
        if adjusted > previous:
            adjusted = previous
        if adjusted > 1.0:
            adjusted = 1.0
        previous = adjusted
        out[valid_idx[sorted_pos]] = adjusted

    return out


@njit(cache=True)
def bh_adjust_flat_matrix(p_values):
    flat_adjusted = bh_adjust_1d(p_values.ravel())
    return flat_adjusted.reshape(p_values.shape)


@njit(cache=True)
def bh_adjust_by_comparison(p_values):
    out = np.full(p_values.shape, np.nan)
    for pair_idx in range(p_values.shape[0]):
        out[pair_idx] = bh_adjust_1d(p_values[pair_idx])
    return out


@njit(cache=True)
def bh_adjust_by_protein(p_values):
    out = np.full(p_values.shape, np.nan)
    for protein in range(p_values.shape[1]):
        out[:, protein] = bh_adjust_1d(p_values[:, protein])
    return out


@njit(cache=True)
def mask_tukey_by_protein(p_values, protein_mask):
    out = np.full(p_values.shape, np.nan)
    for pair_idx in range(p_values.shape[0]):
        for protein in range(p_values.shape[1]):
            if protein_mask[protein]:
                out[pair_idx, protein] = p_values[pair_idx, protein]
    return out
