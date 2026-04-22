from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def presence_masks_from_counts(counts):
    n_proteins, n_conditions = counts.shape
    masks = np.zeros(n_proteins, np.uint64)
    for protein in range(n_proteins):
        mask = np.uint64(0)
        bit = np.uint64(1)
        for condition in range(n_conditions):
            if counts[protein, condition] > 0:
                mask |= bit
            bit <<= np.uint64(1)
        masks[protein] = mask
    return masks


def encode_group_presence_mask(present):
    mask = 0
    bit = 1
    for value in present:
        if bool(value):
            mask |= bit
        bit <<= 1
    return mask


def decode_group_presence_mask(mask, n_conditions):
    mask = int(mask)
    return np.array([bool(mask & (1 << idx)) for idx in range(n_conditions)], dtype=bool)
