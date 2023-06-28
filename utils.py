from typing import Iterable
import numpy as np
from scipy.stats import beta as beta_distr
from scipy.special import binom
from itertools import combinations
from collections import Counter


def to_bitstring(number: int, n_digits: int = 8):
    container = np.zeros(n_digits, dtype=int)
    bitstring = np.asarray(list(bin(number)[2:]), dtype='int')
    container[n_digits - len(bitstring):] = bitstring
    return container


def to_int(bitstrings: np.ndarray):
    multiplier = 2 ** np.tile(
        np.arange(bitstrings.shape[2] - 1, -1, -1),
        bitstrings.shape[0] * bitstrings.shape[1]
    ).reshape(bitstrings.shape)
    return np.sum(multiplier * bitstrings, axis=-1)


def create_sample(alpha: float, beta: float, path: list, sample_size: int, n_digits: int):
    sample_idc = (beta_distr.rvs(alpha, beta, size=sample_size) * len(path)).astype('int')
    data_sample = np.sum(np.stack([to_bitstring(path[si], n_digits=n_digits) for si in sample_idc]), axis=0)
    return data_sample


def decompose_states(data: np.ndarray, n_samples: int):
    cell_sample = np.zeros((n_samples, data.shape[0]), dtype='int')
    for i_d, d in enumerate(data):
        cell_sample[:d, i_d] = 1

    possible_state_combinations = [binom(data.shape[0], i) for i in range(data.shape[0] // 2 + 1)]
    if data.shape[0] % 2 == 1:
        n_possible_samples = sum(possible_state_combinations)
    else:
        possible_state_combinations[-1] /= 2.
        n_possible_samples = sum(possible_state_combinations)

    print('Number of possible state combinations: %d' % n_possible_samples)
    all_samples = [cell_sample.copy()]
    for i_flip, n_flip in enumerate(possible_state_combinations):
        if i_flip == 0:
            continue
        possible_row_comb = list(combinations(range(data.shape[0]), int(i_flip)))[:int(n_flip)]
        for idx_tuple in possible_row_comb:
            idx_array = np.array(list(idx_tuple))
            if np.any(data[idx_array] == 0) or np.any(data[idx_array] == n_samples):
                continue
            cs = cell_sample.copy()
            for idx in idx_array:
                cs[:, idx] = cs[::-1, idx]
            all_samples.append(cs)

    return np.asarray(all_samples)


def calc_weights(idc: Iterable):
    occurrence = Counter(idc.reshape(-1).tolist())
    occurrence = {k: occurrence[k] for k in sorted(occurrence.keys())}
    weighing = np.array(list(occurrence.values()), dtype='float')
    return weighing / np.max(weighing)
