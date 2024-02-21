from typing import Iterable
import numpy as np
from scipy.stats import beta as beta_distr
from scipy.special import binom
from itertools import combinations
from collections import Counter
import networkx as nx

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
    # Flip successively rows to get all state combinations
    # This makes use of the observation that the order of states doesn't matter:
    # The set [state1, state2, state3] describes the same combinations as [state3, state1, state2]
    # We begin with the easiest state combination: fill the first n states for position i with 1
    # where n(i) is the data value at position i
    cell_sample = np.zeros((n_samples, data.shape[0]), dtype='int')
    for i_d, d in enumerate(data):
        cell_sample[:d, i_d] = 1

    # Now we successively flip columns to create all possible state combinations. This flipping
    # produces a result that is equivalent to sorting with respect to the flipped columns.
    # Compute here the number of column flipping to be performed. Note that flipping is supposed to produce a new
    # state permutation. Hence, when flipping all columns, all states remain exactly the same. Similarly, when
    # considering 8 positions, flipping 2 produces the same state combinations as flipping 6.
    # Note that this produces a special case when having an even number of positions. When considering 8 loci again,
    # flipping any 4 columns produces the same combinations as flipping the exact opposite 4.
    possible_state_combinations = [binom(data.shape[0], i) for i in range(data.shape[0] // 2 + 1)]
    if data.shape[0] % 2 == 1:
        n_possible_samples = sum(possible_state_combinations)
    else:
        possible_state_combinations[-1] /= 2.
        n_possible_samples = sum(possible_state_combinations)

    print('Number of possible state combinations: %d' % n_possible_samples)
    all_samples = [cell_sample.copy()]
    for i_flip, n_flip in enumerate(possible_state_combinations):
        # Ignore the case when no column is flipped
        if i_flip == 0:
            continue
        # Get a combination of all column that can be flipped to produce new combinations
        possible_row_comb = list(combinations(range(data.shape[0]), int(i_flip)))[:int(n_flip)]
        for idx_tuple in possible_row_comb:
            idx_array = np.array(list(idx_tuple))
            if np.any(data[idx_array] == 0) or np.any(data[idx_array] == n_samples):
                continue
            # Flip column and add to possible state combinations
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


def create_network(n_states: int, p_edge: float):
    graph_cyclic = nx.fast_gnp_random_graph(n=n_states, p=p_edge, directed=True)
    graph = nx.DiGraph()
    graph.add_nodes_from(graph_cyclic.nodes())
    graph.add_edges_from([(u, v) for (u, v) in graph_cyclic.edges() if u < v])
    return graph


def determine_shortest_longest_path(graph: nx.DiGraph):
    path_dist = nx.floyd_warshall_numpy(graph)
    sources, targets = np.where(path_dist == path_dist[~np.isinf(path_dist)].max())
    idx = np.random.choice(len(sources))
    s, t = sources[idx], targets[idx]
    path = nx.dijkstra_path(graph, s, t)
    return path, path_dist
