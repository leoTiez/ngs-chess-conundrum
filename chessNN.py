import numpy as np
import networkx as nx
from utils import to_bitstring, create_sample
import matplotlib.pyplot as plt


def path_sampler(graph):
    start, end = np.random.choice(graph.node, size=2, replace=False)
    return nx.dijkstra_path(graph, start, end)


def cosine_similarity(x, y):
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x == 0.:
        norm_x = 1.
    if norm_y == 0.:
        norm_y = 1.
    return (np.dot(x, y) / (norm_x * norm_y) + 1) / 2.


def main():
    def calc_eps(i):
        return eps_boundaries[1] + (eps_boundaries[0] - eps_boundaries[1]) * np.exp(- i / eps_boundaries[2])
    n_pos = 10
    ns = 2
    n_states = ns ** n_pos
    print('Number of possible states: %d' % n_states)
    sample_size = 100
    train_sample_size = 100
    cosine_thresh = .25
    # n_epoch = 100000
    n_epoch = 50000
    n_path = 5
    # Create sampling distributions
    beta_modes = np.array([.2, .5, .8])
    beta_concent = 1
    alpha = 1. + beta_concent * beta_modes
    beta = 1. + beta_concent * (1 - beta_modes)
    # eps_boundaries = (.8, 0.005, 30000)
    eps_boundaries = (.9, 0.01, 12000)
    plt.plot(np.arange(n_epoch), calc_eps(np.arange(n_epoch)))
    plt.title('Epsilon during training')
    plt.show()

    p_edge = .01  # 1% connectivity
    graph = nx.fast_gnp_random_graph(n=n_states, p=p_edge, directed=True)

    # create data
    path_dist = nx.floyd_warshall_numpy(graph)
    sources, targets = np.where(path_dist == path_dist[~np.isinf(path_dist)].max())
    idx = np.random.choice(len(sources))
    s, t = sources[idx], targets[idx]
    path = nx.dijkstra_path(graph, s, t)
    print('Length path: %d' % len(path))
    print('Path', path)
    # Create data sample similar to sequencing
    data_sample1 = create_sample(alpha[0], beta[0], path, sample_size=sample_size, n_digits=n_pos)
    data_sample2 = create_sample(alpha[1], beta[1], path, sample_size=sample_size, n_digits=n_pos)
    data_sample3 = create_sample(alpha[2], beta[2], path, sample_size=sample_size, n_digits=n_pos)

    p_start = np.ones(n_states) / float(n_states)
    p_end = np.ones(n_states) / float(n_states)
    for i_epoch in range(n_epoch):
        if i_epoch % 10 == 0:
            print('Epoch: %d' % i_epoch)
            print('Most likely start %d' % np.argmax(p_start))
            print('Most likely end %d' % np.argmax(p_end))
        eps = calc_eps(i_epoch)
        if eps < np.random.random():
            ni = np.random.choice(n_states, p=p_start)
            nj = np.random.choice(n_states, p=p_end)
        else:
            ps_equal = (p_start != 0).astype('float')
            pe_equal = (p_end != 0).astype('float')
            ps_equal /= ps_equal.sum()
            pe_equal /= pe_equal.sum()
            ni = np.random.choice(n_states, p=ps_equal)
            nj = np.random.choice(n_states, p=ps_equal)
        try:
            train_path = nx.dijkstra_path(graph, ni, nj)
        except nx.exception.NetworkXNoPath:
            p_start[ni] = 0.
            p_end[nj] = 0.
            p_start /= np.sum(p_start)
            p_end /= np.sum(p_end)
            continue
        train_sample1 = create_sample(alpha[0], beta[0], train_path, sample_size=train_sample_size, n_digits=n_pos)
        train_sample2 = create_sample(alpha[1], beta[1], train_path, sample_size=train_sample_size, n_digits=n_pos)
        train_sample3 = create_sample(alpha[2], beta[2], train_path, sample_size=train_sample_size, n_digits=n_pos)
        discount = cosine_similarity(train_sample1, data_sample1) * cosine_similarity(
            train_sample2, data_sample2) * cosine_similarity(train_sample3, data_sample3)
        p_start[ni] *= discount + cosine_thresh
        p_end[nj] *= discount + cosine_thresh

        # renormalise
        p_start /= np.sum(p_start)
        p_end /= np.sum(p_end)

    plt.figure(figsize=(18, 7))
    sqrt_n = np.sqrt(n_states)
    node_pos = {i: np.array([i // sqrt_n, i % sqrt_n]) for i in graph.nodes}
    nx.draw_networkx_nodes(
        graph,
        pos=node_pos,
        nodelist=graph.nodes,
        node_color='white',
        edgecolors='black'
    )
    nx.draw_networkx_nodes(
        graph,
        pos=node_pos,
        nodelist=path,
        node_color='tab:blue'
    )
    for i_path, c in enumerate(['red', 'black', 'orange', 'purple', 'green']):
        scale = (n_path - i_path) / float(n_path)
        ni = np.argmax(p_start)
        nj = np.argmax(p_end)
        train_path = nx.dijkstra_path(graph, ni, nj)
        print('Path length %d: %d' % (i_path, len(train_path)))
        edgelist = list(zip(train_path[:-1], train_path[1:]))
        nx.draw_networkx_edges(
            graph,
            node_pos,
            edgelist=edgelist,
            width=2. * scale,
            edge_color=c
        )
        p_start[ni] = 0.
        p_end[nj] = 0.

    plt.show()


if __name__ == '__main__':
    main()

