import sys
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib_venn import venn2
from argparse import ArgumentParser

from utils import create_sample, create_network, determine_shortest_longest_path


def parse_arguments(args):
    parser = ArgumentParser('Evaluation of the chess problem based on network structure without training.')
    parser.add_argument('--n_pos', type=int, default=8,
                        help='Number of positions in the bitstring. This determines number of possible states.')
    parser.add_argument('--ns', type=int, default=2,
                        help='Number of possible states per position. For bit string use 2 (ie 0 or 1).')
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Number of sampled data points per sampled time point.')
    parser.add_argument('--train_sample_size', type=int, default=100,
                        help='Number of sampled data points drawn from sampled path during training.')
    parser.add_argument('--cosine_bias', type=float, default=.15,
                        help='The similarity between real data set and sampled data set during training is measured'
                             'using the cosine similarity which is scaled between 0 and 1. Paths which produce similar '
                             'data should be incentivised whereas all others should be discouraged. The cosine '
                             'similarity is biased by this value to allow values larger than 1 for incentive.')
    parser.add_argument('--beta_concentration', type=float, default=10.,
                        help='Concentration (inverse of variance) of the sampled data points.')
    parser.add_argument('--p_connect', type=float, default=.01,
                        help='Connection probability of Erdős–Rényi model.')
    parser.add_argument('--n_path', type=int, default=5,
                        help='Maximum number of considered paths. Increasing this will increase complexity of plots.')
    parser.add_argument('--n_epoch', type=int, default=10000,
                        help='Number of epochs')
    parser.add_argument('--eps_start', type=float, default=.9,
                        help='Starting value of epsilon decline which implements exploration-exploitation trade-off')
    parser.add_argument('--eps_end', type=float, default=.01,
                        help='Ending value of epsilon decline which implements exploration-exploitation trade-off')
    parser.add_argument('--eps_decline', type=float, default=2500.,
                        help='Determines decline of epsilon function.')
    parser.add_argument('--use_kamada_kawai_layout', action='store_true', dest='use_kamada_kawai_layout',
                        help='If set, use kamada kawai layout for plotting instead of grid.')
    parser.add_argument('--save_prefix', type=str, default='',
                        help='Prefix that is added the saved data files.')
    parser.add_argument('--save_fig', action='store_true', dest='save_fig',
                        help='If set, save figures to file.')
    parser.add_argument("--save_mat", action="store_true", dest="save_mat",
                        help="If set, save probability matrix of start and end point in path after each iteration.")

    return parser.parse_args(args)


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


def sample_to_pair(i: int, n_states: int):
    return i // n_states, i % n_states


def main(args):
    def calc_eps(i):
        return eps_boundaries[1] + (eps_boundaries[0] - eps_boundaries[1]) * np.exp(- i / eps_boundaries[2])

    n_pos = args.n_pos
    ns = args.ns
    sample_size = args.sample_size
    train_sample_size = args.train_sample_size
    cosine_bias = args.cosine_bias
    n_epoch = args.n_epoch
    n_path = args.n_path
    beta_concent = args.beta_concentration
    p_edge = args.p_connect
    use_kamada_kawai_layout = args.use_kamada_kawai_layout
    eps_boundaries = (args.eps_start, args.eps_end, args.eps_decline)
    save_mat = args.save_mat
    save_prefix = args.save_prefix
    save_fig = args.save_fig

    Path('data/power_of_%d/chessnn/pmat' % n_pos).mkdir(exist_ok=True, parents=True)
    cmap = plt.get_cmap('gist_rainbow')
    sm = ScalarMappable(norm=Normalize(vmin=0., vmax=1.), cmap=cmap)

    # Create sampling distributions
    beta_modes = np.array([.2, .5, .8])
    alpha = 1. + beta_concent * beta_modes
    beta = 1. + beta_concent * (1 - beta_modes)
    plt.plot(np.arange(n_epoch), calc_eps(np.arange(n_epoch)))
    plt.title('Epsilon during training')
    plt.xlabel('Epoch')
    plt.ylabel(r'$\epsilon$')
    if save_fig:
        Path('figures/power_of_%d/chessnn' % n_pos).mkdir(exist_ok=True, parents=True)
        plt.savefig('figures/power_of_%d/chessnn/%s_epsilon.png' % (n_pos, save_prefix))
        plt.close()
    else:
        plt.show()

    n_states = ns ** n_pos
    print('Number of possible states: %d' % n_states)
    graph = create_network(n_states, p_edge)

    # create data
    path, path_dist = determine_shortest_longest_path(graph)
    l_path = len(path)
    print('Length path: %d' % len(path))
    print('Path', path)
    # Create data sample similar to sequencing
    data_sample1 = create_sample(alpha[0], beta[0], path, sample_size=sample_size, n_digits=n_pos)
    data_sample2 = create_sample(alpha[1], beta[1], path, sample_size=sample_size, n_digits=n_pos)
    data_sample3 = create_sample(alpha[2], beta[2], path, sample_size=sample_size, n_digits=n_pos)

    p = np.ones((n_states, n_states))
    p[np.isinf(path_dist)] = 0.
    p /= np.sum(p)
    for i_epoch in range(n_epoch):
        if i_epoch % 10 == 0:
            print('Epoch: %d' % i_epoch)
            print('Most likely start %d' % np.argmax(np.max(p, axis=0)))
            print('Most likely end %d' % np.argmax(np.max(p, axis=1)))
            if save_mat:
                np.savetxt('data/power_of_%d/chessnn/pmat/pmat_%d.csv' % (n_pos, i_epoch), p)

        eps = calc_eps(i_epoch)
        if eps < np.random.random():
            n = np.random.choice(n_states * n_states, p=p.reshape(-1))
            ni, nj = sample_to_pair(n, n_states)
        else:
            p_equal = (p != 0).astype('float')
            p_equal /= p_equal.sum()
            n = np.random.choice(n_states * n_states, p=p_equal.reshape(-1))
            ni, nj = sample_to_pair(n, n_states)
        try:
            train_path = nx.dijkstra_path(graph, ni, nj)
        except nx.exception.NetworkXNoPath:
            p[ni, nj] = 0.
            p /= np.sum(p)
            continue
        train_sample1 = create_sample(alpha[0], beta[0], train_path, sample_size=train_sample_size, n_digits=n_pos)
        train_sample2 = create_sample(alpha[1], beta[1], train_path, sample_size=train_sample_size, n_digits=n_pos)
        train_sample3 = create_sample(alpha[2], beta[2], train_path, sample_size=train_sample_size, n_digits=n_pos)

        discount_sample1 = cosine_similarity(train_sample1, data_sample1)
        discount_sample2 = cosine_similarity(train_sample2, data_sample2)
        discount_sample3 = cosine_similarity(train_sample3, data_sample3)
        p[ni, nj] *= discount_sample1 * discount_sample2 * discount_sample3 + cosine_bias
        # nodes within the path aren't important as we aim to find the entire path
        train_path = np.asarray(train_path)
        p[np.repeat(train_path[1:], len(train_path) - 1), np.tile(train_path[:-1], len(train_path) - 1)] = 0.
        # renormalise
        p /= np.sum(p)

    p_c = p.copy()
    for i_path in range(n_path):
        if np.all(p_c == 0):
            break
        ni, nj = sample_to_pair(np.argmax(p_c), n_states)
        train_path = nx.dijkstra_path(graph, ni, nj)
        n_sp_overlap = len(set(train_path).intersection(path))
        print('Overlap All Reference Structure: %d' % n_sp_overlap)
        with open(
                'data/power_of_%d/chessnn/%s_path%d.txt' % (
                        n_pos,
                        save_prefix,
                        i_path
                ), 'w') as allref_structure_file:
            allref_structure_file.write('%d\n' % len(train_path))
            allref_structure_file.write('%d\n' % l_path)
            allref_structure_file.write('%d\n' % n_sp_overlap)

        venn2(
            subsets=(len(train_path) - n_sp_overlap, l_path - n_sp_overlap, n_sp_overlap),
            set_labels=('guessed states', 'real states'),
        )
        print('Statistical path length %d: %d' % (i_path, len(train_path)))
        plt.title('Overlap of states based on statistics\npath number %s' % i_path)
        if save_fig:
            plt.savefig('figures/power_of_%d/chessnn/%s_ml_venn_path%d.png' % (n_pos, save_prefix, i_path))
            plt.close()
        else:
            plt.show()

        p_c[ni, nj] = 0.

    if use_kamada_kawai_layout:
        node_pos = nx.kamada_kawai_layout(graph, weight=2)
    else:
        sqrt_n = np.sqrt(n_states)
        node_pos = {i: np.array([i // sqrt_n, i % sqrt_n]) for i in graph.nodes}
    plt.figure(figsize=(18, 7))
    nx.draw_networkx_nodes(
        graph,
        pos=node_pos,
        nodelist=graph.nodes,
        node_color='white',
        edgecolors='black',
    )
    nx.draw_networkx_nodes(
        graph,
        pos=node_pos,
        nodelist=path,
        node_color='tab:blue',
    )

    for i_path in range(n_path):
        if np.all(p == 0):
            break
        scale = (n_path - i_path) / float(n_path)
        ni, nj = sample_to_pair(np.argmax(p), n_states)
        train_path = nx.dijkstra_path(graph, ni, nj)
        print('Statistical path length %d: %d' % (i_path, len(train_path)))
        edgelist = list(zip(train_path[:-1], train_path[1:]))
        c = cmap(scale)
        nx.draw_networkx_edges(
            graph,
            node_pos,
            edgelist=edgelist,
            width=(2. * scale)**2,
            edge_color=c,
        )
        p[ni, nj] = 0.

    plt.subplots_adjust(right=.85)
    cax = plt.axes([.87, 0.1, 0.03, 0.7])
    plt.colorbar(sm, cax=cax, orientation='vertical', label='weighting')

    if save_fig:
        plt.savefig('figures/power_of_%d/chessnn/%s_nnpaths.png' % (n_pos, save_prefix))
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

