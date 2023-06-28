import sys
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
import networkx as nx
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib_venn import venn2

from utils import to_bitstring, create_sample, to_int, decompose_states, calc_weights


def parse_arguments(args):
    parser = ArgumentParser('Evaluation of the chess problem based on network structure without training.')
    parser.add_argument('--n_pos', type=int, default=8,
                        help='Number of positions in the bitstring. This determines number of possible states.')
    parser.add_argument('--ns', type=int, default=2,
                        help='Number of possible states per position. For bit string use 2 (ie 0 or 1).')
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Number of sampled data points per sampled time point.')
    parser.add_argument('--beta_concentration', type=float, default=10.,
                        help='Concentration (inverse of variance) of the sampled data points.')
    parser.add_argument('--p_connect', type=float, default=.01,
                        help='Connection probability of Erdős–Rényi model.')
    parser.add_argument('--n_path', type=int, default=5,
                        help='Maximum number of considered paths. Increasing this will increase complexity of plots.')
    parser.add_argument('--n_iter', type=int, default=100,
                        help='Number of iterations over setup')
    parser.add_argument('--use_weighting', action='store_true', dest='use_weighting',
                        help='If set, weight paths based on how many nodes are in the guessed intermediate states')
    parser.add_argument('--do_plot_network', action='store_true', dest='do_plot_network',
                        help='If set, plot network and their likely participating nodes based on RPM normalisation.')
    parser.add_argument('--use_kamada_kawai_layout', action='store_true', dest='use_kamada_kawai_layout',
                        help='If set, use kamada kawai layout for plotting instead of grid.')
    parser.add_argument('--save_fig', action='store_true', dest='save_fig',
                        help='If set, save figures to file.')

    return parser.parse_args(args)


def determine_paths(graph: nx.Graph, source_list: List[int], target_list: List[int], min_length: int = 3):
    paths_list = []
    for s in source_list:
        for t in target_list:
            # Catch if no path existent
            try:
                prob_path = nx.dijkstra_path(graph, source=s, target=t)
                if len(prob_path) < min_length:
                    continue
                paths_list.append(prob_path)
            except nx.exception.NetworkXNoPath:
                continue

    return paths_list


def plot_edges(
        graph: nx.Graph,
        node_pos: Dict[int, np.ndarray],
        path_list: List[List[int]],
        ax: plt.Axes,
        n_path: int = 5,
        cmap: Colormap = plt.get_cmap('gist_rainbow'),
):
    for i_path, p in enumerate(path_list):
        scale = (n_path - i_path) / float(n_path)
        simple_edgelist = list(zip(p[:-1], p[1:]))
        c = cmap(scale)
        nx.draw_networkx_edges(
            graph,
            node_pos,
            edgelist=simple_edgelist,
            width=3 * scale,
            edge_color=c,
            ax=ax
        )


def main(args):
    n_pos = args.n_pos
    ns = args.ns
    sample_size = args.sample_size
    p_edge = args.p_connect
    n_path = args.n_path
    n_iter = args.n_iter
    use_kamada_kawai_layout = args.use_kamada_kawai_layout
    use_weighting = args.use_weighting
    do_plot_network = args.do_plot_network
    save_fig = args.save_fig
    cmap = plt.get_cmap('gist_rainbow')

    # Create sampling distributions
    beta_modes = np.array([.2, .5, .8])
    beta_concent = args.beta_concentration
    alpha = 1 + beta_concent * beta_modes
    beta = 1 + beta_concent * (1 - beta_modes)

    n_states = ns ** n_pos
    print('Number of possible states: %d' % n_states)
    lookup = np.stack([to_bitstring(i, n_pos) for i in range(n_states)])

    sm = ScalarMappable(norm=Normalize(vmin=0., vmax=1.), cmap=cmap)
    weighting_str = 'weighted' if use_weighting else ''
    # Due to probabilistic graph, iterate several times over it
    for i_iter in range(n_iter):
        print('Iteration %d' % i_iter)
        # Create graph
        graph = nx.fast_gnp_random_graph(n=n_states, p=p_edge, directed=True)
        # Determine paths
        path_dist = nx.floyd_warshall_numpy(graph)
        # Take longest shortest path
        sources, targets = np.where(path_dist == path_dist[~np.isinf(path_dist)].max())
        idx = np.random.choice(len(sources))
        s, t = sources[idx], targets[idx]
        path = nx.dijkstra_path(graph, s, t)
        l_path = len(path)
        print('Length path: %d' % l_path)
        print('Path', path)

        # Create data sample similar to sequencing
        data_sample1 = create_sample(alpha[0], beta[0], path, sample_size=sample_size, n_digits=n_pos)
        data_sample2 = create_sample(alpha[1], beta[1], path, sample_size=sample_size, n_digits=n_pos)
        data_sample3 = create_sample(alpha[2], beta[2], path, sample_size=sample_size, n_digits=n_pos)

        # Define scaling equivalent to reads per million
        scaling1 = np.sum(data_sample1.reshape(1, -1) * lookup / np.sum(data_sample1).astype('float'), axis=1)
        scaling2 = np.sum(data_sample2.reshape(1, -1) * lookup / np.sum(data_sample2).astype('float'), axis=1)
        scaling3 = np.sum(data_sample3.reshape(1, -1) * lookup / np.sum(data_sample3).astype('float'), axis=1)

        scaling1 = scaling1 / np.max(scaling1).astype('float')
        scaling2 = scaling2 / np.max(scaling2).astype('float')
        scaling3 = scaling3 / np.max(scaling3).astype('float')

        # Calculate node positions
        if use_kamada_kawai_layout:
            node_pos = nx.kamada_kawai_layout(graph, weight=2)
        else:
            sqrt_n = np.sqrt(n_states)
            node_pos = {i: np.array([i // sqrt_n, i % sqrt_n]) for i in graph.nodes}
        if do_plot_network:
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
            nx.draw(graph, pos=node_pos, node_shape='.', node_color=scaling1, cmap='Greens', ax=ax[0])
            nx.draw(graph, pos=node_pos, node_shape='.', node_color=scaling2, cmap='Reds', ax=ax[1])
            nx.draw(graph, pos=node_pos, node_shape='.', node_color=scaling3, cmap='Blues', ax=ax[2])
            ax[0].set_title('first', fontsize=21)
            ax[1].set_title('second', fontsize=21)
            ax[2].set_title('third', fontsize=21)
            fig.suptitle('RPM probability', fontsize=32)
            fig.tight_layout()
            if save_fig:
                Path('figures/power_of_%d' % n_pos).mkdir(exist_ok=True, parents=True)
                plt.savefig('figures/power_of_%d/all_possible_nodes_weighted%d.png' % (n_pos, i_iter))
                plt.close()
            else:
                plt.show()

        # Evaluate graph structure at different time points considering most likely states based on scaled values
        # (ie. read per million)
        idc1, = np.where(scaling1 > .75)
        idc2, = np.where(scaling2 > .75)
        idc3, = np.where(scaling3 > .75)
        idc_all = np.unique(np.concatenate([idc1, idc2, idc3]).reshape(-1))

        # Determine paths that are included in subgraph of most likely time points
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        nx.draw(
            graph.subgraph(idc1),
            pos={int(i): node_pos[int(i)] for i in idc1},
            nodelist=idc1.tolist(),
            node_color=scaling1[idc1],
            cmap='Greens',
            ax=ax[0]
        )
        nx.draw(
            graph.subgraph(idc2),
            pos={int(i): node_pos[int(i)] for i in idc2},
            nodelist=idc2.tolist(),
            node_color=scaling2[idc2],
            cmap='Reds',
            ax=ax[1]
        )
        nx.draw(
            graph.subgraph(idc3),
            pos={int(i): node_pos[int(i)] for i in idc3},
            nodelist=idc3.tolist(),
            node_color=scaling2[idc3],
            cmap='Oranges',
            ax=ax[2]
        )
        nx.draw_networkx_nodes(
            graph,
            pos=node_pos,
            nodelist=path,
            node_color='tab:blue',
            ax=ax[0]
        )
        nx.draw_networkx_nodes(
            graph,
            pos=node_pos,
            nodelist=path,
            node_color='tab:blue',
            ax=ax[1]
        )
        nx.draw_networkx_nodes(
            graph,
            pos=node_pos,
            nodelist=path,
            node_color='tab:blue',
            ax=ax[2]
        )
        ax[0].set_title('first', fontsize=21)
        ax[1].set_title('second', fontsize=21)
        ax[2].set_title('third', fontsize=21)
        fig.suptitle('Connectivity based on most likely nodes\nno total reference', fontsize=32)
        fig.tight_layout()
        if save_fig:
            Path('figures/power_of_%d' % n_pos).mkdir(exist_ok=True, parents=True)
            plt.savefig('figures/power_of_%d/naive_%d.png' % (n_pos, i_iter))
            plt.close()
        else:
            plt.show()

        # Check whether necessary nodes are in subgraph
        likely_subgraph = graph.subgraph(idc_all)
        n_likely_nodes = len(likely_subgraph.nodes)
        n_node_overlap = len(set(likely_subgraph.nodes).intersection(path))
        Path('data/power_of_%d/path_determination' % n_pos).mkdir(exist_ok=True, parents=True)
        with open('data/power_of_%d/path_determination/naive_%d.txt' % (n_pos, i_iter), 'w') as naive_file:
            naive_file.write('%d\n' % n_likely_nodes)
            naive_file.write('%d\n' % l_path)
            naive_file.write('%d\n' % n_node_overlap)

        venn2(
            subsets=(n_likely_nodes, l_path, n_node_overlap),
            set_labels=('guessed states', 'real states')
        )
        plt.title('Overlap of states based on data\nno total reference', fontsize=21)
        if save_fig:
            plt.savefig('figures/power_of_%d/naive_venn_%d.png' % (n_pos, i_iter))
            plt.close()
        else:
            plt.show()

        # Try reconstructing by taking the most likely nodes and connect them using graph structure
        idc1, = np.where(scaling1 > .9)
        idc2, = np.where(scaling2 > .9)
        idc3, = np.where(scaling3 > .9)
        idc_all = np.unique(np.concatenate([idc1, idc2, idc3]).reshape(-1))
        likely_paths = determine_paths(graph, idc1, idc3)
        if use_weighting:
            weighting = calc_weights(idc2)
            sort_fun = lambda x_list: np.sum(np.isin(np.unique(idc2.reshape(-1)), x_list).astype('float')
                                             * weighting)
        else:
            sort_fun = lambda x_list: np.sum(np.isin(np.unique(idc2.reshape(-1)), x_list).astype('float'))

        likely_paths = list(sorted(
            likely_paths,
            key=sort_fun,
            reverse=True)
        )[:n_path]

        for i_path, lp in enumerate(likely_paths):
            n_lp_overlap = len(set(lp).intersection(path))
            print('Overlap No Reference Structure: %d' % n_lp_overlap)
            with open(
                    'data/power_of_%d/path_determination/noref_structure_%d_%s_path%d.txt' % (
                            n_pos,
                            i_iter,
                            weighting_str,
                            i_path
                    ), 'w') as noref_structure_file:
                noref_structure_file.write('%d\n' % len(lp))
                noref_structure_file.write('%d\n' % l_path)
                noref_structure_file.write('%d\n' % n_lp_overlap)
            venn2(
                subsets=(len(lp), l_path, n_lp_overlap),
                set_labels=('guessed states', 'real states')
            )
            plt.title('Overlap of states based on network structure\nno total reference')
            if save_fig:
                plt.savefig('figures/power_of_%d/noref_venn_%d_path%d.png' % (n_pos, i_iter, i_path))
                plt.close()
            else:
                plt.show()
        plt.figure()
        nx.draw_networkx_nodes(
            graph,
            pos=node_pos,
            nodelist=idc_all.tolist(),
            node_color=scaling2[idc_all],
            cmap='Reds',
        )
        nx.draw_networkx_nodes(
            graph,
            pos=node_pos,
            nodelist=path,
            node_color='tab:blue'
        )
        plot_edges(graph, node_pos, likely_paths, ax=plt.gca(), n_path=n_path, cmap=cmap)
        plt.title('Connectivity based on most likely path\nno total reference', fontsize=21)
        plt.tight_layout()
        plt.subplots_adjust(right=.85)
        cax = plt.axes([.87, 0.1, 0.03, 0.7])
        plt.colorbar(sm, cax=cax, orientation='vertical', label='weighting')
        if save_fig:
            plt.savefig('figures/power_of_%d/noref_structure_%d.png' % (n_pos, i_iter))
            plt.close()
        else:
            plt.show()

        possible_state1 = to_int(decompose_states(data_sample1, sample_size))
        possible_state2 = to_int(decompose_states(data_sample2, sample_size))
        possible_state3 = to_int(decompose_states(data_sample3, sample_size))
        all_simple_paths = determine_paths(graph, possible_state1.reshape(-1), possible_state3.reshape(-1))
        if use_weighting:
            weighting = calc_weights(possible_state2)
            sort_fun = lambda x_list: np.sum(np.isin(np.unique(possible_state2.reshape(-1)), x_list).astype('float')
                                             * weighting)
        else:
            sort_fun = lambda x_list: np.sum(np.isin(np.unique(possible_state2.reshape(-1)), x_list).astype('float'))

        all_simple_paths = list(sorted(
            all_simple_paths,
            key=sort_fun,
            reverse=True)
        )[:n_path]

        for i_path, sp in enumerate(all_simple_paths):
            n_sp_overlap = len(set(sp).intersection(path))
            print('Overlap All Reference Structure: %d' % n_sp_overlap)
            with open(
                    'data/power_of_%d/path_determination/allref_structure_%d_%s_path%d.txt' % (
                            n_pos,
                            i_iter,
                            weighting_str,
                            i_path
                    ), 'w') as allref_structure_file:
                allref_structure_file.write('%d\n' % len(sp))
                allref_structure_file.write('%d\n' % l_path)
                allref_structure_file.write('%d\n' % n_sp_overlap)
            venn2(
                subsets=(len(sp), l_path, n_sp_overlap),
                set_labels=('guessed states', 'real states')
            )
            plt.title('Overlap of states based on network structure\ntotal reference')
            if save_fig:
                plt.savefig('figures/power_of_%d/allref_venn_%d_path%d.png' % (n_pos, i_iter, i_path))
                plt.close()
            else:
                plt.show()

        lengths = np.array(list(map(len, all_simple_paths)), dtype='float')
        print('Lengths of found paths: ', lengths)
        lengths /= np.max(lengths)
        plt.figure(figsize=(18, 7))
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
        plot_edges(graph, node_pos, all_simple_paths, ax=plt.gca(), n_path=n_path, cmap=cmap)
        plt.title('Connectivity based on most likely path\ntotal reference')
        plt.tight_layout()
        plt.subplots_adjust(right=.85)
        cax = plt.axes([.87, 0.1, 0.03, 0.7])
        plt.colorbar(sm, cax=cax, orientation='vertical', label='weighting')
        if save_fig:
            plt.savefig('figures/power_of_%d/allref_structure_%d.png' % (n_pos, i_iter))
            plt.close()
        else:
            plt.show()


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


