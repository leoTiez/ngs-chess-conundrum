import sys
import os
import multiprocessing
from pathlib import Path
from argparse import ArgumentParser


def parse_args(args):
    parser = ArgumentParser('Run experiments for the chess problem without machine learning')
    parser.add_argument('--n_cpus', type=int, default=1,
                        help='Number of cpus')
    return parser.parse_args(args)


def main(args):
    n_cpus = args.n_cpus
    npos = range(6, 12, 1)
    beta_concentration = [1., 10., 100., 1000.]
    do_weight = [False, True]

    Path('log').mkdir(exist_ok=True, parents=True)
    with multiprocessing.Pool(processes=n_cpus) as parallel:
        for np in npos:
            for bc in beta_concentration:
                for dw in do_weight:
                    weight_string = '--use_weighting ' if do_weight else ''
                    save_string = '%d_%d_%s' % (np, bc, 'weights' if dw else 'noweights')
                    parallel.apply_async(os.system, args=(
                        'python3.8 chess.py '
                        '--n_pos=%d '
                        '--beta_concentration=%s '
                        '%s'
                        '--do_plot_network '
                        '--save_prefix=%s '
                        '--save_fig  1> log/out_%s.out 2> log/err_%s.err' % (
                            np,
                            bc,
                            weight_string,
                            save_string,
                            save_string,
                            save_string
                        ),
                    ))

        parallel.close()
        parallel.join()


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))


