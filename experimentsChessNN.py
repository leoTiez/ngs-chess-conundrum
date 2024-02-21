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
    n_epoch = 50000
    eps_decline = n_epoch / 4.
    npos = range(6, 12, 1)
    beta_concentration = [1., 10., 100., 1000.]
    n_train_sample = [10., 20., 50., 100., 200]
    n_iter = 500
    Path('log').mkdir(exist_ok=True, parents=True)
    with multiprocessing.Pool(processes=n_cpus) as parallel:
        for i_iter in range(n_iter):
            for np in npos:
                for bc in beta_concentration:
                    for nt in n_train_sample:
                        save_string = 'iter%d_%d_%d_%d' % (i_iter, np, bc, nt)
                        parallel.apply_async(os.system, args=(
                            'python3.8 chessNN.py '
                            '--n_pos=%d '
                            '--beta_concentration=%s '
                            '--train_sample_size=%d '
                            '--n_epoch=%s '
                            '--eps_decline=%s '
                            '--save_prefix=%s '
                            '--save_fig  1> log/out_nn_%s.out 2> log/err_nn_%s.err' % (
                                np,
                                bc,
                                nt,
                                n_epoch,
                                eps_decline,
                                save_string,
                                save_string,
                                save_string
                            ),
                        ))

        parallel.close()
        parallel.join()


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))


