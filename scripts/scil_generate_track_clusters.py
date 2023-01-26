#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import os
import logging

import json
import numpy as np
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('--input',
                   help='Input binary matrix containing significant connections to cluster.')
    p.add_argument('--run_decompose', action='store_true', default=True,
                   help="If True, script will run scil_save_connections_from_hdf5.py to \n"
                        "save raw connections.")
    p.add_argument('--hdf5', required=False,
                   help='HDF5 filename (.h5) for a single subject containing decomposed connections.')
    p.add_argument('--in_connections', required=False,
                   help='Folder containing the already saved .trk files for all connections. \n'
                        'Connections should be saved in the format X_Y.trk in order to be correctly \n'
                        'read by scil_streamlines_math.py.')
    p.add_argument('--output',
                   help='Main output folder. Output structure will be : \n'
                        '                    output/Raw_Connections/ (if --run_decompose)\n'
                        '                          /Clusters/ \n'
                        '                          /Clusters.json \n')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def track_clustering(mat):
    """
    Function to classify connections in clusters (defined as connections linking
    common regions).
    :param mat:         Binary matrix containing connections to sort. (.npy)
    :return:            Dictionary of clusters.
    """

    xindex = mat.shape[0]
    yindex = mat.shape[1]

    mat = np.triu(mat)

    if xindex == yindex:
        logging.info('Input matrix is symmetrical')
    else:
        logging.info('Input matrix is asymmetrical.')

    # Count number of non-zero connections in the matrix.
    unique, counts = np.unique(mat, return_counts=True)
    dic = dict(zip(unique, counts))
    logging.info('Matrix contains {} connections. Extracting connections ...'.format(dic[1]))

    # Extracting all connections with non-zero values in a table.
    x = []
    y = []
    for i in range(0, xindex):
        for j in range(0, yindex):
            if mat[i, j] == 1:
                x.append(i+1)
                y.append(j+1)
    pairs = np.rollaxis(np.array((x, y), dtype=int), 1)
    logging.info('Lookup table created. Clustering connections...')

    # Clustering connections.
    cluster_dict = {}
    n = 1
    while len(pairs) > 0:
        i = np.array(pairs[0][0])
        k = 1
        while k <= len(pairs):
            # Sorting clusters
            ix = pairs[np.isin(pairs[:, 0], i), 1]
            iy = pairs[np.isin(pairs[:, 1], i), 0]
            im = np.append(ix, iy)
            iu = np.unique(np.append(i, im))
            k += 1
            if np.array_equal(iu, i):
                break
            else:
                i = iu
        logging.info(f'Cluster #{n} took {k} iterations to extract.')
        # Saving cluster in dictionary as pairs (X_Y).
        x = pairs[np.isin(pairs[:, 1], i), 0]
        y = pairs[np.isin(pairs[:, 0], i), 1]
        assert len(x) == len(y), "Mismatch in number of starting and receiving regions."
        cluster_dict[f'Cluster_{n}'] = [f'{x[c]}_{y[c]}' for c in range(0, len(x))]
        pairs = np.extract(~np.isin(pairs, i), pairs)
        pairs = pairs.reshape(int(len(pairs)/2), 2)
        n += 1
    logging.info(f'{n-1} clusters extracted. No remaining connections.')

    return cluster_dict


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    assert_inputs_exist(parser, args.input, args.hdf5)
    assert_output_dirs_exist_and_empty(parser, args, args.output)

    # Create clusters.
    mat = np.load(args.input)
    cluster_dict = track_clustering(mat)
    with open(f'{args.output}/Cluster.json', 'w') as fp:
        json.dump(cluster_dict, fp)

    if args.run_decompose:
        if args.hdf5 is None:
            parser.error('To use --run_decompose, you need to provide a hdf5 file'
                         '(--hdf5).')

        # Creating output directory.
        out_dir_raw = os.path.join(args.output, 'Raw_Connections')
        os.mkdir(out_dir_raw)

        # Flattening the connections' list.
        flat_conn = [conn for cluster in list(cluster_dict.values()) for conn in cluster]

        # Run scil_save_connections_from_hdf5.py
        os.system(f'scil_save_connections_from_hdf5.py {args.hdf5} {out_dir_raw} --edge_keys {" ".join(flat_conn)} -f')
    else:
        out_dir_raw = args.in_connections

    # Merge individuals connections into one cluster files.
    out_dir = os.path.join(args.output, 'Clusters')
    os.mkdir(out_dir)

    for cluster in cluster_dict.keys():
        files = [f'{out_dir_raw}/{f}.trk' for f in cluster_dict[f'{cluster}']]

        os.system(f'scil_streamlines_math.py concatenate {" ".join(files)} {out_dir}/{cluster}.trk --robust -f {"-v" if args.verbose else ""}')


if __name__ == '__main__':
    main()
