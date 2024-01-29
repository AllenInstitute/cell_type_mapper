import anndata
import h5py
import numpy as np
import time

import argparse

from cell_type_mapper.utils.distance_utils import (
    correlation_dot)

from cell_type_mapper.cell_by_gene.utils import (
    convert_to_cpm)

from correlate_cells import correlate_cells


def main():

    file0 = 'data/cartoon_a.h5ad'
    file1 = 'data/cartoon_b.h5ad'
    result_file = 'scratch/results.h5ad'

    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--file0',
            type=str,
            default=file0,
            help='path to the first h5ad file to be correlated')

    parser.add_argument(
            '--norm0',
            type=str,
            default='raw',
            help='normalization (either "raw" or "log2CPM") of file0')

    parser.add_argument(
            '--file1',
            type=str,
            default=file1,
            help='path to the second h5ad file to be correlated')

    parser.add_argument(
            '--norm1',
            type=str,
            default='raw',
            help='normalization (either "raw" or "log2CPM") of file1')

    parser.add_argument(
            '--result_file',
            type=str,
            default=result_file,
            help='path to the HDF5 file that will be written')

    parser.add_argument(
            '--n_processors',
            type=int,
            default=None,
            help='number of processors to use')

    parser.add_argument(
            '--cells_at_a_time',
            type=int,
            default=None,
            help='number of cells to read from each file at a time')

    parser.add_argument(
        '--tmp_dir',
        type=str,
        default=None,
        help='path to directory where scratch files can be written')

    args = parser.parse_args()

    t0 = time.time()
    correlate_cells(
        anndata_path_0=args.file0,
        norm0=args.norm0,
        anndata_path_1=args.file1,
        norm1=args.norm1,
        n_processors=args.n_processors,
        cells_at_a_time=args.cells_at_a_time,
        output_path=args.result_file,
        tmp_dir=args.tmp_dir)
    dur = time.time()-t0

    print(f"====ALL DONE {dur:.2e} seconds for processing====")

if __name__ == "__main__":
    main()
