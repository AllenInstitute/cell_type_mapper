import h5py
import multiprocessing
import tempfile
import time

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.utils.multiprocessing_utils import (
    winnow_process_list)

from cell_type_mapper.utils.distance_utils import correlation_dot

from cell_type_mapper.utils.anndata_utils import(
    read_df_from_h5ad)

from cell_type_mapper.anndata_iterator.anndata_iterator import (
    AnnDataRowIterator)

from cell_type_mapper.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)

import argparse


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


def correlate_cells(
        anndata_path_0,
        norm0,
        anndata_path_1,
        norm1,
        n_processors,
        cells_at_a_time,
        output_path,
        tmp_dir=None):
    """
    Correlate all of the cells in one h5ad file against all of the cells
    in another h5ad file in gene space. Save the results to an HDF5 file.

    Parameters
    ----------
    anndata_path_0:
        Path to first h5ad file
    norm0:
        Normalization (either "raw" or "log2CPM") of the
        data in anndata_path_0
    anndata_path_1:
        Path to second h5ad file
    norm1:
        Normalization (either "raw" or "log2CPM") of the
        data in anndata_path_1
    n_processors:
        Number of worker processes to spin up at once
    cells_at_a_time:
        Number of cells to load from each anndata file
        to send to each worker (so, 2*cells_at_a_time
        cells will be loaded by each worker, one batch
        from anndata_path_0; one batch from anndata_path_1)
    output_path:
        Path to HDF5 file that will be written
    tmp_dir:
        Optional directory where scratch files will be written
    normalization:
        Either "raw" or "log2CPM" (the normalization of the cell by gene data)
    """
    tmp_dir = tempfile.mkdtemp(dir=tmp_dir)
    try:
        _correlate_cells(
            anndata_path_0=anndata_path_0,
            norm0=norm0,
            anndata_path_1=anndata_path_1,
            norm1=norm1,
            n_processors=n_processors,
            cells_at_a_time=cells_at_a_time,
            output_path=output_path,
            tmp_dir=tmp_dir)
    finally:
        _clean_up(tmp_dir)


def _correlate_cells(
        anndata_path_0,
        norm0,
        anndata_path_1,
        norm1,
        n_processors,
        cells_at_a_time,
        output_path,
        tmp_dir):

    n1 = len(read_df_from_h5ad(anndata_path_1, df_name='obs'))
    genes = read_df_from_h5ad(anndata_path_1, df_name='var').index.values
    

    data0 = AnnDataRowIterator(
                h5ad_path=anndata_path_0,
                row_chunk_size=cells_at_a_time,
                tmp_dir=tmp_dir)

    data1 = AnnDataRowIterator(
                h5ad_path=anndata_path_1,
                row_chunk_size=cells_at_a_time,
                tmp_dir=tmp_dir)

    process_list = []
    row_spec_to_path = dict()

    n0 = 0
    for chunk0 in data0:
        row_spec_0 = (chunk0[1], chunk0[2])
        n0 += chunk0[2]-chunk0[1]

        if row_spec_0 not in row_spec_to_path:
            row_spec_to_path[row_spec_0] = dict()

        chunk0 = CellByGeneMatrix(
                     data=chunk0[0],
                     gene_identifiers=genes,
                     normalization=norm0)

        for r0 in range(0, n1, cells_at_a_time):

            tmp_path = mkstemp_clean(
                dir=tmp_dir,
                suffix='.h5')

            r1 = min(n1, r0+cells_at_a_time)
            row_spec_1 = (r0, r1)

            row_spec_to_path[row_spec_0][row_spec_1] = tmp_path

            chunk1 = CellByGeneMatrix(
                        data=data1.get_chunk(r0=r0, r1=r1)[0],
                        gene_identifiers=genes,
                        normalization=norm1)
            config = {
                'cell_by_gene_0': chunk0,
                'cell_by_gene_1': chunk1,
                'output_path': tmp_path
            }
            if n_processors == 1:
                correlate_cells_worker(**config)
            else:
                p = multiprocessing.Process(
                    target=correlate_cells_worker,
                    kwargs=config)
                p.start()
                process_list.append(p)
                while len(process_list) >= n_processors:
                    process_list = winnow_process_list(process_list)

    while len(process_list) > 0:
        process_list = winnow_process_list(process_list)

    _merge_files(
        spec_to_path=row_spec_to_path,
        output_path=output_path)


def correlate_cells_worker(
        cell_by_gene_0,
        cell_by_gene_1,
        output_path):
    """
    Correlate the cells in cell_by_gene_0 against the cells in
    cell_by_gene_1 in gene space. Assumes that the two
    CellByGenMatrices have the same genes.

    Parameters
    ----------
    cell_by_gene_0:
        A CellByGeneMatrix containing the first set of
        cells to be correlated
    cell_by_gene_1:
        A CellByGeneMatrix containing the second set of
        cells to be correlated
    output_path:
        Path to an HDF5 file where the result will be
        stored.
    gpu_index:
       Index of GPU to run on (if any)
    """
    if cell_by_gene_0.normalization != 'log2CPM':
        cell_by_gene_0.to_log2CPM_in_place()
    if cell_by_gene_1.normalization != 'log2CPM':
        cell_by_gene_1.to_log2CPM_in_place()

    result = correlation_dot(
        arr0=cell_by_gene_0.data,
        arr1=cell_by_gene_1.data)

    with h5py.File(output_path, 'w') as dst:
        dst.create_dataset(
            'data',
            data=result)


def _merge_files(
        spec_to_path,
        output_path):
    """
    Parameters
    ----------
    spec_to_path:
        A dict mapping [(r0, r1)][(c0, c1)] to HDF5 path
    output_path:
        The path to the file that must be written
    """
    t0 = time.time()
    nrows = 0
    ncols = 0
    for kr in spec_to_path:
        if kr[1] > nrows:
            nrows = kr[1]
        for kc in spec_to_path[kr]:
            if kc[1] > ncols:
                ncols = kc[1]

    with h5py.File(output_path, 'w') as dst:
        dataset = dst.create_dataset(
            'data',
            shape=(nrows, ncols),
            dtype=float,
            chunks=(min(nrows, 1000), min(ncols, 1000)),
            compression=None,
            compression_opts=None)

        for row_key in spec_to_path:
            for col_key in spec_to_path[row_key]:
                with h5py.File(spec_to_path[row_key][col_key], 'r') as src:
                    dataset[row_key[0]:row_key[1],
                            col_key[0]:col_key[1]] = src['data'][()]
    print(f'merging took {time.time()-t0:.2e} seconds')


if __name__ == "__main__":
    main()
