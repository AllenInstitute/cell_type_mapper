from typing import Union, List, Optional, Dict
import h5py
import multiprocessing
import numpy as np
import numbers
import pathlib
import tempfile

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.utils.multiprocessing_utils import (
    winnow_process_list)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad)

from cell_type_mapper.anndata_iterator.anndata_iterator import (
    AnnDataRowIterator)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.utils.stats_utils import (
    summary_stats_for_chunk)

from cell_type_mapper.diff_exp.precompute import (
    _create_empty_stats_file)

from cell_type_mapper.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)


def precompute_summary_stats_from_h5ad(
        data_path: Union[str, pathlib.Path],
        column_hierarchy: Optional[List[str]],
        taxonomy_tree: Optional[TaxonomyTree],
        output_path: Union[str, pathlib.Path],
        rows_at_a_time: int = 10000,
        normalization="log2CPM",
        tmp_dir=None,
        n_processors=3):
    """
    Precompute the summary stats used to identify marker genes

    Parameters
    ----------
    data_path:
        Path to the h5ad file containing the cell x gene matrix

    column_hierarcy:
        The list of columns denoting taxonomic classes,
        ordered from highest (parent) to lowest (child).

    taxonomy_tree:
        instance of
        cell_type_mapper.taxonomty.taxonomy_tree.TaxonomyTree
        ecoding the taxonomy tree

    output_path:
        Path to the HDF5 file that will contain the lookup
        information for the clusters

    rows_at_a_time:
        Number of rows to load at once from the cell x gene
        matrix

    normalization:
        The normalization of the cell by gene matrix in
        the input file; either 'raw' or 'log2CPM'

    tmp_dir:
        Directory where scratch files can be written

    n_processors:
        Number of parallel worker processes to spin up

    Note
    ----
    Assumes that the leaf level of the tree lists the rows in a single
    h5ad file that belong to that cluster
    """
    if taxonomy_tree is not None and column_hierarchy is not None:
        raise RuntimeError(
            "Cannot specify taxonomy_tree and column_hierarchy")

    if taxonomy_tree is None:
        taxonomy_tree = TaxonomyTree.from_h5ad(
            h5ad_path=data_path,
            column_hierarchy=column_hierarchy)

    precompute_summary_stats_from_h5ad_and_tree(
        data_path=data_path,
        taxonomy_tree=taxonomy_tree,
        output_path=output_path,
        rows_at_a_time=rows_at_a_time,
        normalization=normalization,
        tmp_dir=tmp_dir,
        n_processors=n_processors)


def precompute_summary_stats_from_h5ad_and_tree(
        data_path: Union[str, pathlib.Path],
        taxonomy_tree: TaxonomyTree,
        output_path: Union[str, pathlib.Path],
        rows_at_a_time: int = 10000,
        normalization='log2CPM',
        tmp_dir=None,
        n_processors=3):
    """
    Precompute the summary stats used to identify marker genes

    Parameters
    ----------
    data_path:
        Path to the h5ad file containing the cell x gene matrix

    taxonomy_tree:
        instance of
        cell_type_mapper.taxonomty.taxonomy_tree.TaxonomyTree
        ecoding the taxonomy tree

    output_path:
        Path to the HDF5 file that will contain the lookup
        information for the clusters

    rows_at_a_time:
        Number of rows to load at once from the cell x gene
        matrix

    normalization:
        The normalization of the cell by gene matrix in
        the input file; either 'raw' or 'log2CPM'

    tmp_dir:
        Directory where scratch files can be written

    n_processors:
        Number of parallel worker processes to spin up

    Note
    ----
    Assumes that the leaf level of the tree lists the rows in a single
    h5ad file that belong to that cluster
    """
    cluster_to_input_row = taxonomy_tree.leaf_to_cells

    cluster_list = list(cluster_to_input_row)
    cluster_list.sort()
    cluster_to_output_row = {c: int(ii)
                             for ii, c in enumerate(cluster_list)}

    obs = read_df_from_h5ad(data_path, 'obs')
    cell_name_list = list(obs.index.values)
    cell_name_to_cluster_name = dict()
    for cluster_name in cluster_to_output_row:
        for cell_idx in cluster_to_input_row[cluster_name]:
            if not isinstance(cell_idx, numbers.Integral):
                raise RuntimeError(
                    f"cell_idx is of type {type(cell_idx)}\n"
                    "this TaxonomyTree does not seem to map leaf "
                    "nodes to row indexes")
            cell_name = cell_name_list[cell_idx]
            cell_name_to_cluster_name[cell_name] = cluster_name

    precompute_summary_stats_from_h5ad_and_lookup(
        data_path_list=[data_path],
        cell_name_to_cluster_name=cell_name_to_cluster_name,
        cluster_to_output_row=cluster_to_output_row,
        output_path=output_path,
        rows_at_a_time=rows_at_a_time,
        normalization=normalization,
        tmp_dir=tmp_dir,
        n_processors=n_processors)

    with h5py.File(output_path, 'a') as out_file:
        out_file.create_dataset(
            'taxonomy_tree',
            data=taxonomy_tree.to_str().encode('utf-8'))


def precompute_summary_stats_from_h5ad_list_and_tree(
        data_path_list: List[Union[str, pathlib.Path]],
        taxonomy_tree: TaxonomyTree,
        output_path: Union[str, pathlib.Path],
        rows_at_a_time: int = 10000,
        normalization='log2CPM',
        cell_set=None,
        tmp_dir=None,
        n_processors=3):
    """
    Precompute the summary stats used to identify marker genes

    Parameters
    ----------
    data_path_list:
        List of paths to the h5ad files containing
        the cell x gene matrix

    taxonomy_tree:
        instance of
        cell_type_mapper.taxonomty.taxonomy_tree.TaxonomyTree
        ecoding the taxonomy tree

    output_path:
        Path to the HDF5 file that will contain the lookup
        information for the clusters

    rows_at_a_time:
        Number of rows to load at once from the cell x gene
        matrix

    normalization:
        The normalization of the cell by gene matrix in
        the input file; either 'raw' or 'log2CPM'

    cell_set:
        Optional set of cell identifiers. If not None, only cells
        in this set will be included in the precomputed stats
        computation.

    tmp_dir:
        Directory where scratch files can be written

    n_processors:
        Number of parallel worker processes to spin up

    Note
    ----
    Assumes that the leaf level of the tree lists the the names of
    cells that belong to a given cluster
    """

    leaf_to_cells = taxonomy_tree.leaf_to_cells

    cluster_list = list(leaf_to_cells.keys())
    cluster_list.sort()
    cluster_to_output_row = {c: int(ii)
                             for ii, c in enumerate(cluster_list)}

    cell_name_to_cluster_name = dict()
    for cluster in leaf_to_cells:
        for cell in leaf_to_cells[cluster]:
            cell_str = str(cell)
            if cell_set is None or cell_str in cell_set:
                cell_name_to_cluster_name[cell_str] = cluster

    precompute_summary_stats_from_h5ad_and_lookup(
        data_path_list=data_path_list,
        cell_name_to_cluster_name=cell_name_to_cluster_name,
        cluster_to_output_row=cluster_to_output_row,
        output_path=output_path,
        rows_at_a_time=rows_at_a_time,
        normalization=normalization,
        tmp_dir=tmp_dir,
        n_processors=n_processors)

    with h5py.File(output_path, 'a') as out_file:
        out_file.create_dataset(
            'taxonomy_tree',
            data=taxonomy_tree.to_str().encode('utf-8'))


def precompute_summary_stats_from_h5ad_and_lookup(
        data_path_list: List[Union[str, pathlib.Path]],
        cell_name_to_cluster_name: Dict[str, str],
        cluster_to_output_row: Dict[str, int],
        output_path: Union[str, pathlib.Path],
        rows_at_a_time: int = 10000,
        normalization='log2CPM',
        tmp_dir=None,
        n_processors=3):

    """
    Precompute the summary stats used to identify marker genes

    Parameters
    ----------
    data_path_list:
        List of paths to the h5ad files containing the cell x gene
        data

    cell_name_to_cluster_name:
        dict mapping cell name to the name of the cluster it belongs
        to

    cluster_to_output_row:
        dict mapping cluster name to output row in the precomputed
        matrix

    output_path:
        Path to the HDF5 file that will contain the lookup
        information for the clusters

    rows_at_a_time:
        Number of rows to load at once from the cell x gene
        matrix

    normalization:
        The normalization of the cell by gene matrix in
        the input file; either 'raw' or 'log2CPM'

    tmp_dir:
        Directory where scratch files can be written

    n_processors:
        Number of parallel worker processes to spin up
    """
    tmp_dir = tempfile.mkdtemp(dir=tmp_dir)

    try:
        _precompute_summary_stats_from_h5ad_and_lookup(
            data_path_list=data_path_list,
            cell_name_to_cluster_name=cell_name_to_cluster_name,
            cluster_to_output_row=cluster_to_output_row,
            output_path=output_path,
            rows_at_a_time=rows_at_a_time,
            normalization=normalization,
            tmp_dir=tmp_dir,
            n_processors=n_processors)
    finally:
        _clean_up(tmp_dir)


def _precompute_summary_stats_from_h5ad_and_lookup(
        data_path_list: List[Union[str, pathlib.Path]],
        cell_name_to_cluster_name: Dict[str, str],
        cluster_to_output_row: Dict[str, int],
        output_path: Union[str, pathlib.Path],
        rows_at_a_time: int = 10000,
        normalization='log2CPM',
        tmp_dir=None,
        n_processors=3):

    gene_names = None
    for pth in data_path_list:
        these_genes = list(read_df_from_h5ad(pth, 'var').index.values)
        if gene_names is None:
            gene_names = these_genes
        else:
            if gene_names != these_genes:
                raise RuntimeError(
                    f"{pth}\nhas gene_names\n{these_genes}\n"
                    f"which does not match\n{data_path_list[0]}\n"
                    f"genes\n{gene_names}")

    n_clusters = len(cluster_to_output_row)
    n_genes = len(gene_names)

    cell_name_to_output_row = dict()
    for cell_name in cell_name_to_cluster_name:
        cell_name_to_output_row[cell_name] = cluster_to_output_row[
            cell_name_to_cluster_name[cell_name]]

    bad_row_idx = -999

    desired_cells = set(cell_name_to_cluster_name.keys())
    buffer_path_list = []

    process_list = []

    for data_path in data_path_list:

        cell_name_list = list(
            read_df_from_h5ad(data_path, 'obs').index.values)

        n_overlap = len(set(cell_name_list).intersection(desired_cells))
        if n_overlap == 0:
            continue

        chunk_iterator = AnnDataRowIterator(
            h5ad_path=data_path,
            row_chunk_size=rows_at_a_time)

        print(f'loading {data_path}')

        for chunk in chunk_iterator:

            buffer_path = mkstemp_clean(
                dir=tmp_dir,
                prefix='precomputation_buffer_',
                suffix='.h5')
            buffer_path_list.append(buffer_path)

            if n_processors <= 1:

                _process_chunk(
                    chunk=chunk,
                    gene_names=gene_names,
                    cell_name_to_output_row=cell_name_to_output_row,
                    cell_name_list=cell_name_list,
                    bad_row_idx=bad_row_idx,
                    normalization=normalization,
                    n_clusters=n_clusters,
                    buffer_path=buffer_path)

            else:
                p = multiprocessing.Process(
                        target=_process_chunk,
                        kwargs={
                            'chunk': chunk,
                            'gene_names': gene_names,
                            'cell_name_to_output_row': cell_name_to_output_row,
                            'cell_name_list': cell_name_list,
                            'bad_row_idx': bad_row_idx,
                            'normalization': normalization,
                            'n_clusters': n_clusters,
                            'buffer_path': buffer_path
                        }
                    )

                p.start()

                process_list.append(p)

                while len(process_list) >= n_processors:
                    process_list = winnow_process_list(process_list)

    for p in process_list:
        p.join()

    _create_empty_stats_file(
        output_path=output_path,
        cluster_to_output_row=cluster_to_output_row,
        n_clusters=n_clusters,
        n_genes=n_genes,
        col_names=gene_names)

    with h5py.File(output_path, 'a') as out_file:
        for buffer_path in buffer_path_list:
            with h5py.File(buffer_path, 'r') as src:
                for k in src.keys():
                    if k == 'n_cells':
                        out_file[k][:] += src[k][()]
                    else:
                        out_file[k][:, :] += src[k][()]


def _process_chunk(
        chunk,
        gene_names,
        cell_name_to_output_row,
        cell_name_list,
        bad_row_idx,
        normalization,
        n_clusters,
        buffer_path):
    """
    Assemble the summary stats from a chunk of data and write it to
    an HDF5 file at buffer_path
    """

    n_genes = len(gene_names)

    buffer_dict = dict()
    buffer_dict['n_cells'] = np.zeros(n_clusters, dtype=int)
    buffer_dict['sum'] = np.zeros((n_clusters, n_genes), dtype=float)
    buffer_dict['sumsq'] = np.zeros((n_clusters, n_genes), dtype=float)
    buffer_dict['gt0'] = np.zeros((n_clusters, n_genes), dtype=int)
    buffer_dict['gt1'] = np.zeros((n_clusters, n_genes), dtype=int)
    buffer_dict['ge1'] = np.zeros((n_clusters, n_genes), dtype=int)

    r0 = chunk[1]
    r1 = chunk[2]
    cluster_chunk = np.array([
        cell_name_to_output_row[cell_name_list[idx]]
        if cell_name_list[idx] in cell_name_to_output_row
        else bad_row_idx
        for idx in range(r0, r1)
    ])
    for unq_cluster in np.unique(cluster_chunk):
        if unq_cluster == bad_row_idx:
            continue
        valid = np.where(cluster_chunk == unq_cluster)[0]
        valid = np.sort(valid)
        this_cluster = chunk[0][valid, :]

        if not isinstance(this_cluster, np.ndarray):
            this_cluster = this_cluster.toarray()

        this_cluster = CellByGeneMatrix(
            data=this_cluster,
            gene_identifiers=gene_names,
            normalization=normalization)

        if this_cluster.normalization != 'log2CPM':
            this_cluster.to_log2CPM_in_place()

        summary_chunk = summary_stats_for_chunk(this_cluster)

        for k in summary_chunk.keys():
            if len(buffer_dict[k].shape) == 1:
                buffer_dict[k][unq_cluster] += summary_chunk[k]
            else:
                buffer_dict[k][unq_cluster, :] += summary_chunk[k]

    with h5py.File(buffer_path, 'w') as dst:
        for k in buffer_dict:
            dst.create_dataset(k, data=buffer_dict[k])
