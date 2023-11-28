import json
import h5py
import multiprocessing
import numpy as np
import pathlib
import shutil
import tempfile
import time

from cell_type_mapper.utils.utils import (
    print_timing,
    _clean_up,
    mkstemp_clean,
    choose_int_dtype)

from cell_type_mapper.utils.multiprocessing_utils import (
    winnow_process_dict)

from cell_type_mapper.utils.stats_utils import (
    boring_t_from_p_value)

from cell_type_mapper.diff_exp.scores import (
    read_precomputed_stats,
    _get_this_cluster_stats,
    score_differential_genes)

from cell_type_mapper.utils.csc_to_csr import (
    transpose_sparse_matrix_on_disk)


def find_markers_for_all_taxonomy_pairs(
        precomputed_stats_path,
        taxonomy_tree,
        output_path,
        p_th=0.01,
        q1_th=0.5,
        qdiff_th=0.7,
        log2_fold_th=1.0,
        q1_min_th=0.1,
        qdiff_min_th=0.1,
        log2_fold_min_th=0.8,
        n_processors=4,
        tmp_dir=None,
        max_gb=20,
        exact_penetrance=False,
        n_valid=30,
        gene_list=None):
    """
    Create differential expression scores and validity masks
    for differential genes between all relevant pairs in a
    taxonomy*

    * relevant pairs are defined as nodes in the tree that are
    on the same level and share the same parent.

    Parameters
    ----------
    precomputed_stats_path:
        Path to HDF5 file containing precomputed stats for leaf nodes

    taxonomy_tree:
        instance of
        cell_type_mapper.taxonomty.taxonomy_tree.TaxonomyTree
        ecoding the taxonomy tree

    output_path:
        Path to the HDF5 file where results will be stored

    p_th/q1_th/qdiff_th/log2_fold_th
        Thresholds for determining if a gene is a valid marker.
        See Notes under score_differential_genes

    q1_min_th/qdiff_min_th/log2_fold_min_th
        Minimum thresholds below which genes will not be
        considered marker genes. See Notes under
        score_differential_genes.

    n_processors:
        Number of independent worker processes to spin out

    max_gb:
        maximum number of GB to load at once

    exact_penetrance:
        If False, allow genes that technically fail penetrance
        and fold-change thresholds to be marker genes.

    n_valid:
        The number of markers to find per pair (when using
        approximate penetrance test)

    gene_list:
        Optional list limiting the genes that can be considered
        as markers.

    Returns
    --------
    None
        Data is written to HDF5 file

    Notes
    -----
    HDF5 file will contain the following datasets
        'pair_to_idx' -> JSONized dict mapping [level][node1][node2] to row
        index in other data arrays

        'scores' -> (n_sibling_pairs, n_genes) array of differential expression
        scores

        'validity' -> (n_sibling_pairs, n_genes) array of booleans indicating
        whether or not the gene passed the validity thresholds

        'ranked_list' -> (n_sibling_pairs, n_genes) array of ints.
        Each row gives the ranked indexes of the discriminator genes,
        i.e. if ranked_list[2, :] = [9, 1,...., 101] then, for sibling
        pair at idx=2 (see pair_to_idx), gene_9 is the best discriminator,
        gene_1 is the second best discrminator, and gene_101 is the worst
        discriminator
    """

    tmp_dir = tempfile.mkdtemp(dir=tmp_dir, prefix='find_markers_')
    tmp_dir = pathlib.Path(tmp_dir)

    tmp_thinned_path = create_sparse_by_pair_marker_file(
        precomputed_stats_path=precomputed_stats_path,
        taxonomy_tree=taxonomy_tree,
        p_th=p_th,
        q1_th=q1_th,
        qdiff_th=qdiff_th,
        log2_fold_th=log2_fold_th,
        q1_min_th=q1_min_th,
        qdiff_min_th=qdiff_min_th,
        log2_fold_min_th=log2_fold_min_th,
        n_processors=n_processors,
        tmp_dir=tmp_dir,
        max_bytes=max(1024**2, np.round(max_gb*1024**3).astype(int)),
        exact_penetrance=exact_penetrance,
        n_valid=n_valid,
        gene_list=gene_list)

    with h5py.File(precomputed_stats_path, 'r') as in_file:
        n_genes = len(json.loads(
            in_file['col_names'][()].decode('utf-8')))

    add_sparse_by_gene_markers_to_file(
        h5_path=tmp_thinned_path,
        n_genes=n_genes,
        max_gb=max_gb,
        tmp_dir=tmp_dir)

    shutil.move(
        src=tmp_thinned_path,
        dst=output_path)

    _clean_up(tmp_dir)


def create_sparse_by_pair_marker_file(
        precomputed_stats_path,
        taxonomy_tree,
        p_th=0.01,
        q1_th=0.5,
        qdiff_th=0.7,
        log2_fold_th=1.0,
        q1_min_th=0.1,
        qdiff_min_th=0.1,
        log2_fold_min_th=0.8,
        n_processors=4,
        tmp_dir=None,
        max_bytes=6*1024**3,
        exact_penetrance=False,
        n_valid=30,
        gene_list=None):
    """
    Create differential expression scores and validity masks
    for differential genes between all relevant pairs in a
    taxonomy*

    * relevant pairs are defined as nodes in the tree that are
    on the same level and share the same parent.

    Parameters
    ----------
    precomputed_stats_path:
        Path to HDF5 file containing precomputed stats for leaf nodes

    taxonomy_tree:
        instance of
        cell_type_mapper.taxonomty.taxonomy_tree.TaxonomyTree
        ecoding the taxonomy tree

    p_th/q1_th/qdiff_th
        Thresholds for determining if a gene is a valid marker.
        See Notes under score_differential_genes

    n_processors:
        Number of independent worker processes to spin out

    max_bytes:
        Maximum number of bytes to load when thinning marker file

    exact_penetrance:
        If False, allow genes that technically fail penetrance
        and fold-change thresholds to be marker genes.

    n_valid:
        The number of markers to find per pair (when using
        approximate penetrance test)

    gene_list:
        Optional list limiting the genes that can be considered
        as markers.

    Returns
    --------
    Path to a file in tmp_dir where the data is stored

    Notes
    -----
    This method stores the markers as sparse arrays with taxonomic
    pairs as the indptr axis.
    """
    inner_tmp_dir = tempfile.mkdtemp(dir=tmp_dir)
    tmp_output_path = pathlib.Path(
        mkstemp_clean(
            dir=inner_tmp_dir,
            prefix='unthinned_',
            suffix='.h5'))

    tree_as_leaves = taxonomy_tree.as_leaves

    precomputed_stats = read_precomputed_stats(
           precomputed_stats_path=precomputed_stats_path,
           taxonomy_tree=taxonomy_tree,
           for_marker_selection=True)
    cluster_stats = precomputed_stats['cluster_stats']
    gene_names = precomputed_stats['gene_names']
    del precomputed_stats

    if gene_list is not None:
        gene_set = set(gene_list)
        valid_gene_idx = np.array([
            idx for idx, g in enumerate(gene_names)
            if g in gene_set
        ])

        if len(valid_gene_idx) == 0:
            msg = (
                "Genes in query data file do not overlap genes in "
                "reference data file.\n"
                f"example query genes: {gene_list[:10]}\n"
                f"example reference genes: {gene_names[:10]}\n"
            )
            raise RuntimeError(msg)

    else:
        valid_gene_idx = None

    n_genes = len(gene_names)

    idx_to_pair = _prep_output_file(
            output_path=tmp_output_path,
            taxonomy_tree=taxonomy_tree,
            gene_names=gene_names)

    idx_values = list(idx_to_pair.keys())
    idx_values.sort()

    process_dict = {}
    tmp_path_dict = {}
    n_pairs = len(idx_to_pair)

    # how many pairs to run per proceess
    n_per = min(1000000, n_pairs//(2*n_processors))
    n_per -= (n_per % 8)
    n_per = max(8, n_per)
    t0 = time.time()
    ct_complete = 0

    for col0 in range(0, n_pairs, n_per):
        col1 = col0+n_per
        tmp_path = mkstemp_clean(
            dir=inner_tmp_dir,
            prefix=f'columns_{col0}_{col1}_',
            suffix='.h5')
        tmp_path_dict[col0] = pathlib.Path(tmp_path)

        this_idx_values = idx_values[col0:col1]
        this_idx_to_pair = {
            ii: idx_to_pair.pop(ii)
            for ii in this_idx_values}

        (this_cluster_stats,
         this_tree_as_leaves) = _get_this_cluster_stats(
            cluster_stats=cluster_stats,
            idx_to_pair=this_idx_to_pair,
            tree_as_leaves=tree_as_leaves)

        p = multiprocessing.Process(
                target=_find_markers_worker,
                kwargs={
                    'cluster_stats': this_cluster_stats,
                    'tree_as_leaves': this_tree_as_leaves,
                    'idx_to_pair': this_idx_to_pair,
                    'n_genes': n_genes,
                    'p_th': p_th,
                    'q1_th': q1_th,
                    'qdiff_th': qdiff_th,
                    'log2_fold_th': log2_fold_th,
                    'q1_min_th': q1_min_th,
                    'qdiff_min_th': qdiff_min_th,
                    'log2_fold_min_th': log2_fold_min_th,
                    'tmp_path': tmp_path,
                    'exact_penetrance': exact_penetrance,
                    'n_valid': n_valid,
                    'valid_gene_idx': valid_gene_idx})
        p.start()
        process_dict[col0] = p
        while len(process_dict) >= n_processors:
            n0 = len(process_dict)
            process_dict = winnow_process_dict(process_dict)
            n1 = len(process_dict)
            if n1 < n0:
                ct_complete += (n0-n1)*n_per
                print_timing(
                    t0=t0,
                    i_chunk=ct_complete,
                    tot_chunks=n_pairs,
                    unit='hr')

    del cluster_stats
    del tree_as_leaves
    del this_cluster_stats
    del this_idx_values
    del this_idx_to_pair
    del this_tree_as_leaves

    while len(process_dict) > 0:
        n0 = len(process_dict)
        process_dict = winnow_process_dict(process_dict)
        n1 = len(process_dict)
        if n1 < n0:
            ct_complete += (n0-n1)*n_per
            print_timing(
                t0=t0,
                i_chunk=ct_complete,
                tot_chunks=n_pairs,
                unit='hr')

    n_up_indices = 0
    n_down_indices = 0
    for col0 in tmp_path_dict:
        tmp_path = tmp_path_dict[col0]
        with h5py.File(tmp_path, 'r') as src:
            n_up_indices += src['n_up_indices'][()]
            n_down_indices += src['n_down_indices'][()]

    gene_idx_dtype = choose_int_dtype((0, n_genes))
    up_pair_idx_dtype = choose_int_dtype((0, n_up_indices))
    down_pair_idx_dtype = choose_int_dtype((0, n_down_indices))

    up_pair_offset = 0
    down_pair_offset = 0
    with h5py.File(tmp_output_path, 'a') as dst:

        dst_grp = dst.create_group('sparse_by_pair')

        dst_grp.create_dataset(
            'up_pair_idx',
            shape=(n_pairs+1,),
            dtype=up_pair_idx_dtype)
        dst_grp.create_dataset(
            'up_gene_idx',
            shape=(n_up_indices,),
            dtype=gene_idx_dtype)
        dst_grp.create_dataset(
            'down_pair_idx',
            shape=(n_pairs+1,),
            dtype=down_pair_idx_dtype)
        dst_grp.create_dataset(
            'down_gene_idx',
            shape=(n_down_indices,),
            dtype=gene_idx_dtype)

        col0_values = list(tmp_path_dict.keys())
        col0_values.sort()
        for col0 in col0_values:
            tmp_path = tmp_path_dict[col0]
            with h5py.File(tmp_path, 'r') as src:
                pair_idx_values = json.loads(
                    src['pair_idx_values'][()].decode('utf-8'))
                pair_idx_values.sort()

                up_gene_idx = \
                    src['up_gene_idx'][()].astype(
                        gene_idx_dtype)

                down_gene_idx = \
                    src['down_gene_idx'][()].astype(
                        gene_idx_dtype)

                up_pair_idx = \
                    src['up_pair_idx'][()].astype(up_pair_idx_dtype) \
                    + up_pair_offset

                down_pair_idx = \
                    src['down_pair_idx'][()].astype(down_pair_idx_dtype) \
                    + down_pair_offset

                i0 = min(pair_idx_values)
                i1 = i0 + len(pair_idx_values)
                dst_grp['up_pair_idx'][i0:i1] = up_pair_idx[:-1]
                dst_grp['down_pair_idx'][i0:i1] = down_pair_idx[:-1]

                i0 = up_pair_idx[0]
                i1 = i0 + len(up_gene_idx)
                dst_grp['up_gene_idx'][i0:i1] = up_gene_idx

                i0 = down_pair_idx[0]
                i1 = i0 + len(down_gene_idx)
                dst_grp['down_gene_idx'][i0:i1] = down_gene_idx

                up_pair_offset += len(up_gene_idx)
                down_pair_offset += len(down_gene_idx)

        dst_grp['up_pair_idx'][-1] = n_up_indices
        dst_grp['down_pair_idx'][-1] = n_down_indices

    return tmp_output_path


def add_sparse_by_gene_markers_to_file(
        h5_path,
        n_genes,
        max_gb,
        tmp_dir):
    """
    Add the "sparse_by_gene" representation of markers to
    a marker file that already contains the
    "sparse_by_pairs" representation.
    """

    tmp_dir = pathlib.Path(tempfile.mkdtemp(dir=tmp_dir))

    with h5py.File(h5_path, 'a') as dst:
        dst.create_group('sparse_by_gene')

    for direction in ('up', 'down'):
        transposed_path = mkstemp_clean(
            dir=tmp_dir,
            prefix='transposed_',
            suffix='.h5')

        with h5py.File(h5_path, 'r') as src:
            transpose_sparse_matrix_on_disk(
                indices_handle=src[f'sparse_by_pair/{direction}_gene_idx'],
                indptr_handle=src[f'sparse_by_pair/{direction}_pair_idx'],
                data_handle=None,
                n_indices=n_genes,
                max_gb=max_gb,
                output_path=transposed_path)

        with h5py.File(transposed_path, 'r') as src:
            with h5py.File(h5_path, 'a') as dst:
                grp = dst['sparse_by_gene']
                grp.create_dataset(
                    f'{direction}_gene_idx',
                    data=src['indptr'],
                    chunks=src['indptr'].chunks)
                grp.create_dataset(
                    f'{direction}_pair_idx',
                    data=src['indices'],
                    chunks=src['indices'].chunks)

    _clean_up(tmp_dir)


def _find_markers_worker(
        cluster_stats,
        tree_as_leaves,
        idx_to_pair,
        n_genes,
        p_th,
        q1_th,
        qdiff_th,
        log2_fold_th,
        q1_min_th,
        qdiff_min_th,
        log2_fold_min_th,
        tmp_path,
        exact_penetrance=False,
        n_valid=30,
        valid_gene_idx=None):
    """
    Score and rank differentiallly expressed genes for
    a subset of taxonomic siblings. Write the results to
    a temporary HDF5 file, then write the contents of that
    file to the final output file.

    Parameters
    ----------
    cluster_stats:
        Result of read_precomputed_stats (just 'cluster_stats')
    tree_as_leaves:
        Result of convert_tree_to_leaves
    idx_to_pair:
        Dict mapping col in final output file to
        (level, node1, node2) sibling pair
        [Just the columns that this worker is responsible for]
    n_genes:
        Number of genes in dataset
    p_th/q1_th/qdiff_th
        Thresholds for determining if a gene is a valid marker.
        See Notes under score_differential_genes
    tmp_path:
        Path to temporary HDF5 file where results for this worker
        will be stored (this process creates that file)
    exact_penetrance:
        If False, allow genes that technically fail penetrance
        and fold-change thresholds to be marker genes.
    valid_gene_idx:
        Optional array of gene indices indicating which genes
        can be considered valid markers
    """

    n_genes = len(cluster_stats[list(cluster_stats.keys())[0]]['mean'])
    idx_dtype = choose_int_dtype((0, n_genes))

    boring_t = boring_t_from_p_value(p_th)

    idx_values = list(idx_to_pair.keys())
    idx_values.sort()
    col0 = min(idx_values)
    if col0 % 8 != 0:
        raise RuntimeError(
            f"col0 ({col0}) is not an integer multiple of 8")

    up_reg_lookup = dict()
    down_reg_lookup = dict()
    for idx in idx_values:
        sibling_pair = idx_to_pair[idx]
        level = sibling_pair[0]
        node1 = sibling_pair[1]
        node2 = sibling_pair[2]

        (scores,
         validity_mask,
         up_mask) = score_differential_genes(
                         node_1=f'{level}/{node1}',
                         node_2=f'{level}/{node2}',
                         precomputed_stats=cluster_stats,
                         p_th=p_th,
                         q1_th=q1_th,
                         qdiff_th=qdiff_th,
                         log2_fold_th=log2_fold_th,
                         q1_min_th=q1_min_th,
                         qdiff_min_th=qdiff_min_th,
                         log2_fold_min_th=log2_fold_min_th,
                         boring_t=boring_t,
                         exact_penetrance=exact_penetrance,
                         n_valid=n_valid,
                         valid_gene_idx=valid_gene_idx)

        up_reg_lookup[idx] = np.where(
            np.logical_and(validity_mask, up_mask))[0].astype(idx_dtype)
        down_reg_lookup[idx] = np.where(
            np.logical_and(validity_mask,
                           np.logical_not(up_mask)))[0].astype(idx_dtype)

    (up_pair_idx,
     up_gene_idx) = _lookup_to_sparse(up_reg_lookup)

    (down_pair_idx,
     down_gene_idx) = _lookup_to_sparse(down_reg_lookup)

    valid_gene_idx = set(up_gene_idx)
    valid_gene_idx = valid_gene_idx.union(
        set(down_gene_idx))

    n_up_indices = len(up_gene_idx)
    n_down_indices = len(down_gene_idx)

    pair_idx_values = list(up_reg_lookup.keys())
    pair_idx_values.sort()
    with h5py.File(tmp_path, 'a') as out_file:
        out_file.create_dataset(
            'pair_idx_values',
            data=json.dumps(pair_idx_values).encode('utf-8'))

        out_file.create_dataset(
            'up_pair_idx', data=up_pair_idx, dtype=up_pair_idx.dtype)

        out_file.create_dataset(
            'up_gene_idx', data=up_gene_idx, dtype=up_gene_idx.dtype)

        out_file.create_dataset(
            'down_pair_idx', data=down_pair_idx, dtype=down_pair_idx.dtype)

        out_file.create_dataset(
            'down_gene_idx', data=down_gene_idx, dtype=down_gene_idx.dtype)

        out_file.create_dataset(
            'valid_gene_idx',
            data=np.array(list(valid_gene_idx)).astype(idx_dtype),
            dtype=idx_dtype)

        out_file.create_dataset(
            'n_down_indices', data=n_down_indices)

        out_file.create_dataset(
            'n_up_indices', data=n_up_indices)


def _prep_output_file(
       output_path,
       taxonomy_tree,
       gene_names):
    """
    Create the HDF5 file where the differential gene scoring stats
    will be stored.

    Parameters
    ----------
    output_path:
        Path to the HDF5 file
    taxonomy_tree:
        instance of
        cell_type_mapper.taxonomty.taxonomy_tree.TaxonomyTree
        ecoding the taxonomy tree
    gene_names:
        Ordered list of gene names for entire dataset

    Returns
    -------
    idx_to_pair:
        Dict mapping the row index of a sibling pair
        in the final output file to that sibling pair's
        (level, node1, node2) specification.

    Notes
    -----
    This method also creates the file at output_path with
    empty datasets for the stats that need to be saved.
    """
    siblings = taxonomy_tree.siblings

    idx_to_pair = dict()
    pair_to_idx_out = dict()
    for idx, sibling_pair in enumerate(siblings):
        level = sibling_pair[0]
        node1 = sibling_pair[1]
        node2 = sibling_pair[2]
        idx_to_pair[idx] = sibling_pair

        if level not in pair_to_idx_out:
            pair_to_idx_out[level] = dict()
        if node1 not in pair_to_idx_out[level]:
            pair_to_idx_out[level][node1] = dict()
        if node2 not in pair_to_idx_out[level]:
            pair_to_idx_out[level][node2] = dict()

        pair_to_idx_out[level][node1][node2] = idx

    with h5py.File(output_path, 'w') as out_file:
        out_file.create_dataset(
            'gene_names',
            data=json.dumps(gene_names).encode('utf-8'))

        out_file.create_dataset(
            'pair_to_idx',
            data=json.dumps(pair_to_idx_out).encode('utf-8'))

        out_file.create_dataset(
            'n_pairs',
            data=len(idx_to_pair))

    return idx_to_pair


def _lookup_to_sparse(
        indptr_to_indices):
    """
    Map a lookup of indptr idx to indices to a sparse
    matrix array
    """
    n_indices = 0
    max_indices = 0
    for idx in indptr_to_indices:
        n_indices += len(indptr_to_indices[idx])
        if len(indptr_to_indices[idx]) > 0:
            this_max = max(indptr_to_indices[idx])
            if this_max > max_indices:
                max_indices = this_max
    indptr_dtype = choose_int_dtype((0, n_indices))
    indices_dtype = choose_int_dtype((0, max_indices))
    indptr = np.zeros(len(indptr_to_indices)+1, dtype=indptr_dtype)
    indices = np.zeros(n_indices, dtype=indices_dtype)
    idx_list = list(indptr_to_indices.keys())
    idx_list.sort()
    indptr_val = 0
    for local_idx, global_idx in enumerate(idx_list):
        this_indices = indptr_to_indices[global_idx]
        indptr[local_idx] = indptr_val
        indices[indptr_val:indptr_val+len(this_indices)] = this_indices
        indptr_val += len(this_indices)

    indptr[-1] = n_indices

    return indptr, indices
