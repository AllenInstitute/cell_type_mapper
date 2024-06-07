import h5py
import multiprocessing
import numpy as np
import scipy.sparse as scipy_sparse
import tempfile
import time

from cell_type_mapper.utils.utils import (
    print_timing,
    mkstemp_clean,
    choose_int_dtype,
    _clean_up)

from cell_type_mapper.utils.multiprocessing_utils import (
    winnow_process_dict)

from cell_type_mapper.utils.stats_utils import (
    boring_t_from_p_value)

from cell_type_mapper.diff_exp.score_utils import (
    read_precomputed_stats,
    _get_this_cluster_stats,
    pij_from_stats,
    q_score_from_pij)

from cell_type_mapper.diff_exp.scores import (
    diffexp_p_values_from_stats,
    penetrance_parameter_distance)

from cell_type_mapper.diff_exp.markers import (
    _prep_output_file)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)


def create_p_value_mask_file(
        precomputed_stats_path,
        dst_path,
        p_th=0.01,
        q1_th=0.5,
        q1_min_th=0.1,
        qdiff_th=0.7,
        qdiff_min_th=0.1,
        log2_fold_th=1.0,
        log2_fold_min_th=0.8,
        n_processors=4,
        tmp_dir=None,
        n_per=10000):
    """
    Create the "P-value mask" file, an HDF5 file storing
    the parameter space distance of each (cell type pair, gene)
    pair from the point in parameter space defining
    marker genes.

    Parameters
    ----------
    precomputed_stats_path:
        Path to HDF5 file containing precomputed stats for leaf nodes

    dst_path:
        Path to the HDF5 file that will be written

    p_th/q1_th/qdiff_th/log2_fold_th
        Thresholds for determining if a gene is a valid marker.
        See Notes under diffexp.scores.score_differential_genes

    q1_min_th/qdiff_min_th/log2_fold_min_th
        Minimum thresholds below which genes will not be
        considered marker genes. See Notes under
        diffexp.scores.score_differential_genes.

    n_processors:
        Number of independent worker processes to spin out

    tmp_dir:
        Path to a directory where scratch files can be written

    n_per:
        Number of rows to load at a time (per worker)

    Returns
    --------
    Noting. The file at dst_path is created.

    Notes
    -----
    The HDF5 file created by this function contains the following
    datasets.
         'pair_to_idx': a UTF-encoded JSON-serialized dict recording metadata
         mapping cell type pairs to row numbers. The dict is structured like
         {'taxonomy_level_1': [
                             'node1': {'node2': 0, 'node3': 1, ...},
                             'node2': {'node3': 2, ...}
                            ]
          'taxonomy_level_2':...
         }

        'gene_names': a UTF-encoded, JSON-serialized list of the names of
        the genes in this dataset (for mapping column index to gene name)

        'data'/'indices'/'inpdtr': they arrays needed to store the
        parameter space distance of the (cell type pair, gene) pairs
        from the "critical point" of marker gene membership, with
        cell type pair being the major axis (i.e. indptr[2:3]
        denotes the row associated with the 2nd cell type pair)

    (cell type pair, gene) pairs that pass the strict test for being
    a marker gene have a parameter space distance of -1.0. Pairs that
    are disqualified from being a marker because they violate the
    minimum statistical thresholds are absent from the sparse matrix
    (i.e. they have parameter space distances of zero).
    """
    tmp_dir = tempfile.mkdtemp(dir=tmp_dir)
    try:
        _create_p_value_mask_file(
            precomputed_stats_path=precomputed_stats_path,
            dst_path=dst_path,
            p_th=p_th,
            n_processors=n_processors,
            tmp_dir=tmp_dir,
            n_per=n_per,
            q1_th=q1_th,
            q1_min_th=q1_min_th,
            qdiff_th=qdiff_th,
            qdiff_min_th=qdiff_min_th,
            log2_fold_th=log2_fold_th,
            log2_fold_min_th=log2_fold_min_th)
    finally:
        _clean_up(tmp_dir)


def _create_p_value_mask_file(
        precomputed_stats_path,
        dst_path,
        p_th=0.01,
        n_processors=4,
        tmp_dir=None,
        n_per=10000,
        q1_th=0.5,
        q1_min_th=0.1,
        qdiff_th=0.7,
        qdiff_min_th=0.1,
        log2_fold_th=1.0,
        log2_fold_min_th=0.8):

    taxonomy_tree = TaxonomyTree.from_precomputed_stats(
        precomputed_stats_path)

    tree_as_leaves = taxonomy_tree.as_leaves

    precomputed_stats = read_precomputed_stats(
           precomputed_stats_path=precomputed_stats_path,
           taxonomy_tree=taxonomy_tree,
           for_marker_selection=True)

    cluster_stats = precomputed_stats['cluster_stats']
    gene_names = precomputed_stats['gene_names']

    del precomputed_stats

    n_genes = len(gene_names)

    idx_to_pair = _prep_output_file(
            output_path=dst_path,
            taxonomy_tree=taxonomy_tree,
            gene_names=gene_names)

    idx_values = list(idx_to_pair.keys())
    idx_values.sort()

    process_dict = {}
    tmp_path_list = []
    n_pairs = len(idx_to_pair)

    # how many pairs to run per proceess
    n_per -= (n_per % 8)
    n_per = max(8, n_per)
    t0 = time.time()
    ct_complete = 0

    for col0 in range(0, n_pairs, n_per):
        col1 = col0+n_per
        tmp_path = mkstemp_clean(
            dir=tmp_dir,
            prefix=f'columns_{col0}_{col1}_',
            suffix='.h5')
        tmp_path_list.append(tmp_path)

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
                target=_p_values_worker,
                kwargs={
                    'cluster_stats': this_cluster_stats,
                    'tree_as_leaves': this_tree_as_leaves,
                    'idx_to_pair': this_idx_to_pair,
                    'n_genes': n_genes,
                    'p_th': p_th,
                    'tmp_path': tmp_path,
                    'q1_th': q1_th,
                    'q1_min_th': q1_min_th,
                    'qdiff_th': qdiff_th,
                    'qdiff_min_th': qdiff_min_th,
                    'log2_fold_th': log2_fold_th,
                    'log2_fold_min_th': log2_fold_min_th})
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

    # do the joining
    _merge_masks(
        src_path_list=tmp_path_list,
        dst_path=dst_path)


def _p_values_worker(
        cluster_stats,
        tree_as_leaves,
        idx_to_pair,
        n_genes,
        p_th,
        tmp_path,
        q1_th=0.5,
        q1_min_th=0.1,
        qdiff_th=0.7,
        qdiff_min_th=0.1,
        log2_fold_th=1.0,
        log2_fold_min_th=0.8):
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

    p_th:
        Thresholds for determining if a gene is a valid marker.
        See Notes under score_differential_genes

    tmp_path:
        Path to temporary HDF5 file where results for this worker
        will be stored (this process creates that file)

    p_th/q1_th/qdiff_th/log2_fold_th
        Thresholds for determining if a gene is a valid marker.
        See Notes under diffexp.scores.score_differential_genes

    q1_min_th/qdiff_min_th/log2_fold_min_th
        Minimum thresholds below which genes will not be
        considered marker genes. See Notes under
        diffexp.scores.score_differential_genes.
    """

    (dense_mask,
     idx_values,
     boring_t,
     idx_dtype) = _prepare_mask(
        cluster_stats=cluster_stats,
        idx_to_pair=idx_to_pair,
        p_th=p_th)

    for pair_ct, idx in enumerate(idx_values):
        sibling_pair = idx_to_pair[idx]
        level = sibling_pair[0]
        node_1 = f'{level}/{sibling_pair[1]}'
        node_2 = f'{level}/{sibling_pair[2]}'

        p_values = diffexp_p_values_from_stats(
            node_1=node_1,
            node_2=node_2,
            precomputed_stats=cluster_stats,
            p_th=p_th,
            boring_t=boring_t,
            big_nu=None)

        (pij_1,
         pij_2,
         log2_fold) = pij_from_stats(
             cluster_stats=cluster_stats,
             node_1=node_1,
             node_2=node_2)

        (q1_score,
         qdiff_score) = q_score_from_pij(
             pij_1=pij_1,
             pij_2=pij_2)

        distances = penetrance_parameter_distance(
            q1_score=q1_score,
            qdiff_score=qdiff_score,
            log2_fold=log2_fold,
            q1_th=q1_th,
            q1_min_th=q1_min_th,
            qdiff_th=qdiff_th,
            qdiff_min_th=qdiff_min_th,
            log2_fold_th=log2_fold_th,
            log2_fold_min_th=log2_fold_min_th)

        wgt = distances['wgt']
        wgt = np.clip(
            a=wgt,
            a_min=0.0,
            a_max=np.finfo(np.float16).max-1)

        # so that genes with weighted distance == 0 get kept
        # in the sparse matrix
        wgt[wgt == 0.0] = -1.0
        eps = np.finfo(np.float16).resolution
        wgt[np.abs(wgt) < eps] = eps

        valid = (p_values < p_th)

        # so that invalid genes (according to penetrance min
        # thresholds do not get carried over into the sparse
        # matrix
        valid[distances['invalid']] = False

        dense_mask[pair_ct, valid] = wgt[valid]

    _save_sub_mask(
        dense_mask=dense_mask,
        idx_dtype=idx_dtype,
        idx_values=idx_values,
        dst_path=tmp_path)


def _prepare_mask(
        cluster_stats,
        idx_to_pair,
        p_th):
    """
    Find a bunch of boilerplate parameters needed for P-value mask
    creation.

    Parameters
    ----------
    cluster_stats:
        Result of read_precomputed_stats (just 'cluster_stats')

    idx_to_pair:
        Dict mapping col in final output file to
        (level, node1, node2) sibling pair
        [Just the columns that this worker is responsible for]

    p_th:
        Thresholds for determining if a gene is a valid marker.
        See Notes under score_differential_genes

    Returns
    -------
    dense_mask:
        np.array of zeros for storing the p-value mask
    idx_values:
        list of idx values (i.e. the indices of the cell type pairs
        stored in the mask)
    boring_t:
        to be passed to the welch_t_test function
    idx_dtype:
        dtype necessary to store the indices of the sparse matrix
    """

    n_genes = len(cluster_stats[list(cluster_stats.keys())[0]]['mean'])
    idx_dtype = choose_int_dtype((0, n_genes))

    boring_t = boring_t_from_p_value(p_th)

    idx_values = list(idx_to_pair.keys())
    idx_values.sort()
    idx_values = np.array(idx_values)

    # make sure these are consecutive
    delta = np.unique(np.diff(idx_values))
    if len(delta) != 1 or delta[0] != 1:
        raise RuntimeError(
            "p-value worker was passed non-consecutive pairs")

    col0 = min(idx_values)
    if col0 % 8 != 0:
        raise RuntimeError(
            f"col0 ({col0}) is not an integer multiple of 8")

    n_pairs = len(idx_values)
    dense_mask = np.zeros((n_pairs, n_genes))
    return dense_mask, idx_values, boring_t, idx_dtype


def _save_sub_mask(
        dense_mask,
        idx_values,
        idx_dtype,
        dst_path):
    """
    Save P-value mask as a sparse matrix in an HDF5 file.
    (Meant for saving a subset of the final mask in a temporary
    file).

    Parameters
    ----------
    dense_mask:
        The mask as a dense matrix
    idx_values:
        An array of ints. The indexes in the final mask of the cell type
        pairs stored in this HDF5 file.
    idx_dtype:
        The dtype for storing the sparse matrix indices.
    dst_path:
        Path to the HDF5 file being created.

    Returns
    -------
    None
        The file at dst_path is created and written to.
    """
    n_genes = dense_mask.shape[1]
    n_pairs = dense_mask.shape[0]
    sparse_mask = scipy_sparse.csr_matrix(dense_mask)

    indices = np.copy(sparse_mask.indices)
    indptr = np.copy(sparse_mask.indptr).astype(np.int64)
    data = np.copy(sparse_mask.data).astype(np.float16)
    del sparse_mask
    indices = indices.astype(idx_dtype)

    # store mask as just the indices, indptr arrays from the
    # sparse mask (since this is a boolean array that can only
    # have values 0, 1
    with h5py.File(dst_path, 'w') as out_file:
        out_file.create_dataset(
            'n_genes', data=n_genes)
        out_file.create_dataset(
            'n_pairs', data=n_pairs)
        out_file.create_dataset(
            'indices', data=indices, dtype=idx_dtype)
        out_file.create_dataset(
            'indptr', data=indptr, dtype=np.int64)
        out_file.create_dataset(
            'data', data=data, dtype=np.float16)
        out_file.create_dataset(
            'min_row', data=idx_values.min())


def _merge_masks(
        src_path_list,
        dst_path):
    """
    Merge the temporary files created to store chunks of p-value masks.

    Parameters
    ----------
    src_path_list:
        List of files to merge
    dst_path:
        Final HDF5 file to create
    """
    compression = 'gzip'
    compression_opts = 4
    data_dtype = np.float16
    n_genes = None
    n_indices = 0
    n_indptr = 0
    idx_to_path = dict()
    for pth in src_path_list:
        with h5py.File(pth, 'r') as src:
            this_n = src['n_genes'][()]
            if n_genes is None:
                n_genes = this_n
            if this_n != n_genes:
                raise RuntimeError(
                    f"disagreement on number of genes {this_n} vs {n_genes}")
            n_indices += src['indices'].shape[0]
            n_indptr += src['indptr'].shape[0]-1
            min_row = src['min_row'][()]
            idx_to_path[min_row] = pth

    idx_values = list(idx_to_path.keys())
    idx_values.sort()

    indices_dtype = choose_int_dtype((0, max(n_indices, n_genes)))
    with h5py.File(dst_path, 'a') as dst:
        dst_indices = dst.create_dataset(
            'indices',
            shape=(n_indices,),
            dtype=indices_dtype,
            chunks=(min(n_indices, 1000000),),
            compression=compression,
            compression_opts=compression_opts)

        dst_data = dst.create_dataset(
            'data',
            shape=(n_indices,),
            dtype=data_dtype,
            chunks=(min(n_indices, 1000000),),
            compression=compression,
            compression_opts=compression_opts)

        dst_indptr = dst.create_dataset(
            'indptr',
            shape=(n_indptr+1,),
            dtype=indices_dtype)

        idx_0 = 0
        indptr_0 = 0

        for min_row in idx_values:
            src_path = idx_to_path[min_row]
            with h5py.File(src_path, 'r') as src:
                indices = src['indices'][()].astype(indices_dtype)
                indptr = src['indptr'][()]
                indptr += idx_0
                n_this = len(indices)
                dst_indices[idx_0:idx_0+n_this] = indices
                del indices
                dst_data[idx_0:
                         idx_0+n_this] = src['data'][()].astype(data_dtype)
                dst_indptr[indptr_0:indptr_0+len(indptr)-1] = indptr[:-1]
                idx_0 += n_this
                indptr_0 += len(indptr)-1
        dst_indptr[-1] = n_indices
