import json
import h5py
import multiprocessing
import numpy as np
import pathlib
import scipy.sparse as scipy_sparse
import tempfile
import time

from cell_type_mapper.utils.utils import (
    print_timing,
    mkstemp_clean,
    choose_int_dtype)

from cell_type_mapper.utils.multiprocessing_utils import (
    winnow_process_dict)

from cell_type_mapper.utils.stats_utils import (
    boring_t_from_p_value)

from cell_type_mapper.diff_exp.scores import (
    read_precomputed_stats,
    _get_this_cluster_stats,
    diffexp_p_values_from_stats)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)


def create_p_value_mask_file(
        precomputed_stats_path,
        dst_path,
        p_th=0.01,
        n_processors=4,
        tmp_dir=None,
        max_bytes=6*1024**3,
        gene_list=None,
        drop_level=None):
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

    p_th
        Threshold on p-value for determining if gene is a marker

    n_processors:
        Number of independent worker processes to spin out

    max_bytes:
        Maximum number of bytes to load when thinning marker file

    gene_list:
        Optional list limiting the genes that can be considered
        as markers.

    drop_level:
        Optional level to drop from taxonomy tree

    Returns
    --------
    Path to a file in tmp_dir where the data is stored

    Notes
    -----
    This method stores the markers as sparse arrays with taxonomic
    pairs as the indptr axis.
    """

    taxonomy_tree = TaxonomyTree.from_precomputed_stats(
        precomputed_stats_path)

    if drop_level is not None and drop_level in taxonomy_tree.hierarchy:
        taxonomy_tree = taxonomy_tree.drop_level(drop_level)

    inner_tmp_dir = tempfile.mkdtemp(dir=tmp_dir)

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
    else:
        valid_gene_idx = None

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
    n_per = min(100000, n_pairs//(2*n_processors))
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

    # do the joining spock
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
    p_th:
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
    dense_mask = np.zeros((n_pairs, n_genes), dtype=np.uint8)

    for pair_ct, idx in enumerate(idx_values):
        sibling_pair = idx_to_pair[idx]
        level = sibling_pair[0]
        node1 = sibling_pair[1]
        node2 = sibling_pair[2]

        p_values = diffexp_p_values_from_stats(
            node_1=f'{level}/{node1}',
            node_2=f'{level}/{node2}',
            precomputed_stats=cluster_stats,
            p_th=p_th,
            boring_t=boring_t,
            big_nu=None)

        dense_mask[pair_ct, :] = (p_values < p_th)

    sparse_mask = scipy_sparse.csr_matrix(dense_mask)

    indices = np.copy(sparse_mask.indices)
    indptr = np.copy(sparse_mask.indptr).astype(np.int64)
    del sparse_mask
    indices = indices.astype(idx_dtype)

    # store mask as just the indices, indptr arrays from the
    # sparse mask (since this is a boolean array that can only
    # have values 0, 1
    with h5py.File(tmp_path, 'a') as out_file:
        out_file.create_dataset(
            'n_genes', data=n_genes)
        out_file.create_dataset(
            'n_pairs', data=n_pairs)
        out_file.create_dataset(
            'indices', data=indices, dtype=idx_dtype)
        out_file.create_dataset(
            'indptr', data=indptr, dtype=np.int64)
        out_file.create_dataset(
            'min_row', data=idx_values.min())


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


def _merge_masks(
        src_path_list,
        dst_path):
    """
    Merge the temporary files created to store chunks of p-value masks
    """
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

    indices_dtype = choose_int_dtype((0, n_genes))
    with h5py.File(dst_path, 'a') as dst:
        dst_indices = dst.create_dataset(
            'indices',
            shape=(n_indices,),
            dtype=indices_dtype,
            chunks=(min(n_indices, 1000),))
        dst_indptr = dst.create_dataset(
            'indptr',
            shape=(n_indptr+1,),
            dtype=np.int64)

        idx_0 = 0
        indptr_0 = 0

        for min_row in idx_values:
            src_path = idx_to_path[min_row]
            with h5py.File(src_path, 'r') as src:
                indices = src['indices'][()].astype(indices_dtype)
                indptr = src['indptr'][()]
                assert indptr.min() >= 0
                indptr += idx_0
                dst_indices[idx_0:idx_0+len(indices)] = indices
                dst_indptr[indptr_0:indptr_0+len(indptr)-1] = indptr[:-1]
                idx_0 += len(indices)
                indptr_0 += len(indptr)-1
                assert idx_0 > 0
                assert indptr_0 > 0
        dst_indptr[-1] = n_indices
