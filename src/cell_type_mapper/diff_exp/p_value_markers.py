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

from cell_type_mapper.diff_exp.score_utils import (
    read_precomputed_stats)

from cell_type_mapper.diff_exp.markers import (
    add_sparse_by_gene_markers_to_file,
    _write_to_tmp_file,
    _prep_chunk,
    _merge_sparse_by_pair_files)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)


def find_markers_for_all_taxonomy_pairs_from_p_mask(
        precomputed_stats_path,
        p_value_mask_path,
        output_path,
        n_processors=4,
        tmp_dir=None,
        max_gb=20,
        n_valid=30,
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

    taxonomy_tree:
        instance of
        cell_type_mapper.taxonomty.taxonomy_tree.TaxonomyTree
        ecoding the taxonomy tree

    output_path:
        Path to the HDF5 file where results will be stored

    n_processors:
        Number of independent worker processes to spin out

    max_gb:
        maximum number of GB to load at once

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
    try:
        _find_markers_for_all_taxonomy_pairs_from_p_mask(
            precomputed_stats_path=precomputed_stats_path,
            p_value_mask_path=p_value_mask_path,
            output_path=output_path,
            n_processors=n_processors,
            tmp_dir=tmp_dir,
            max_gb=max_gb,
            n_valid=n_valid,
            gene_list=gene_list,
            drop_level=drop_level)
    finally:
        _clean_up(tmp_dir)


def _find_markers_for_all_taxonomy_pairs_from_p_mask(
        precomputed_stats_path,
        p_value_mask_path,
        output_path,
        n_processors=4,
        tmp_dir=None,
        max_gb=20,
        n_valid=30,
        gene_list=None,
        drop_level=None):
    full_t0 = time.time()

    taxonomy_tree = TaxonomyTree.from_precomputed_stats(
        precomputed_stats_path)

    if drop_level is not None and drop_level in taxonomy_tree.hierarchy:
        taxonomy_tree = taxonomy_tree.drop_level(drop_level)

    t0 = time.time()
    tmp_thinned_path = create_sparse_by_pair_marker_file_from_p_mask(
        precomputed_stats_path=precomputed_stats_path,
        p_value_mask_path=p_value_mask_path,
        taxonomy_tree=taxonomy_tree,
        n_processors=n_processors,
        tmp_dir=tmp_dir,
        max_gb=0.5*max_gb,
        n_valid=n_valid,
        gene_list=gene_list)
    print(f'===== initial creation took {time.time()-t0:.2e} =====')

    with h5py.File(precomputed_stats_path, 'r') as in_file:
        n_genes = len(json.loads(
            in_file['col_names'][()].decode('utf-8')))

    t0 = time.time()
    add_sparse_by_gene_markers_to_file(
        h5_path=tmp_thinned_path,
        n_genes=n_genes,
        max_gb=max_gb,
        tmp_dir=tmp_dir,
        n_processors=n_processors)
    print(f'===== transposition took {time.time()-t0:.2e} =====')

    t0 = time.time()
    shutil.move(
        src=tmp_thinned_path,
        dst=output_path)

    print(f'======= copying took {time.time()-t0:.2e} =====')
    print(f'======= full thing {time.time()-full_t0:.2e} =======')


def create_sparse_by_pair_marker_file_from_p_mask(
        precomputed_stats_path,
        p_value_mask_path,
        taxonomy_tree,
        n_processors=4,
        tmp_dir=None,
        max_gb=6,
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

    n_processors:
        Number of independent worker processes to spin out

    max_gb:
        Maximum number of GB to load when thinning marker file

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
    else:
        valid_gene_idx = None

    n_genes = len(gene_names)

    with h5py.File(p_value_mask_path, 'r') as src:
        with h5py.File(tmp_output_path, 'w') as dst:
            src_gene_names = json.loads(
                src['gene_names'][()].decode('utf-8'))
            dst.create_dataset(
                'gene_names',
                data=src['gene_names'][()])
            dst.create_dataset(
                'pair_to_idx',
                data=src['pair_to_idx'][()])
            dst.create_dataset(
                'n_pairs',
                data=src['n_pairs'][()])
            pair_to_idx = json.loads(src['pair_to_idx'][()].decode('utf-8'))

    if src_gene_names != gene_names:
        raise RuntimeError(
            "gene names mismatch between p-value file "
            "and precomputed_stats file")

    idx_to_pair = dict()
    for level in pair_to_idx:
        for node_1 in pair_to_idx[level]:
            for node_2 in pair_to_idx[level][node_1]:
                idx = pair_to_idx[level][node_1][node_2]
                idx_to_pair[idx] = (level, node_1, node_2)

    del pair_to_idx
    idx_values = list(idx_to_pair.keys())
    idx_values.sort()

    process_dict = {}
    tmp_path_dict = {}
    n_pairs = len(idx_to_pair)

    # how many pairs to run per proceess
    bytes_per_pair = n_genes*20
    max_bytes = max_gb*1024**3
    n_per = np.round(max_bytes/bytes_per_pair).astype(int)

    if n_per > n_pairs//(2*n_processors):
        n_per = n_pairs//(2*n_processors)

    print(f'running with n_per {n_per}')

    if n_per == 0:
        n_per = 10000

    n_per -= (n_per % 8)
    n_per = max(8, n_per)
    t0 = time.time()
    ct_complete = 0

    for col0 in range(0, n_pairs, n_per):

        (col1,
         this_idx_to_pair,
         this_cluster_stats,
         this_tree_as_leaves,
         tmp_path) = _prep_chunk(
                            col0=col0,
                            n_per=n_per,
                            idx_values=idx_values,
                            idx_to_pair=idx_to_pair,
                            cluster_stats=cluster_stats,
                            tree_as_leaves=tree_as_leaves,
                            tmp_dir=inner_tmp_dir)

        tmp_path_dict[col0] = pathlib.Path(tmp_path)

        p = multiprocessing.Process(
                target=_find_markers_from_p_mask_worker,
                kwargs={
                    'p_value_mask_path': p_value_mask_path,
                    'cluster_stats': this_cluster_stats,
                    'tree_as_leaves': this_tree_as_leaves,
                    'idx_to_pair': this_idx_to_pair,
                    'n_genes': n_genes,
                    'tmp_path': tmp_path,
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

    _merge_sparse_by_pair_files(
        tmp_path_dict=tmp_path_dict,
        n_genes=n_genes,
        n_pairs=n_pairs,
        output_path=tmp_output_path)

    return tmp_output_path


def _find_markers_from_p_mask_worker(
        p_value_mask_path,
        cluster_stats,
        tree_as_leaves,
        idx_to_pair,
        n_genes,
        tmp_path,
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
    tmp_path:
        Path to temporary HDF5 file where results for this worker
        will be stored (this process creates that file)
    valid_gene_idx:
        Optional array of gene indices indicating which genes
        can be considered valid markers
    """
    n_genes = len(cluster_stats[list(cluster_stats.keys())[0]]['mean'])
    idx_dtype = choose_int_dtype((0, n_genes))

    idx_values = list(idx_to_pair.keys())
    idx_values.sort()
    col0 = min(idx_values)
    if col0 % 8 != 0:
        raise RuntimeError(
            f"col0 ({col0}) is not an integer multiple of 8")

    # check that we got a contiguous set of indices
    delta = np.diff(idx_values)
    delta = np.unique(delta)
    if len(delta) != 1 or delta[0] != 1:
        raise RuntimeError(
            "Got non-contiguous set of indices")
    idx_min = idx_values[0]
    idx_max = idx_values[-1]

    up_reg_lookup = dict()
    down_reg_lookup = dict()

    up_mask = np.zeros(n_genes, dtype=bool)

    # load the relevant p-value mask
    with h5py.File(p_value_mask_path, mode='r', swmr=True) as src:
        p_indptr = src['indptr'][()]
        p_i0 = p_indptr[idx_min]
        p_i1 = p_indptr[idx_max+1]
        p_indices = src['indices'][p_i0:p_i1]
        sparse_dist = src['data'][p_i0:p_i1]

    p_indptr = p_indptr[idx_min:idx_max+2]
    p_indptr -= p_indptr[0]

    for ct, idx in enumerate(idx_values):

        if not p_indptr[ct+1] <= len(p_indices):
            raise RuntimeError(
                f"last p_indptr {p_indptr[ct+1]}; "
                f"len(p_indices) {len(p_indices)}; "
                f"{p_i0}:{p_i1}; {idx_min} {idx_max}")
        these_indices = p_indices[p_indptr[ct]:p_indptr[ct+1]]
        these_distances = sparse_dist[
                p_indptr[ct]:p_indptr[ct+1]].astype(float)

        validity_mask = _get_validity_mask(
            n_valid=n_valid,
            n_genes=n_genes,
            gene_indices=these_indices,
            raw_distances=these_distances,
            valid_gene_idx=valid_gene_idx)

        # determine if a gene is up- or down-regulated in this
        # taxon pair
        sibling_pair = idx_to_pair[idx]
        level = sibling_pair[0]
        node_1 = f'{level}/{sibling_pair[1]}'
        node_2 = f'{level}/{sibling_pair[2]}'
        stats_1 = cluster_stats[node_1]
        stats_2 = cluster_stats[node_2]
        up_mask[:] = False
        up_mask[stats_2["mean"] > stats_1["mean"]] = True

        up_reg_lookup[idx] = np.where(
            np.logical_and(validity_mask, up_mask))[0]
        down_reg_lookup[idx] = np.where(
            np.logical_and(validity_mask, np.logical_not(up_mask)))[0]

    _write_to_tmp_file(
        up_reg_lookup=up_reg_lookup,
        down_reg_lookup=down_reg_lookup,
        output_path=tmp_path,
        idx_dtype=idx_dtype)


def _get_validity_mask(
        n_valid,
        n_genes,
        gene_indices,
        raw_distances,
        valid_gene_idx=None):
    """
    Get the validity mask for the reference marker
    genes corresponding to one cluster pair.

    Parameters
    ----------
    n_valid:
        The number of desired valid reference markers
    n_genes:
        The number of genes in the dataset
    gene_indices:
        The indices of genes that passed the p-value test
    raw_distances:
        The penetrance parameter space distances corresponding
        to the genes in gene_indices
    valid_gene_idx:
        Indexes of genes that are acceptable as markers.
        If None, all genes are acceptable as markers.

    Returns
    -------
    A numpy array of booleans indicating which genes
    are valid markers for this cluster pair
    """
    if valid_gene_idx is not None:
        prior_invalid_genes = np.ones(n_genes, dtype=bool)
        prior_invalid_genes[valid_gene_idx] = False

    eps = 1.0e-6
    p_mask = np.zeros(n_genes, dtype=bool)
    penetrance_dist = np.zeros(n_genes, dtype=float)

    p_mask[gene_indices] = True

    penetrance_dist[gene_indices] = raw_distances
    penetrance_dist = np.clip(penetrance_dist, a_min=0.0, a_max=None)

    good_dist = penetrance_dist.max()
    bad_dist = 2.0*(good_dist+1.0)

    penetrance_dist[np.logical_not(p_mask)] = 1.5*bad_dist

    # make sure that genes which are marked as invalid
    # a priori fail the penetrance distance check.
    if valid_gene_idx is not None:
        penetrance_dist[prior_invalid_genes] = 1.5*bad_dist

    invalid = (penetrance_dist >= bad_dist)
    abs_valid = (penetrance_dist < eps)

    validity_mask = np.logical_and(
        p_mask,
        abs_valid)

    if validity_mask.sum() < n_valid:
        sorted_dex = np.argsort(penetrance_dist)
        cutoff = penetrance_dist[sorted_dex[n_valid-1]]
        penetrance_mask = (penetrance_dist <= cutoff)
        penetrance_mask[invalid] = False
        penetrance_mask[abs_valid] = True

        validity_mask = np.logical_and(
            p_mask,
            penetrance_mask)

    return validity_mask
