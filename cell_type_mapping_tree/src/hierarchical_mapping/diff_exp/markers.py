import json
import h5py
import multiprocessing
import numpy as np
import os
import pathlib
import tempfile
import time

from hierarchical_mapping.utils.utils import (
    print_timing,
    _clean_up)

from hierarchical_mapping.utils.multiprocessing_utils import (
    DummyLock)

from hierarchical_mapping.utils.taxonomy_utils import (
    convert_tree_to_leaves,
    get_all_pairs)

from hierarchical_mapping.utils.stats_utils import (
    welch_t_test,
    correct_ttest)

from hierarchical_mapping.diff_exp.scores import (
    read_precomputed_stats,
    _prep_output_file,
    _get_this_cluster_stats,
    score_differential_genes,
    rank_genes)


def find_markers_for_all_taxonomy_pairs(
        precomputed_stats_path,
        taxonomy_tree,
        output_path,
        p_th=0.01,
        q1_th=0.5,
        qdiff_th=0.7,
        flush_every=1000,
        n_processors=4,
        tmp_dir=None,
        keep_all_stats=True,
        genes_to_keep=None):
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
        Dict encoding the taxonomy tree (created when we create the
        contiguous zarr file and stored in that file's metadata.json)

    output_path:
        Path to the HDF5 file where results will be stored

    p_th/q1_th/qdiff_th
        Thresholds for determining if a gene is a valid marker.
        See Notes under score_differential_genes

    flush_every:
        Write to HDF5 every flush_every pairs

    n_processors:
        Number of independent worker processes to spin out

    keep_all_stats:
        If True, keep scores and validity as well as ranked_list
        for each cell type pair. If False,  only keep ranked_list.
        (True results in a much larger file than False)

    genes_to_keep:
        If None, store data for all genes. If an integer, keep only the
        top-ranked genes_to_keep genes.

        Since validity, scores and ranked_list are not in the same order,
        can only call non-none genes_to_keep if keep_all_stats is False.

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

    if genes_to_keep is not None and keep_all_stats:
        raise RuntimeError(
            "Cannot have non-None genes_to_keep if "
            "keep_all_stats is True; "
            f"you have genes_to_keep={genes_to_keep}")

    tmp_dir = tempfile.mkdtemp(dir=tmp_dir)
    tmp_dir = pathlib.Path(tmp_dir)

    tree_as_leaves = convert_tree_to_leaves(taxonomy_tree)

    precomputed_stats = read_precomputed_stats(
           precomputed_stats_path)
    cluster_stats = precomputed_stats['cluster_stats']
    gene_names = precomputed_stats['gene_names']
    del precomputed_stats

    n_genes = len(gene_names)
    if n_genes < 2**8:
        rank_dtype = np.uint8
    elif n_genes < 2**16:
        rank_dtype = np.uint16
    elif n_genes < 2**32:
        rank_dtype = np.uint32
    else:
        rank_dtype = np.uint

    n_genes_to_keep = n_genes
    if genes_to_keep is not None:
        n_genes_to_keep = genes_to_keep

    idx_to_pair = _prep_output_file(
            output_path=output_path,
            taxonomy_tree=taxonomy_tree,
            n_genes=n_genes,
            n_genes_to_keep=n_genes_to_keep,
            keep_all_stats=keep_all_stats,
            gene_names=gene_names,
            rank_dtype=rank_dtype)

    print("starting to score")
    t0 = time.time()

    idx_values = list(idx_to_pair.keys())
    idx_values.sort()

    process_list = []
    tmp_path_list = []
    mgr = multiprocessing.Manager()
    output_lock = mgr.Lock()
    n_per = len(idx_values)//n_processors
    for i_process in range(n_processors):
        i0 = i_process*n_per
        i1 = i0+n_per
        if i_process == n_processors-1:
            i1 = len(idx_values)

        tmp_path = tempfile.mkstemp(
                        dir=tmp_dir,
                        suffix='.h5')
        os.close(tmp_path[0])
        tmp_path = pathlib.Path(tmp_path[1])

        this_idx_values = idx_values[i0:i1]
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
                    'idx_values': this_idx_values,
                    'n_genes': n_genes,
                    'p_th': p_th,
                    'q1_th': q1_th,
                    'qdiff_th': qdiff_th,
                    'flush_every': flush_every,
                    'tmp_path': tmp_path,
                    'keep_all_stats': keep_all_stats,
                    'rank_dtype': rank_dtype,
                    'genes_to_keep': genes_to_keep,
                    'output_path': output_path,
                    'output_lock': output_lock})
        p.start()
        process_list.append(p)
        tmp_path_list.append(tmp_path)

    del cluster_stats
    del tree_as_leaves
    del this_cluster_stats
    del this_idx_values
    del this_idx_to_pair
    del this_tree_as_leaves

    for p in process_list:
        p.join()

    _clean_up(tmp_dir)
    duration = time.time()-t0
    print(f"that took {duration/3600.0:.2e} hrs")


def _find_markers_worker(
        cluster_stats,
        tree_as_leaves,
        idx_to_pair,
        idx_values,
        n_genes,
        p_th,
        q1_th,
        qdiff_th,
        flush_every,
        tmp_path,
        keep_all_stats,
        rank_dtype,
        genes_to_keep,
        output_path,
        output_lock=None):
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
        Dict mapping row in output file to
        (level, node1, node2) sibling pair
    idx_values:
        Row indexes to be processed by this worker
    n_genes:
        Number of genes in dataset
    p_th/q1_th/qdiff_th
        Thresholds for determining if a gene is a valid marker.
        See Notes under score_differential_genes
    flush_every:
        Write to temporary output file every flush_every rows
    tmp_path:
        Path to temporary HDF5 file where results for this worker
        will be stored (this process creates and destroys that file)
    keep_all_stats:
        If True, keep scores and validity as well as ranked_list
        for each cell type pair. If False,  only keep ranked_list.
        (True results in a much larger file than False)
    rank_dtype:
        The dtype to use when storing ranked lists of genes
    genes_to_keep:
        If None, store data for all genes. If an integer, keep only the
        top-ranked genes_to_keep genes.

        Since validity, scores and ranked_list are not in the same order,
        can only call non-none genes_to_keep if keep_all_stats is False.
    output_path:
        Final output file
    output_lock:
        Multiprocessing lock to prevent multiple workers from writing
        to file at output_path simultaneously
    """
    if genes_to_keep is not None and keep_all_stats:
        raise RuntimeError(
            "Cannot have non-None genes_to_keep if "
            "keep_all_stats is True; "
            f"you have genes_to_keep={genes_to_keep}")

    n_genes_to_keep = n_genes
    if genes_to_keep is not None:
        n_genes_to_keep = genes_to_keep

    n_sibling_pairs = len(idx_values)

    chunk_size = (max(1, min(1000000//n_genes_to_keep, n_sibling_pairs)),
                  n_genes_to_keep)

    this_bounds = (idx_values[0], idx_values[-1]+1)

    with h5py.File(tmp_path, 'w') as out_file:

        out_file.create_dataset(
            'ranked_list',
            shape=(n_sibling_pairs, n_genes_to_keep),
            dtype=rank_dtype,
            chunks=chunk_size,
            compression='gzip')

        if keep_all_stats:
            out_file.create_dataset(
                'validity',
                shape=(n_sibling_pairs, n_genes),
                dtype=bool,
                chunks=chunk_size,
                compression='gzip')

            out_file.create_dataset(
                'scores',
                shape=(n_sibling_pairs, n_genes),
                dtype=float,
                chunks=chunk_size)

    t0 = time.time()
    if output_lock is None:
        output_lock = DummyLock()

    ranked_list_buffer = np.zeros((flush_every, n_genes_to_keep),
                                  dtype=rank_dtype)

    if keep_all_stats:
        validity_buffer = np.zeros((flush_every, n_genes), dtype=bool)
        score_buffer = np.zeros((flush_every, n_genes), dtype=float)

    buffer_idx = 0
    ct = 0
    for idx in idx_values:
        sibling_pair = idx_to_pair[idx]
        level = sibling_pair[0]
        node1 = sibling_pair[1]
        node2 = sibling_pair[2]

        pop1 = tree_as_leaves[level][node1]
        pop2 = tree_as_leaves[level][node2]
        (scores,
         validity,
         _) = score_differential_genes(
                         leaf_population_1=pop1,
                         leaf_population_2=pop2,
                         precomputed_stats=cluster_stats,
                         p_th=p_th,
                         q1_th=q1_th,
                         qdiff_th=qdiff_th)

        ranked_list = rank_genes(
                         scores=scores,
                         validity=validity)

        if genes_to_keep is not None:
            ranked_list = ranked_list[:n_genes_to_keep]

        ranked_list_buffer[buffer_idx, :] = ranked_list.astype(rank_dtype)

        if keep_all_stats:
            score_buffer[buffer_idx, :] = scores
            validity_buffer[buffer_idx, :] = validity

        buffer_idx += 1
        ct += 1
        if buffer_idx == flush_every or idx == idx_values[-1]:
            with h5py.File(tmp_path, 'a') as out_file:
                out_idx1 = idx+1-min(idx_values)
                out_idx0 = out_idx1-buffer_idx
                out_file['ranked_list'][out_idx0:out_idx1,
                                        :] = ranked_list_buffer[:buffer_idx, :]

                if keep_all_stats:
                    out_file['scores'][out_idx0:out_idx1,
                                       :] = score_buffer[:buffer_idx, :]

                    out_file['validity'][out_idx0:out_idx1,
                                         :] = validity_buffer[:buffer_idx, :]

                buffer_idx = 0

            print_timing(
                t0=t0,
                i_chunk=ct+1,
                tot_chunks=len(idx_values),
                unit='hr')

    # write this output from the temporary file to
    # the final output file
    if keep_all_stats:
        key_list = ('ranked_list', 'scores', 'validity')
    else:
        key_list = ('ranked_list', )

    with output_lock:
        d_write = max(1, 10000000//n_genes_to_keep)
        with h5py.File(tmp_path, 'r') as in_file:
            with h5py.File(output_path, 'a') as out_file:
                for i0 in range(this_bounds[0], this_bounds[1], d_write):
                    i1 = min(this_bounds[1], i0+d_write)
                    for k in key_list:
                        out_file[k][i0:i1] = in_file[k][i0-this_bounds[0]:
                                                        i1-this_bounds[0]]
    tmp_path.unlink()
