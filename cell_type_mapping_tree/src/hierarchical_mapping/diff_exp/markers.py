import json
import h5py
import multiprocessing
import pathlib
import tempfile
import time

from hierarchical_mapping.utils.utils import (
    print_timing,
    _clean_up,
    mkstemp_clean)

from hierarchical_mapping.utils.multiprocessing_utils import (
    winnow_process_dict)

from hierarchical_mapping.utils.taxonomy_utils import (
    convert_tree_to_leaves,
    get_all_pairs)

from hierarchical_mapping.diff_exp.scores import (
    read_precomputed_stats,
    _get_this_cluster_stats,
    score_differential_genes)

from hierarchical_mapping.binary_array.binary_array import (
    BinarizedBooleanArray)

from hierarchical_mapping.binary_array.backed_binary_array import (
    BackedBinarizedBooleanArray)


def find_markers_for_all_taxonomy_pairs(
        precomputed_stats_path,
        taxonomy_tree,
        output_path,
        p_th=0.01,
        q1_th=0.5,
        qdiff_th=0.7,
        flush_every=1000,
        n_processors=4,
        tmp_dir=None):
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

    tmp_dir = tempfile.mkdtemp(dir=tmp_dir)
    tmp_dir = pathlib.Path(tmp_dir)

    tree_as_leaves = convert_tree_to_leaves(taxonomy_tree)

    precomputed_stats = read_precomputed_stats(
           precomputed_stats_path)
    cluster_stats = precomputed_stats['cluster_stats']
    gene_names = precomputed_stats['gene_names']
    del precomputed_stats

    n_genes = len(gene_names)

    idx_to_pair = _prep_output_file(
            output_path=output_path,
            taxonomy_tree=taxonomy_tree,
            gene_names=gene_names)

    print("starting to score")
    t0 = time.time()

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
            dir=tmp_dir,
            prefix=f'columns_{col0}_{col1}_',
            suffix='.h5')
        tmp_path_dict[col0] = tmp_path

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
                    'tmp_path': tmp_path})
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
                    tot_chunks=len(idx_to_pair),
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
                tot_chunks=len(idx_to_pair),
                unit='hr')

    marker_flag = BackedBinarizedBooleanArray(
        h5_path=output_path,
        h5_group='markers',
        n_rows=n_genes,
        n_cols=n_pairs)

    up_regulated_flag = BackedBinarizedBooleanArray(
        h5_path=output_path,
        h5_group='up_regulated',
        n_rows=n_genes,
        n_cols=n_pairs)

    print("copying from temp files")
    for col0 in tmp_path_dict:
        markers = BinarizedBooleanArray.read_from_h5(
            h5_path=tmp_path_dict[col0],
            h5_group='markers')
        up_reg = BinarizedBooleanArray.read_from_h5(
            h5_path=tmp_path_dict[col0],
            h5_group='up_regulated')

        marker_flag.copy_other_as_columns(
            other=markers, col0=col0)
        up_regulated_flag.copy_other_as_columns(
            other=up_reg, col0=col0)

    _clean_up(tmp_dir)
    duration = time.time()-t0
    print(f"that took {duration/3600.0:.2e} hrs")


def _find_markers_worker(
        cluster_stats,
        tree_as_leaves,
        idx_to_pair,
        n_genes,
        p_th,
        q1_th,
        qdiff_th,
        tmp_path):
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
    """

    idx_values = list(idx_to_pair.keys())
    idx_values.sort()
    col0 = min(idx_values)
    if col0 % 8 != 0:
        raise RuntimeError(
            f"col0 ({col0}) is not an integer multiple of 8")
    n_sibling_pairs = len(idx_to_pair)

    marker_mask = BinarizedBooleanArray(
        n_rows=n_genes,
        n_cols=n_sibling_pairs)

    up_regulated_mask = BinarizedBooleanArray(
        n_rows=n_genes,
        n_cols=n_sibling_pairs)

    for idx in idx_values:
        sibling_pair = idx_to_pair[idx]
        level = sibling_pair[0]
        node1 = sibling_pair[1]
        node2 = sibling_pair[2]

        pop1 = tree_as_leaves[level][node1]
        pop2 = tree_as_leaves[level][node2]
        (scores,
         validity_mask,
         up_mask) = score_differential_genes(
                         leaf_population_1=pop1,
                         leaf_population_2=pop2,
                         precomputed_stats=cluster_stats,
                         p_th=p_th,
                         q1_th=q1_th,
                         qdiff_th=qdiff_th)

        this_col = idx-col0
        marker_mask.set_col(
            this_col,
            validity_mask)
        up_regulated_mask.set_col(
            this_col,
            up_mask.astype(bool))

    marker_mask.write_to_h5(
        h5_path=tmp_path,
        h5_group='markers')
    up_regulated_mask.write_to_h5(
        h5_path=tmp_path,
        h5_group='up_regulated')
    with h5py.File(tmp_path, 'a') as out_file:
        out_file.create_dataset(
            'col0',
            data=col0)


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
        Dict encoding the taxonomy tree (created when we create the
        contiguous zarr file and stored in that file's metadata.json)
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
    siblings = get_all_pairs(taxonomy_tree)
    n_sibling_pairs = len(siblings)
    print(f"{n_sibling_pairs:.2e} sibling pairs")

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
