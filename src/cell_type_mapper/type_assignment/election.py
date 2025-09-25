import h5py
import json
import multiprocessing
import numpy as np
import pathlib
import tempfile
import time

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad)

from cell_type_mapper.utils.utils import (
    print_timing,
    mkstemp_clean
)

from cell_type_mapper.utils.multiprocessing_utils import (
    winnow_process_list)

from cell_type_mapper.type_assignment.utils import (
    reconcile_taxonomy_and_markers,
)

from cell_type_mapper.type_assignment.matching import (
   get_leaf_means,
)

from cell_type_mapper.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)

from cell_type_mapper.anndata_iterator.anndata_iterator import (
    AnnDataRowIterator)

import cell_type_mapper.type_assignment.hierarchical_mapping as hier
import cell_type_mapper.hann_mapping.hann_mapping as hann


def run_type_assignment_on_h5ad_cpu(
        query_h5ad_path,
        precomputed_stats_path,
        marker_gene_cache_path,
        taxonomy_tree,
        n_processors,
        chunk_size,
        bootstrap_factor_lookup,
        bootstrap_iteration,
        rng,
        n_assignments=26,
        normalization='log2CPM',
        tmp_dir=None,
        log=None,
        max_gb=10,
        output_taxonomy_tree=None,
        results_output_path=None,
        algorithm="hierarchical"):
    """
    Assign types at all levels of the taxonomy to the query cells
    in an h5ad file.

    Parameters
    ----------
    query_h5ad_path:
        Path to the h5ad file containing the query gene data.

    precomputed_stats_path:
        Path to the HDF5 file where precomputed stats on the
        clusters in our taxonomy are stored.

    marker_gene_cache_path:
        Path to the HDF5 file where lists of marker genes for
        discriminating betwen clustes in our taxonomy are stored.

        Note: This file takes into account the genes available
        in the query data. So: it is specific to this combination
        of taxonomy/reference set and query data set.

    taxonomy_tree:
        instance of
        cell_type_mapper.taxonomty.taxonomy_tree.TaxonomyTree
        encoding the taxonomy tree

    n_processors:
        Number of independent worker processes to spin up

    chunk_size:
        Number of rows (cells) to process at a time.
        Note: if this is larger than n_rows/n_processors,
        then this will get changed to n_rows/n_processors

    bootstrap_factor_lookup:
        A dict mapping the levels in taxonomy_tree.hierarchy to
        fractions (<=1.0) by which to sampel the marker gene set
        at each bootstrapping iteration

    bootstrap_iteration:
        How many booststrap iterations to run when assigning
        cells to cell types

    rng:
        A random number generator

    n_assignments:
        The number of vote getters to track data for.
        Ultimate concequence of this is that n_assignments-1
        "runners up" get reported at each taxonomic level.

    normalization:
        The normalization of the cell by gene matrix in
        the input file; either 'raw' or 'log2CPM'

    tmp_dir:
       Optional directory where query data will be rewritten
       for faster row iteration (if query data is in the form
       of a CSC matrix)

    log:
        Optional CommandLog for tracking warnings emitted by CLI

    max_gb:
        Approximate maximum number of gigabytes of memory to use
        when converting a CSC matrix to CSR (if necessary)

    output_taxonomy_tree:
        optional taxonomy tree reflecting the taxonomy
        to which the data is to be shaped on output
        (this might be different from taxonomy_tree because,
        for instance, the mapper is being run with
        flatten=True or non-NULL drop_level)

    results_output_path:
        Output path for run assignment. If given will save individual chunks of
        the run assignment process to separate files.

    algorithm:
        either "hierarchical" or "hann".
        Indicates which mapping algorithm to run

    Returns
    -------
    A list of paths to the temporary files that represent the chunks
    processed by the mapping worker function.

    Notes
    -----
    In the case of hierarchical mapping, each file is a JSON file containing
    a list of dicts. Each dict correponds to a cell in full_query_gene_data.
    The dict maps level in the hierarchy to the type (at that level)
    the cell has been assigned.

    Dict will look like
        {'cell_id': id_of_cell,
         taxonomy_level1 : {'assignment': chosen_node,
                            'bootstrapping_probability': fraction_of_votes},
         taxonomy_level2 : {'assignment': chosen_node,
                            'bootstrapping_probability': fraction_of_votes},
         ...}
    """

    valid_algorithms = ("hierarchical", "hann")
    if algorithm not in valid_algorithms:
        raise ValueError(
            f"'{algorithm}' is not a valid algorithm; "
            f"only {valid_algorithms} are valid"
        )

    if algorithm == "hierarchical":
        subset_suffix = ".json"
    elif algorithm == "hann":
        subset_suffix = ".h5"

    if results_output_path is not None:
        buffer_dir = pathlib.Path(
                tempfile.mkdtemp(
                    dir=results_output_path,
                    prefix='results_buffer_'))
    else:
        buffer_dir = pathlib.Path(
            tempfile.mkdtemp(
                dir=tmp_dir,
                prefix='tmp_results_buffer_'
            )
        )

    if log is not None:
        log.info("Running CPU implementation of type assignment.")

    (chunk_iterator,
     leaf_node_matrix,
     query_cell_names,
     all_query_identifiers,
     all_query_markers) = preprocess_taxonomy_for_mapping(
                             taxonomy_tree=taxonomy_tree,
                             marker_gene_cache_path=marker_gene_cache_path,
                             query_h5ad_path=query_h5ad_path,
                             precomputed_stats_path=precomputed_stats_path,
                             chunk_size=chunk_size,
                             n_processors=n_processors,
                             tmp_dir=tmp_dir,
                             max_gb=max_gb,
                             log=log)

    process_list = []
    tot_rows = chunk_iterator.n_rows
    row_ct = 0
    t0 = time.time()

    tmp_path_list = []

    for chunk in chunk_iterator:
        r0 = chunk[1]
        r1 = chunk[2]
        name_chunk = query_cell_names[r0:r1]
        tmp_path = mkstemp_clean(
            dir=buffer_dir,
            prefix=f'results_{r0}_{r1}_',
            suffix=subset_suffix
        )
        tmp_path_list.append(tmp_path)

        data = chunk[0]

        data = CellByGeneMatrix(
            data=data,
            gene_identifiers=all_query_identifiers,
            cell_identifiers=name_chunk,
            normalization=normalization,
            log=log)

        if data.normalization != 'log2CPM':
            data.to_log2CPM_in_place()

        # downsample to just include marker genes
        # to limit memory footprint
        data.downsample_genes_in_place(all_query_markers)

        p = multiprocessing.Process(
                target=_run_type_assignment_on_h5ad_worker,
                kwargs={
                    'query_cell_chunk': data,
                    'leaf_node_matrix': leaf_node_matrix,
                    'marker_gene_cache_path': marker_gene_cache_path,
                    'taxonomy_tree': taxonomy_tree,
                    'bootstrap_factor_lookup': bootstrap_factor_lookup,
                    'bootstrap_iteration': bootstrap_iteration,
                    'rng': np.random.default_rng(rng.integers(99, 2**32)),
                    'n_assignments': n_assignments,
                    'results_output_path': tmp_path,
                    'output_taxonomy_tree': output_taxonomy_tree,
                    'algorithm': algorithm})
        p.start()
        process_list.append(p)
        while len(process_list) >= n_processors:
            n0 = len(process_list)
            process_list = winnow_process_list(process_list)
            n1 = len(process_list)
            if n1 < n0:
                row_ct += (n0-n1)*chunk_size
                if row_ct >= n_processors*chunk_size:
                    print_timing(
                        t0=t0,
                        i_chunk=row_ct,
                        tot_chunks=tot_rows,
                        unit=None,
                        chunk_unit="cells")

    while len(process_list) > 0:
        process_list = winnow_process_list(process_list)

    return tmp_path_list


def preprocess_taxonomy_for_mapping(
        taxonomy_tree,
        marker_gene_cache_path,
        query_h5ad_path,
        precomputed_stats_path,
        chunk_size,
        n_processors,
        tmp_dir,
        max_gb,
        log=None):
    """
    Perform boilerplate preprocessing on taxonomy before
    actual mapping happens.

    Parameters
    ----------
    taxonomy_tree:
        a TaxonomyTree
    marker_gene_cache_path:
        path to the marker gene cache file
    query_h5ad_path:
        path to the query data
    precomputed_stats_path:
        path to the precomputed_stats file
    chunk_size:
        number of cells to process at a time
    n_processors:
        number of parallel worker processes to
        spin up
    tmp_dir:
        directory where temporary files will be written
    max_gb:
        maximum GB of memory to use at once
        (only used if we have to transpose the query data
        to CSR)
    log:
        optional CommandLog

    Returns
    -------
    chunk_iterator:
        an AnnDataRowIterator for iterating over chunks
        of query data
    leaf_node_matrix:
        a CellByGeneMatrix of the mean cluster profiles
    query_cell_names:
        list of cell identifiers in query set
    all_query_identifiers:
        list of all query genes
    all_query_markers:
        list of all marker genes in query data set
    """

    (taxonomy_validity,
     taxonomy_msg) = reconcile_taxonomy_and_markers(
         taxonomy_tree=taxonomy_tree,
         marker_cache_path=marker_gene_cache_path)

    if not taxonomy_validity:
        full_msg = "taxonomy_tree and marker_cache "
        full_msg += "appear to describe different taxonomies\n"
        full_msg += taxonomy_msg
        if log is not None:
            log.error(full_msg)
        else:
            raise RuntimeError(full_msg)

    obs = read_df_from_h5ad(query_h5ad_path, 'obs')
    query_cell_names = list(obs.index.values)
    n_rows = len(obs)
    max_chunk_size = max(1, np.ceil(n_rows/n_processors).astype(int))
    chunk_size = min(max_chunk_size, chunk_size)

    with h5py.File(marker_gene_cache_path, 'r', swmr=True) as in_file:
        all_query_identifiers = json.loads(
            in_file["query_gene_names"][()].decode("utf-8"))
        all_query_markers = [
            all_query_identifiers[ii]
            for ii in in_file["all_query_markers"][()]]

    chunk_iterator = AnnDataRowIterator(
        h5ad_path=query_h5ad_path,
        row_chunk_size=chunk_size,
        tmp_dir=tmp_dir,
        log=log,
        max_gb=max_gb,
        n_processors=max(4, n_processors//2))

    # get a CellByGeneMatrix of average expression
    # profiles for each leaf in the taxonomy
    leaf_node_matrix = get_leaf_means(
        taxonomy_tree=taxonomy_tree,
        precompute_path=precomputed_stats_path,
        for_marker_selection=False)

    return (
        chunk_iterator,
        leaf_node_matrix,
        query_cell_names,
        all_query_identifiers,
        all_query_markers
    )


def _run_type_assignment_on_h5ad_worker(
        query_cell_chunk,
        leaf_node_matrix,
        marker_gene_cache_path,
        taxonomy_tree,
        bootstrap_factor_lookup,
        bootstrap_iteration,
        rng,
        n_assignments,
        results_output_path,
        output_taxonomy_tree=None,
        algorithm="hierarchical"):

    if algorithm == "hierarchical":
        assignment = hier.run_hierarchical_type_assignment(
            full_query_gene_data=query_cell_chunk,
            leaf_node_matrix=leaf_node_matrix,
            marker_gene_cache_path=marker_gene_cache_path,
            taxonomy_tree=taxonomy_tree,
            bootstrap_factor_lookup=bootstrap_factor_lookup,
            bootstrap_iteration=bootstrap_iteration,
            rng=rng,
            n_assignments=n_assignments,
            output_taxonomy_tree=output_taxonomy_tree)

        for idx in range(len(assignment)):
            assignment[idx]['cell_id'] = query_cell_chunk.cell_identifiers[idx]

        hier.save_results(assignment, results_output_path)
        return None

    elif algorithm == "hann":
        results = hann.hann_tally_votes(
            full_query_data=query_cell_chunk,
            leaf_node_matrix=leaf_node_matrix,
            marker_gene_cache_path=marker_gene_cache_path,
            taxonomy_tree=taxonomy_tree,
            bootstrap_factor_lookup=bootstrap_factor_lookup,
            bootstrap_iteration=bootstrap_iteration,
            rng=rng
        )
        hann.save_results(results, results_output_path)
        return None

    raise ValueError(
        "worker does not know what to do for algorithm: "
        f"'{algorithm}'"
    )
