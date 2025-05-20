import os
import h5py
import json
import multiprocessing
import numpy as np
import pathlib
import tempfile
import time

from cell_type_mapper.utils.torch_utils import (
    is_torch_available,
    use_torch)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad)

from cell_type_mapper.utils.utils import (
    print_timing,
    update_timer,
    choose_int_dtype,
    clean_for_json,
    _clean_up)

from cell_type_mapper.utils.multiprocessing_utils import (
    winnow_process_list)

import cell_type_mapper.utils.distance_utils as distance_utils

from cell_type_mapper.type_assignment.utils import (
    reconcile_taxonomy_and_markers,
    infer_election)

from cell_type_mapper.type_assignment.matching import (
   get_leaf_means,
   assemble_query_data)

from cell_type_mapper.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)

from cell_type_mapper.anndata_iterator.anndata_iterator import (
    AnnDataRowIterator)

if is_torch_available():
    import torch


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
        results_output_path=None):
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

    Returns
    -------
    A list of dicts. Each dict correponds to a cell in full_query_gene_data.
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
    if results_output_path is not None:
        buffer_dir = pathlib.Path(
                tempfile.mkdtemp(
                    dir=results_output_path,
                    prefix='results_buffer_'))
    else:
        buffer_dir = None

    if log is not None:
        log.info("Running CPU implementation of type assignment.")

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
    del obs

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

    process_list = []
    if results_output_path:
        output_list, output_lock = [], None
    else:
        mgr = multiprocessing.Manager()
        output_list = mgr.list()
        output_lock = mgr.Lock()

    tot_rows = chunk_iterator.n_rows
    row_ct = 0
    t0 = time.time()

    # get a CellByGeneMatrix of average expression
    # profiles for each leaf in the taxonomy
    leaf_node_matrix = get_leaf_means(
        taxonomy_tree=taxonomy_tree,
        precompute_path=precomputed_stats_path,
        for_marker_selection=False)

    chunk_index = -1
    for chunk in chunk_iterator:
        chunk_index += 1
        r0 = chunk[1]
        r1 = chunk[2]
        name_chunk = query_cell_names[r0:r1]

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
                    'r0': r0,
                    'r1': r1,
                    'query_cell_chunk': data,
                    'query_cell_names': name_chunk,
                    'leaf_node_matrix': leaf_node_matrix,
                    'marker_gene_cache_path': marker_gene_cache_path,
                    'taxonomy_tree': taxonomy_tree,
                    'bootstrap_factor_lookup': bootstrap_factor_lookup,
                    'bootstrap_iteration': bootstrap_iteration,
                    'rng': np.random.default_rng(rng.integers(99, 2**32)),
                    'n_assignments': n_assignments,
                    'output_list': output_list,
                    'output_lock': output_lock,
                    'results_output_path': buffer_dir,
                    'output_taxonomy_tree': output_taxonomy_tree})
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

    if buffer_dir is not None:
        path_list = [n for n in buffer_dir.iterdir()]
        path_list.sort()
        output_list = []
        for path in path_list:
            output_list += json.load(open(path, 'rb'))
        _clean_up(buffer_dir)
    else:
        output_list = list(output_list)

    return output_list


def save_results(result, results_output_path):
    with open(results_output_path, "w") as outfile:
        json.dump(clean_for_json(result), outfile)


def _run_type_assignment_on_h5ad_worker(
        r0,
        r1,
        query_cell_chunk,
        query_cell_names,
        leaf_node_matrix,
        marker_gene_cache_path,
        taxonomy_tree,
        bootstrap_factor_lookup,
        bootstrap_iteration,
        rng,
        n_assignments,
        output_list,
        output_lock,
        results_output_path=None,
        output_taxonomy_tree=None):

    assignment = run_type_assignment(
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
        assignment[idx]['cell_id'] = query_cell_names[idx]

    if results_output_path:
        this_output_path = os.path.join(results_output_path,
                                        f"{r0}_{r1}_assignment.json")
        save_results(assignment, this_output_path)
    else:
        with output_lock:
            output_list += assignment


def run_type_assignment(
        full_query_gene_data,
        leaf_node_matrix,
        marker_gene_cache_path,
        taxonomy_tree,
        bootstrap_factor_lookup,
        bootstrap_iteration,
        rng,
        output_taxonomy_tree=None,
        n_assignments=25,
        gpu_index=0,
        timers=None):
    """
    Assign types at all levels of the taxonomy to a set of
    query cells.

    Parameters
    ----------
    full_query_gene_data:
        A CellByGeneMatrix containing the query data.
        Must have normalization == 'log2CPM'.

    leaf_node_matrix:
        A CellByGeneMatrix containing the average expression
        profiles for the leaf nodes of the taxonomy

    marker_gene_cache_path:
        Path to the HDF5 file where lists of marker genes for
        discriminating betwen clustes in our taxonomy are stored.

        Note: This file takes into account the genes available
        in the query data. So: it is specific to this combination
        of taxonomy/reference set and query data set.

    taxonomy_tree:
        instance of
        cell_type_mapper.taxonomy.taxonomy_tree.TaxonomyTree
        encoding the taxonomy tree

    bootstrap_factor_lookup:
        A dict mapping the levels in taxonomy_tree.hierarchy to
        fractions (<=1.0) by which to sampel the marker gene set
        at each bootstrapping iteration

    bootstrap_iteration:
        How many booststrap iterations to run when assigning
        cells to cell types

    rng:
        A random number generator

    output_taxonomy_tree:
        optional taxonomy tree reflecting the taxonomy
        to which the data is to be shaped on output
        (this might be different from taxonomy_tree because,
        for instance, the mapper is being run with
        flatten=True or non-NULL drop_level)

    n_assignments:
        The number of vote getters to track data for.
        Ultimate concequence of this is that n_assignments-1
        "runners up" get reported at each taxonomic level.

    gpu_index:
        Index of the GPU for this operation. Supports multi-gpu usage

    Returns
    -------
    A list of dicts. Each dict correponds to a cell in full_query_gene_data.
    The dict maps level in the hierarchy to the type (at that level)
    the cell has been assigned. The dict will also include the confidence level
    of the assignment (i.e. what fraction of bootstrap iterations thought the
    cell belongs to that node)

    Dict will look like
        {taxonomy_level : {
            'assignment': chosen_node,
            'bootstrapping_probability': fraction_of_votes,
            'avg_correlation': correlation averaged over iterations
            'runner_up_assignment': [runner, up, nodes],
            'runner_up_correlation': [runner, up, correlation],
            'runner_up_probability': [runner, up, bootstrapping, probability]}}
    """

    # create effectively empty list of dicts to
    # store the hierarchical classification of
    # each cell in full_query_gene_data
    hierarchy = taxonomy_tree.hierarchy

    result = [{level: None for level in hierarchy}
              for _ in range(full_query_gene_data.n_cells)]

    # list of levels in the taxonomy (None means consider all clusters)
    level_list = [None] + list(hierarchy)

    # dict to keep track of which rows were assigned to which
    # child types in a rapidly-accessible way
    previously_assigned = dict()

    # loop over parent_level, assigning query cells to the child types
    # of that level
    for parent_level, child_level in zip(level_list[:-1], level_list[1:]):

        # build list of parent nodes to search from
        if parent_level is None:
            parent_node_list = [None]
        else:
            k_list = taxonomy_tree.nodes_at_level(parent_level)
            k_list.sort()
            parent_node_list = [(parent_level, k) for k in k_list]

        previously_assigned[child_level] = dict()

        for parent_node in parent_node_list:

            # only consider the query cells that were assigned
            # to the current parent
            if parent_node is None:
                chosen_idx = np.arange(full_query_gene_data.n_cells)
                chosen_query_data = full_query_gene_data
            else:
                if parent_node[1] in previously_assigned[parent_level]:
                    chosen_idx = previously_assigned[
                        parent_level][parent_node[1]]

                    chosen_query_data = \
                        full_query_gene_data.downsample_cells_by_idx(
                            selected_cell_idx=chosen_idx)

                else:
                    chosen_idx = []

            if len(chosen_idx) == 0:
                continue

            # see how many children this parent node has;
            # if == 1, assignment is trivial
            if parent_level is not None:
                possible_children = taxonomy_tree.children(
                    level=parent_level,
                    node=parent_node[1])
            else:
                possible_children = taxonomy_tree.children(None, None)

            if len(possible_children) > 1:
                t = time.time()

                bootstrap_factor = bootstrap_factor_lookup[str(parent_level)]

                # spock
                # return raw votes here
                # reshape in this function
                # then, we can just do any level inferring at this point
                (votes,
                 corr_sum,
                 reference_types) = _run_type_assignment(
                                full_query_gene_data=chosen_query_data,
                                leaf_node_matrix=leaf_node_matrix,
                                marker_gene_cache_path=marker_gene_cache_path,
                                taxonomy_tree=taxonomy_tree,
                                parent_node=parent_node,
                                bootstrap_factor=bootstrap_factor,
                                bootstrap_iteration=bootstrap_iteration,
                                rng=rng,
                                gpu_index=gpu_index,
                                timers=timers)

                update_timer("run_type_assignment", t, timers)

            elif len(possible_children) == 1:
                votes = None
                corr_sum = None
                reference_types = possible_children

            else:
                raise RuntimeError(
                    "Not sure how to proceed;\n"
                    f"parent {parent_node}\n"
                    f"has children {possible_children}")

            result = update_mapping_result(
                result=result,
                child_level=child_level,
                cell_idx=chosen_idx,
                votes=votes,
                corr_sum=corr_sum,
                reference_types=reference_types,
                n_assignments=n_assignments,
                taxonomy_tree=taxonomy_tree,
                output_taxonomy_tree=output_taxonomy_tree
            )

            previously_assigned = update_previously_assigned(
                result=result,
                child_level=child_level,
                chosen_idx=chosen_idx,
                previously_assigned=previously_assigned
            )

    return result


def _run_type_assignment(
        full_query_gene_data,
        leaf_node_matrix,
        marker_gene_cache_path,
        taxonomy_tree,
        parent_node,
        bootstrap_factor,
        bootstrap_iteration,
        rng,
        gpu_index=0,
        timers=None):
    """
    Assign a set of query cells to types that are children
    of a specified parent node in our taxonomy.

    Parameters
    ----------
    full_query_gene_data:
        A CellByGeneMatrix containing the query data.
        Must have normalization == 'log2CPM'

    leaf_node_matrix:
        A CellByGeneMatrix containing the mean gene expression
        profiles of each cell type cluster

    marker_gene_cache_path:
        Path to the HDF5 file where lists of marker genes for
        discriminating betwen clusters in our taxonomy are stored.

        Note: This file takes into account the genes available
        in the query data. So: it is specific to this combination
        of taxonomy/reference set and query data set.

    taxonomy_tree:
        instance of
        cell_type_mapper.taxonomty.taxonomy_tree.TaxonomyTree
        encoding the taxonomy tree

    parent_node:
        Tuple of the form (level, cell_type) that encodes
        the parent to whose children we are mapping the cells
        in the query data

    bootstrap_factor:
        Fraction (<=1.0) by which to sample the marker gene set
        at each bootstrapping iteration

    bootstrap_iteration:
        How many booststrap iterations to run when assigning
        cells to cell types

    rng:
        A random number generator

    gpu_index:
        Index of the GPU for this operation. Supports multi-gpu usage

    Returns
    -------
    votes:
        (n_cells, n_types) array of votes each unique cell type got
        for each cell
    corr_sum:
        (n_cells, n_types) array of sums of correlations between
        cells and unique cell types
    reference_types:
        array of unique cell types indicating the meaning of columns
        in votes and corr_sum
    """

    t0 = time.time()

    query_data = assemble_query_data(
        full_query_data=full_query_gene_data,
        mean_profile_matrix=leaf_node_matrix,
        marker_cache_path=marker_gene_cache_path,
        taxonomy_tree=taxonomy_tree,
        parent_node=parent_node)
    update_timer("assemble", t0, timers)

    t0 = time.time()

    (votes,
     corr_sum,
     reference_types) = tally_votes(
        query_gene_data=query_data['query_data'].data,
        reference_gene_data=query_data['reference_data'].data,
        reference_types=query_data['reference_types'],
        bootstrap_factor=bootstrap_factor,
        bootstrap_iteration=bootstrap_iteration,
        rng=rng,
        gpu_index=gpu_index,
        timers=timers)

    update_timer("choose_node_p2", t0, timers)

    return votes, corr_sum, reference_types


def tally_votes(
         query_gene_data,
         reference_gene_data,
         reference_types,
         bootstrap_factor,
         bootstrap_iteration,
         rng,
         gpu_index=0,
         timers=None):
    """
    Parameters
    ----------
    query_gene_data
        A numpy array of cell-by-marker-gene data for the query set
    reference_gene_data
        A numpy array of cell-by-marker-gene data for the reference set
    reference_types
        array of cell types we are chosing from
    bootstrap_factor
        Factor by which to subsample reference genes at each bootstrap
    bootstrap_iteration
        Number of bootstrapping iterations
    rng
        random number generator
    gpu_index:
        Index of the GPU for this operation. Supports multi-gpu usage

    Returns
    -------
    votes:
        (n_cells, n_types) array of votes each unique cell type got
        for each cell
    corr_sum:
        (n_cells, n_types) array of sums of correlations between
        cells and unique cell types
    reference_types:
        array of unique cell types indicating the meaning of columns
        in votes and corr_sum
    """

    t0 = time.time()

    (votes,
     corr_sum) = tally_raw_votes(
        query_gene_data=query_gene_data,
        reference_gene_data=reference_gene_data,
        bootstrap_factor=bootstrap_factor,
        bootstrap_iteration=bootstrap_iteration,
        rng=rng,
        gpu_index=gpu_index,
        timers=timers)

    update_timer("tally_votes", t0, timers)

    if len(set(reference_types)) < len(reference_types):
        (votes,
         corr_sum,
         reference_types) = aggregate_votes(
             vote_array=votes,
             correlation_array=corr_sum,
             reference_types=reference_types)

    return (votes, corr_sum, reference_types)


def reshape_type_assignment(
        votes,
        corr_sum,
        reference_types,
        n_assignments,
        forced_assignments=None):
    """
    Take votes, correlations, and cell types.

    Return output shaped for saving to disk.

    Parameters
    ----------
    votes:
        (n_cells, n_types) array of votes for types in
        reference_types

    corr_sum:
        (n_cells, n_types) array of correlations between cells
        and reference_types

    reference_types:
        array of cell types we are chosing from

    n_assignments:
        The number of vote getters to track data for.
        Ultimate concequence of this is that n_assignments-1
        "runners up" get reported at each taxonomic level.

    forced_assignments:
        An optional array of cell types to which these
        cells *must* be assigned (for instance, to preserve
        inheritance when backfilling levels in the taxonomy
        that were skipped). If not None, these cell types
        will be chosen for the cells, regardless of where
        they fall in the hierarchy of vote fractions.

    Returns
    -------
    Array of cell type assignments (majority rule)

    Array of vote fractions

    Array of the average correlation value of the chosen nearest neighbors

    Array of runner up tuples that look like
        (runner_up_type,
         boolean indicating whether any votes were received or not,
         avg_correlation,
         bootstrappping_probability)
    """
    if forced_assignments is not None:
        label_to_idx = {
            label: idx
            for idx, label in enumerate(reference_types)
        }
        forced_idx = np.array([
            label_to_idx[label]
            for label in forced_assignments
        ])

    bootstrap_iteration = votes[0, :].sum()

    if forced_assignments is not None:
        result = np.array(forced_assignments)
        assignment_vote_fractions = np.array(
            [votes[ii, forced_idx[ii]]
             for ii in range(votes.shape[0])]
        ).astype(float)

        assignment_corr = np.array(
            [corr_sum[ii, forced_idx[ii]]
             for ii in range(votes.shape[0])]
        ) / np.where(assignment_vote_fractions > 0,
                     assignment_vote_fractions,
                     1)

        assignment_vote_fractions /= bootstrap_iteration

    n_assignments = min(n_assignments, votes.shape[1])

    sorted_by_votes = np.argsort(votes, axis=1)[:, -1::-1]
    sorted_by_votes = sorted_by_votes[:, :n_assignments]

    idx_array_2d = np.array([[ii]*sorted_by_votes.shape[1]
                             for ii in range(sorted_by_votes.shape[0])])

    votes = votes[idx_array_2d, sorted_by_votes]
    vote_fractions = votes / bootstrap_iteration

    denom = np.where(votes > 0, votes, 1)

    avg_corr = corr_sum[idx_array_2d, sorted_by_votes] / denom

    if forced_assignments is None:
        result = np.array(
            [reference_types[ii] for ii in sorted_by_votes[:, 0]]
        )
        assignment_vote_fractions = vote_fractions[:, 0]
        assignment_corr = avg_corr[:, 0]

    runners_up = [
        [(reference_types[sorted_by_votes[i_row, i_col]],
          votes[i_row, i_col] > 0,
          avg_corr[i_row, i_col],
          vote_fractions[i_row, i_col])
         for i_col in range(0, n_assignments, 1)
         if reference_types[sorted_by_votes[i_row, i_col]] != result[i_row]]
        for i_row in range(sorted_by_votes.shape[0])
    ]

    # trim runners_up down to specified size (in case forced_assignment
    # causes one of the rows to be too long
    for row in runners_up:
        if len(row) >= n_assignments:
            row.pop(-1)

    return (result,
            assignment_vote_fractions,
            assignment_corr,
            runners_up)


def tally_raw_votes(
         query_gene_data,
         reference_gene_data,
         bootstrap_factor,
         bootstrap_iteration,
         rng,
         gpu_index=0,
         timers=None):
    """
    Parameters
    ----------
    query_gene_data
        A numpy array of cell-by-marker-gene data for the query set
    reference_gene_data
        A numpy array of cell-by-marker-gene data for the reference set
    bootstrap_factor
        Factor by which to subsample reference genes at each bootstrap
    bootstrap_iteration
        Number of bootstrapping iterations
    rng
        random number generator
    gpu_index:
        Index of the GPU for this operation. Supports multi-gpu usage

    Returns
    -------
    votes:
        Array of ints. Each row is a query cell. Each column is a
        reference cell. The value is how many iterations voted for
        "this query cell is the same type as this reference cell"

    corr_sum:
        Array of floats. Each row is a query cell. Each column
        is a reference cell. The values are the sum of the
        correlation values over the bootstrapping iterations
        that assigned that query cell to that reference cell.
    """
    n_markers = query_gene_data.shape[1]
    marker_idx = np.arange(n_markers)
    n_bootstrap = np.round(bootstrap_factor*n_markers).astype(int)
    if n_markers > 0:
        n_bootstrap = max(n_bootstrap, 1)

    result_shape = (query_gene_data.shape[0], reference_gene_data.shape[0])

    vote_dtype = choose_int_dtype((0, bootstrap_iteration))

    votes = np.zeros(
        result_shape,
        dtype=vote_dtype)

    corr_sum = np.zeros(
        result_shape,
        dtype=float)

    neighbors = []
    corr = []

    # query_idx is needed to associate each vote with its row
    # in the votes array
    query_idx = np.arange(query_gene_data.shape[0])

    t = time.time()
    for i_iteration in range(bootstrap_iteration):
        t2 = time.time()
        chosen_idx = rng.choice(marker_idx, n_bootstrap, replace=False)
        chosen_idx = np.sort(chosen_idx)
        bootstrap_query = query_gene_data[:, chosen_idx]
        bootstrap_reference = reference_gene_data[:, chosen_idx]
        update_timer("looppreproc", t2, timers)

        t3 = time.time()
        (these_neighbors,
         these_corr) = distance_utils.correlation_nearest_neighbors(
            baseline_array=bootstrap_reference,
            query_array=bootstrap_query,
            gpu_index=gpu_index,
            timers=timers,
            return_correlation=True)
        update_timer("correlation_nearest_neighbors", t3, timers)

        t3 = time.time()
        neighbors.append(these_neighbors)
        corr.append(these_corr)
        update_timer("neighbor_assign", t3, timers)

    if use_torch():
        t = time.time()
        neighbors = torch.stack(neighbors)
        corr = torch.stack(corr)
        update_timer("stack", t, timers)

        t = time.time()
        neighbors = neighbors.detach().cpu().numpy()
        corr = corr.detach().cpu().numpy()
        update_timer("tocpu", t, timers)

    for nearest_neighbors, corr_values in zip(neighbors, corr):
        t4 = time.time()
        votes[query_idx, nearest_neighbors] += 1
        corr_sum[query_idx, nearest_neighbors] += corr_values
        update_timer("votes_counter", t4, timers)

    update_timer("tally_loop", t, timers)

    return votes, corr_sum


def aggregate_votes(
        vote_array,
        correlation_array,
        reference_types):
    """
    Take the raw results of tally_votes and combine any columns
    that point to the same reference type (i.e for cases where
    we are matching to a non-leaf node in the taxonomy tree).

    Parameters
    ----------
    vote_array:
        (n_query, n_reference) array of votes
    correlation_array:
        (n_query, n_reference) array of correlation sums
    reference_types:
        (n_reference,) array of cell types associated with the
        reference "cells" (i.e. the leaf nodes of the taxonomy)

    Returns
    -------
    vote_array_agg:
        vote_array with columns that point to the same reference_type
        combined
    correlation_array_agg:
        correlation_array with columns that point to the same
        reference_type combined
    reference_types_agg:
        array of reference types indicating which types the columns
        of vote_array_agg and correlation_agg point to
    """

    reference_types = np.array(reference_types)
    unq_types = list(set(reference_types))
    unq_types.sort()
    type_to_idx = {
        t: np.where(reference_types == t)[0]
        for t in unq_types}

    n_query = vote_array.shape[0]
    n_unq = len(unq_types)
    vote_array_agg = np.zeros((n_query, n_unq), dtype=int)
    corr_array_agg = np.zeros((n_query, n_unq), dtype=float)

    for new_idx, ref_type in enumerate(unq_types):
        col_idx = type_to_idx[ref_type]
        vote_array_agg[:, new_idx] = vote_array[:, col_idx].sum(axis=1)
        corr_array_agg[:, new_idx] = correlation_array[:, col_idx].sum(axis=1)

    return (vote_array_agg,
            corr_array_agg,
            unq_types)


def update_previously_assigned(
        result,
        child_level,
        chosen_idx,
        previously_assigned):

    assignment = [
        result[ii][child_level]['assignment']
        for ii in chosen_idx
    ]

    # populate the dict keeping track of the rows in
    # full_query_gene_data that were assigned to each
    # possible child type

    type_to_idx = dict()
    idx_to_type = []
    for idx, celltype in enumerate(set(assignment)):
        type_to_idx[celltype] = idx
        idx_to_type.append(celltype)

    assignment_idx = np.array([type_to_idx[celltype]
                               for celltype in assignment])

    for idx in range(len(idx_to_type)):
        celltype = idx_to_type[idx]
        assigned_this = (assignment_idx == idx)
        assigned_this = chosen_idx[assigned_this]
        previously_assigned[child_level][celltype] = assigned_this

    return previously_assigned


def update_mapping_result(
        result,
        child_level,
        cell_idx,
        votes,
        corr_sum,
        reference_types,
        n_assignments,
        taxonomy_tree,
        output_taxonomy_tree):

    parent_level = None
    if child_level is not None:
        parent_level = taxonomy_tree.parent_level(child_level)

    if votes is None:
        n_cells = len(cell_idx)
        votes = np.ones((n_cells, 1))
        corr_sum = np.ones((n_cells, 1), dtype=float)
        if parent_level is not None:
            for ii, cell_id in enumerate(cell_idx):
                corr_sum[ii, 0] = (
                   result[cell_id][parent_level]['avg_correlation']
                )

    result = _update_mapping_result(
        result=result,
        current_level=child_level,
        cell_idx=cell_idx,
        votes=votes,
        corr_sum=corr_sum,
        reference_types=reference_types,
        n_assignments=n_assignments,
        taxonomy_tree=taxonomy_tree,
        directly_assigned=True)

    # backfill any levels above the child level using these votes
    if output_taxonomy_tree is not None:
        cell0 = cell_idx[0]
        child_array = np.array([
            result[ii][child_level]['assignment']
            for ii in cell_idx
        ])
        for backfill_level in output_taxonomy_tree.hierarchy:

            if backfill_level == child_level:
                break

            if backfill_level in result[cell0]:
                continue

            (backfill_votes,
             backfill_corr,
             backfill_types) = infer_election(
                 votes=votes,
                 corr_sum=corr_sum,
                 reference_types=reference_types,
                 reference_level=child_level,
                 inference_level=backfill_level,
                 taxonomy_tree=output_taxonomy_tree)

            result = _update_mapping_result(
                result=result,
                current_level=backfill_level,
                cell_idx=cell_idx,
                votes=backfill_votes,
                corr_sum=backfill_corr,
                reference_types=backfill_types,
                n_assignments=n_assignments,
                taxonomy_tree=output_taxonomy_tree,
                directly_assigned=False,
                known_child_level=child_level,
                known_child_array=child_array
            )

    return result


def _update_mapping_result(
        result,
        current_level,
        cell_idx,
        votes,
        corr_sum,
        reference_types,
        n_assignments,
        taxonomy_tree,
        directly_assigned,
        known_child_level=None,
        known_child_array=None):
    """
    known_child_level:
        level in the taxonomy tree at which the last election
        did take place (a descendant of the assignments being inferred here)
    known_child_array:
        (n_cells,) array of assignments made at known_child_level
        if known_child_level and known_child_array are specified,
        the assignment
    """

    has_kcl = (known_child_level is not None)
    has_kca = (known_child_array is not None)
    if (has_kcl and not has_kca) or (has_kca and not has_kcl):
        raise RuntimeError(
            "Must specify both known_child_level and known_child_array "
            "or neither. You have:\n"
            f"level: {known_child_level}\n"
            f"array: {known_child_array}"
        )

    parent_level = None
    if current_level is not None:
        parent_level = taxonomy_tree.parent_level(current_level)

    # find last parent that was directly assigned; use that level's
    # aggregate_probability as the base for aggregate probability
    # at the current level
    cell0 = cell_idx[0]
    effective_child = current_level
    while True:
        candidate_parent = taxonomy_tree.parent_level(effective_child)
        if candidate_parent is None:
            dir_assn_parent_level = None
            break
        is_present = (candidate_parent in result[cell0])
        is_direct = False
        if is_present:
            is_direct = result[cell0][candidate_parent]['directly_assigned']
        if is_direct:
            dir_assn_parent_level = candidate_parent
            break
        effective_child = candidate_parent

    # if there is a known child node to which we must enforce
    # consistent inheritance, propagate that through to
    # a forced_assignment
    if has_kcl:
        forced_assignments = np.array([
            taxonomy_tree.parents(
                level=known_child_level,
                node=known_child)[current_level]
            for known_child in known_child_array
        ])
    else:
        forced_assignments = None

    (assignment,
     bootstrapping_probability,
     avg_corr,
     runners_up) = reshape_type_assignment(
                     votes=votes,
                     corr_sum=corr_sum,
                     reference_types=reference_types,
                     n_assignments=n_assignments,
                     forced_assignments=forced_assignments)

    # assign cells to their chosen current_level nodes
    for i_cell, assigned_type, prob, corr, r_up in zip(
                    cell_idx,
                    assignment,
                    bootstrapping_probability,
                    avg_corr,
                    runners_up):

        if r_up is None:
            runner_up_assignments = []
            runner_up_correlation = []
            runner_up_probability = []
        else:
            runner_up_assignments = [
                this[0] for this in r_up if this[1]]
            runner_up_correlation = [
                this[2] for this in r_up if this[1]]
            runner_up_probability = [
                this[3] for this in r_up if this[1]]

        if dir_assn_parent_level is None:
            parent_probability = 1.0
        else:
            parent_probability = (
                result[i_cell][dir_assn_parent_level]['aggregate_probability']
            )

        aggregate_probability = parent_probability * prob

        if corr is None:
            if parent_level is not None:
                corr = result[i_cell][parent_level]['avg_correlation']

        result[i_cell][current_level] = {
            'assignment': assigned_type,
            'bootstrapping_probability': prob,
            'aggregate_probability': aggregate_probability,
            'avg_correlation': corr,
            'runner_up_assignment': runner_up_assignments,
            'runner_up_correlation': runner_up_correlation,
            'runner_up_probability': runner_up_probability,
            'directly_assigned': directly_assigned}

    return result
