import anndata
import multiprocessing
import numpy as np
import time

from hierarchical_mapping.utils.utils import (
    print_timing)

from hierarchical_mapping.utils.multiprocessing_utils import (
    winnow_process_list)

from hierarchical_mapping.utils.distance_utils import (
    correlation_nearest_neighbors)

from hierarchical_mapping.type_assignment.matching import (
   get_leaf_means,
   assemble_query_data)


def run_type_assignment_on_h5ad(
        query_h5ad_path,
        precomputed_stats_path,
        marker_gene_cache_path,
        taxonomy_tree,
        n_processors,
        chunk_size,
        bootstrap_factor,
        bootstrap_iteration,
        rng):
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
        Dict encoding the cell type taxonomy we are matching to

    n_processors:
        Number of independent worker processes to spin up

    chunk_size:
        Number of rows (cells) to process at a time.

    bootstrap_factor:
        Fraction (<=1.0) by which to sampel the marker gene set
        at each bootstrapping iteration

    bootstrap_iteration:
        How many booststrap iterations to run when assigning
        cells to cell types

    rng:
        A random number generator

    Returns
    -------
    A list of dicts. Each dict correponds to a cell in full_query_gene_data.
    The dict maps level in the hierarchy to the type (at that level)
    the cell has been assigned.
    """

    a_data = anndata.read_h5ad(query_h5ad_path, backed='r')
    query_cell_names = list(a_data.obs_names)
    chunk_iterator = a_data.chunked_X(chunk_size=chunk_size)

    process_list = []
    mgr = multiprocessing.Manager()
    output_list = mgr.list()
    output_lock = mgr.Lock()

    tot_rows = a_data.X.shape[0]
    row_ct = 0
    t0 = time.time()

    print("starting type assignment")
    for chunk in chunk_iterator:
        r0 = chunk[1]
        r1 = chunk[2]
        name_chunk = query_cell_names[r0:r1]
        if isinstance(chunk[0], np.ndarray):
            data = chunk[0]
        else:
            data = chunk[0].toarray()

        p = multiprocessing.Process(
                target=_run_type_assignment_on_h5ad_worker,
                kwargs={
                    'query_cell_chunk': data,
                    'query_cell_names': name_chunk,
                    'precomputed_stats_path': precomputed_stats_path,
                    'marker_gene_cache_path': marker_gene_cache_path,
                    'taxonomy_tree': taxonomy_tree,
                    'bootstrap_factor': bootstrap_factor,
                    'bootstrap_iteration': bootstrap_iteration,
                    'rng': np.random.default_rng(rng.integers(99, 2**32)),
                    'output_list': output_list,
                    'output_lock': output_lock})
        p.start()
        process_list.append(p)
        while len(process_list) >= n_processors:
            n0 = len(process_list)
            process_list = winnow_process_list(process_list)
            n1 = len(process_list)
            if n1 < n0:
                row_ct += (n0-n1)*chunk_size
                print_timing(
                    t0=t0,
                    i_chunk=row_ct,
                    tot_chunks=tot_rows,
                    unit='hr')
    print("final join of worker processes")
    for p in process_list:
        p.join()

    output_list = list(output_list)
    return output_list


def _run_type_assignment_on_h5ad_worker(
        query_cell_chunk,
        query_cell_names,
        precomputed_stats_path,
        marker_gene_cache_path,
        taxonomy_tree,
        bootstrap_factor,
        bootstrap_iteration,
        rng,
        output_list,
        output_lock):

    assignment = run_type_assignment(
        full_query_gene_data=query_cell_chunk,
        precomputed_stats_path=precomputed_stats_path,
        marker_gene_cache_path=marker_gene_cache_path,
        taxonomy_tree=taxonomy_tree,
        bootstrap_factor=bootstrap_factor,
        bootstrap_iteration=bootstrap_iteration,
        rng=rng)

    for idx in range(len(assignment)):
        assignment[idx]['cell_id'] = query_cell_names[idx]

    with output_lock:
        output_list += assignment


def run_type_assignment(
        full_query_gene_data,
        precomputed_stats_path,
        marker_gene_cache_path,
        taxonomy_tree,
        bootstrap_factor,
        bootstrap_iteration,
        rng):
    """
    Assign types at all levels of the taxonomy to a set of
    query cells.

    Parameters
    ----------
    full_query_gene_data:
        (n_query_cells, n_query_genes) numpy array. The cells
        to be mapped.

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
        Dict encoding the cell type taxonomy we are matching to

    bootstrap_factor:
        Fraction (<=1.0) by which to sampel the marker gene set
        at each bootstrapping iteration

    bootstrap_iteration:
        How many booststrap iterations to run when assigning
        cells to cell types

    rng:
        A random number generator

    Returns
    -------
    A list of dicts. Each dict correponds to a cell in full_query_gene_data.
    The dict maps level in the hierarchy to the type (at that level)
    the cell has been assigned.
    """
    # get a dict mapping leaf node name
    # to mean gene expression profile
    leaf_node_lookup = get_leaf_means(
        taxonomy_tree=taxonomy_tree,
        precompute_path=precomputed_stats_path)

    # create effectively empty list of dicts to
    # store the hierarchical classification of
    # each cell in full_query_gene_data
    hierarchy = taxonomy_tree['hierarchy']
    result = []
    for i_cell in range(full_query_gene_data.shape[0]):
        this = dict()
        for level in hierarchy:
            this[level] = None
        result.append(this)

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
            k_list = list(taxonomy_tree[parent_level].keys())
            k_list.sort()
            parent_node_list = []
            for k in k_list:
                parent_node_list.append((parent_level, k))

        previously_assigned[child_level] = dict()

        for parent_node in parent_node_list:

            # only consider the query cells that were assigned
            # to the current parent
            if parent_node is None:
                chosen_idx = np.arange(full_query_gene_data.shape[0])
                chosen_query_data = full_query_gene_data
            else:
                if parent_node[1] in previously_assigned[parent_level]:
                    chosen_idx = previously_assigned[
                        parent_level][parent_node[1]]

                    chosen_query_data = full_query_gene_data[chosen_idx, :]

                else:
                    chosen_idx = []

            if len(chosen_idx) == 0:
                continue

            # see how many children this parent node has;
            # if == 1, assignment is trivial
            if parent_level is not None:
                possible_children = taxonomy_tree[parent_level][parent_node[1]]
            else:
                possible_children = list(taxonomy_tree[
                        taxonomy_tree['hierarchy'][0]].keys())

            if len(possible_children) > 1:
                assignment = _run_type_assignment(
                                full_query_gene_data=chosen_query_data,
                                leaf_node_lookup=leaf_node_lookup,
                                marker_gene_cache_path=marker_gene_cache_path,
                                taxonomy_tree=taxonomy_tree,
                                parent_node=parent_node,
                                bootstrap_factor=bootstrap_factor,
                                bootstrap_iteration=bootstrap_iteration,
                                rng=rng)
            elif len(possible_children) == 1:
                assignment = [possible_children[0]]*chosen_query_data.shape[0]
            else:
                raise RuntimeError(
                    "Not sure how to proceed;\n"
                    f"parent {parent_node}\n"
                    f"has children {possible_children}")

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

            # assign cells to their chosen child_level nodes
            for i_cell, assigned_type in zip(chosen_idx, assignment):
                result[i_cell][child_level] = assigned_type

    return result


def _run_type_assignment(
        full_query_gene_data,
        leaf_node_lookup,
        marker_gene_cache_path,
        taxonomy_tree,
        parent_node,
        bootstrap_factor,
        bootstrap_iteration,
        rng):
    """
    Assign a set of query cells to types that are children
    of a specified parent node in our taxonomy.

    Parameters
    ----------
    full_query_gene_data:
        (n_query_cells, n_query_genes) numpy array. The cells
        to be mapped.

    leaf_node_lookup:
        Dict that maps cluster names to mean gene expression
        profile of cells in that cluster.

    marker_gene_cache_path:
        Path to the HDF5 file where lists of marker genes for
        discriminating betwen clustes in our taxonomy are stored.

        Note: This file takes into account the genes available
        in the query data. So: it is specific to this combination
        of taxonomy/reference set and query data set.

    taxonomy_tree:
        A dict that encodes our cell types taxonomy

    parent_node:
        Tuple of the form (level, cell_type) that encodes
        the parent to whose children we are mapping the cells
        in the query data

    bootstrap_factor:
        Fraction (<=1.0) by which to sampel the marker gene set
        at each bootstrapping iteration

    bootstrap_iteration:
        How many booststrap iterations to run when assigning
        cells to cell types

    rng:
        A random number generator

    Returns
    -------
    A list of strings. There is one string per row in the
    full_query_gene_data array. Each string is the child of
    parent_node to which that cell was assigned.
    """

    query_data = assemble_query_data(
        full_query_data=full_query_gene_data,
        mean_profile_lookup=leaf_node_lookup,
        marker_cache_path=marker_gene_cache_path,
        taxonomy_tree=taxonomy_tree,
        parent_node=parent_node)

    result = choose_node(
        query_gene_data=query_data['query_data'],
        reference_gene_data=query_data['reference_data'],
        reference_types=query_data['reference_types'],
        bootstrap_factor=bootstrap_factor,
        bootstrap_iteration=bootstrap_iteration,
        rng=rng)

    return result


def choose_node(
         query_gene_data,
         reference_gene_data,
         reference_types,
         bootstrap_factor,
         bootstrap_iteration,
         rng):
    """
    Parameters
    ----------
    query_gene_data
        cell-by-marker-gene array of query data
    reference_gene_data
        cell-by-marker-gene array of reference data
    reference_types
        array of cell types we are chosing from (n_cells in size)
    bootstrap_factor
        Factor by which to subsample reference genes at each bootstrap
    bootstrap_iteration
        Number of bootstrapping iterations
    rng
        random number generator

    Returns
    -------
    Array of cell type assignments (majority rule)
    """

    votes = tally_votes(
        query_gene_data=query_gene_data,
        reference_gene_data=reference_gene_data,
        bootstrap_factor=bootstrap_factor,
        bootstrap_iteration=bootstrap_iteration,
        rng=rng)

    chosen_type = np.argmax(votes, axis=1)
    result = [reference_types[ii] for ii in chosen_type]
    return np.array(result)


def tally_votes(
         query_gene_data,
         reference_gene_data,
         bootstrap_factor,
         bootstrap_iteration,
         rng):
    """
    Parameters
    ----------
    query_gene_data
        cell-by-marker-gene array of query data
    reference_gene_data
        cell-by-marker-gene array of reference data
    reference_types
        array of cell types we are chosing from (n_cells in size)
    bootstrap_factor
        Factor by which to subsample reference genes at each bootstrap
    bootstrap_iteration
        Number of bootstrapping iterations
    rng
        random number generator

    Returns
    -------
    Array of ints. Each row is a query cell. Each column is a
    reference cell. The value is how many iterations voted for
    "this query cell is the same type as this reference cell"
    """
    n_markers = query_gene_data.shape[1]
    marker_idx = np.arange(n_markers)
    n_bootstrap = np.round(bootstrap_factor*n_markers).astype(int)

    votes = np.zeros((query_gene_data.shape[0], reference_gene_data.shape[0]),
                     dtype=int)

    # query_idx is needed to associate each vote with its row
    # in the votes array
    query_idx = np.arange(query_gene_data.shape[0])

    for i_iteration in range(bootstrap_iteration):
        chosen_idx = rng.choice(marker_idx, n_bootstrap, replace=False)
        chosen_idx = np.sort(chosen_idx)
        bootstrap_query = query_gene_data[:, chosen_idx]
        bootstrap_reference = reference_gene_data[:, chosen_idx]

        nearest_neighbors = correlation_nearest_neighbors(
            baseline_array=bootstrap_reference,
            query_array=bootstrap_query)

        votes[query_idx, nearest_neighbors] += 1
    return votes
