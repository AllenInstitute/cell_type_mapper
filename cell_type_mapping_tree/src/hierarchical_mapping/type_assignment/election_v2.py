"""
This implementation of type assignment will handle one parent
at a time, presumably so that larger chunks of data can be
run through np.dot at once.
"""
import h5py
import json
import multiprocessing
import numpy as np
import pathlib
import tempfile

from hierarchical_mapping.utils.utils import (
    _clean_up,
    mkstemp_clean)

from hierarchical_mapping.utils.multiprocessing_utils import (
    winnow_process_list)

from hierarchical_mapping.utils.anndata_utils import (
    read_df_from_h5ad)

from hierarchical_mapping.anndata_iterator.anndata_iterator import (
    AnnDataRowIterator)

from hierarchical_mapping.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)

from hierarchical_mapping.type_assignment.matching import (
   get_leaf_means)

from hierarchical_mapping.type_assignment.matching import (
    assemble_markers)

from hierarchical_mapping.type_assignment.election import (
    _run_type_assignment)


def run_type_assignment_on_h5ad_v2(
        query_h5ad_path,
        precomputed_stats_path,
        marker_gene_cache_path,
        taxonomy_tree,
        n_processors,
        chunk_size,
        bootstrap_factor,
        bootstrap_iteration,
        rng,
        normalization='log2CPM',
        tmp_dir=None,
        log=None,
        max_gb=10):
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
        hierarchical_mapping.taxonomty.taxonomy_tree.TaxonomyTree
        ecoding the taxonomy tree

    n_processors:
        Number of independent worker processes to spin up

    chunk_size:
        Number of rows (cells) to process at a time.
        Note: if this is larger than n_rows/n_processors,
        then this will get changed to n_rows/n_processors

    bootstrap_factor:
        Fraction (<=1.0) by which to sampel the marker gene set
        at each bootstrapping iteration

    bootstrap_iteration:
        How many booststrap iterations to run when assigning
        cells to cell types

    rng:
        A random number generator

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

    Returns
    -------
    A list of dicts. Each dict correponds to a cell in full_query_gene_data.
    The dict maps level in the hierarchy to the type (at that level)
    the cell has been assigned.

    Dict will look like
        {'cell_id': id_of_cell,
         taxonomy_level1 : {'assignment': chosen_node,
                           'confidence': fraction_of_votes},
         taxonomy_level2 : {'assignment': chosen_node,
                           'confidence': fraction_of_votes},
         ...}
    """

    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(
            dir=tmp_dir,
            prefix='election_runner_dir_'))

    root_output_dir = tmp_dir / 'None'
    root_output_dir.mkdir()

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

    leaf_node_matrix = get_leaf_means(
        taxonomy_tree=taxonomy_tree,
        precompute_path=precomputed_stats_path)

    marker_lookup = assemble_markers(
        marker_cache_path=marker_gene_cache_path,
        taxonomy_tree=taxonomy_tree,
        parent_node=None)

    all_ref_identifiers = marker_lookup['all_ref_identifiers']
    all_query_identifiers = marker_lookup['all_query_identifiers']
    reference_markers = marker_lookup['reference_markers']
    raw_query_markers = marker_lookup['query_markers']

    chunk_iterator = AnnDataRowIterator(
        h5ad_path=query_h5ad_path,
        row_chunk_size=chunk_size,
        tmp_dir=tmp_dir,
        log=log,
        max_gb=max_gb)

    mgr = multiprocessing.Manager()
    process_list = []
    for chunk in chunk_iterator:
        chunk_data = chunk[0]
        r0 = chunk[1]
        r1 = chunk[2]

        query_data = CellByGeneMatrix(
            data=chunk_data,
            gene_identifiers=all_query_identifiers,
            normalization=normalization)

        these_cell_names = query_cell_names[r0:r1]
        p = multiprocessing.Process(
                target=_root_worker,
                kwargs={
                    'full_query_gene_data': query_data,
                    'query_cell_names': these_cell_names,
                    'leaf_node_matrix': leaf_node_matrix,
                    'taxonomy_tree': taxonomy_tree,
                    'parent_node': None,
                    'bootstrap_factor': bootstrap_factor,
                    'bootstrap_iteration': bootstrap_iteration,
                    'rng': np.random.default_rng(rng.integers(1, 2**32-1)),
                    'all_ref_identifiers': all_ref_identifiers,
                    'all_query_identifiers': all_query_identifiers,
                    'reference_markers': reference_markers,
                    'raw_query_markers': raw_query_markers,
                    'output_dir': root_output_dir})
        p.start()
        process_list.append(p)
        while len(process_list) >= n_processors:
            process_list = winnow_process_list(process_list)

    while len(process_list) > 0:
        process_list = winnow_process_list(process_list)

    _clean_up(tmp_dir)


def _root_worker(
        full_query_gene_data,
        query_cell_names,
        leaf_node_matrix,
        all_ref_identifiers,
        all_query_identifiers,
        reference_markers,
        raw_query_markers,
        taxonomy_tree,
        parent_node,
        bootstrap_factor,
        bootstrap_iteration,
        rng,
        output_dir):

    output_path = mkstemp_clean(
        dir=output_dir,
        suffix='.json')

    (assignment,
     confidence) = _run_type_assignment(
         full_query_gene_data=full_query_gene_data,
         leaf_node_matrix=leaf_node_matrix,
         taxonomy_tree=taxonomy_tree,
         parent_node=parent_node,
         bootstrap_factor=bootstrap_factor,
         bootstrap_iteration=bootstrap_iteration,
         rng=rng,
         marker_gene_cache_path=None,
         all_ref_identifiers=all_ref_identifiers,
         all_query_identifiers=all_query_identifiers,
         reference_markers=reference_markers,
         raw_query_markers=raw_query_markers)

    output = dict()
    output['parent_node'] = str(parent_node)

    for cell_id, a, c in zip(query_cell_names, assignment, confidence):
        output[cell_id] = {
           'assignment': a,
           'confidence': c}

    with open(output_path, 'w') as out_file:
        out_file.write(json.dumps(output, indent=2))
