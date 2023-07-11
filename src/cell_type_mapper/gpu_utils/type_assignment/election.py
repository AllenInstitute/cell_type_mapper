import os
import h5py
import json
import time
import numpy as np
import torch  # type: ignore
import torch.nn as nn

from cell_type_mapper.utils.torch_utils import (
    is_cuda_available)
from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad)
from cell_type_mapper.gpu_utils.anndata_iterator.anndata_iterator import (
    get_torch_dataloader)
from cell_type_mapper.type_assignment.election import (
    run_type_assignment, save_results)
from cell_type_mapper.type_assignment.matching import (
   get_leaf_means)
from cell_type_mapper.gpu_utils.utils.utils import (
    get_timers, AverageMeter, ProgressMeter)
from cell_type_mapper.utils.utils import (
    update_timer)


class TypeAssignment(nn.Module):
    def __init__(self):
        super(TypeAssignment, self).__init__()

    def forward(self, x, config):
        assignment = run_type_assignment(
            full_query_gene_data=x,
            leaf_node_matrix=config["leaf_node_matrix"],
            marker_gene_cache_path=config["marker_gene_cache_path"],
            taxonomy_tree=config["taxonomy_tree"],
            bootstrap_factor=config["bootstrap_factor"],
            bootstrap_iteration=config["bootstrap_iteration"],
            rng=np.random.default_rng(config["rng"].integers(99, 2**32)),
            gpu_index=config["gpu_index"],
            timers=config["timers"])
        return assignment


def run_type_assignment_on_h5ad_gpu(
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
        max_gb=10,
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

    results_output_path:
        Output path for run assignment (a directory).
        If given will save individual chunks of the run assignment process
        to separate files.

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
    if log is not None:
        log.info("Running GPU implementation of type assignment.")

    # read query file
    obs = read_df_from_h5ad(query_h5ad_path, 'obs')
    query_cell_names = list(obs.index.values)
    n_rows = len(obs)
    num_workers = min(n_processors, np.ceil(n_rows/chunk_size).astype(int))
    del obs

    with h5py.File(marker_gene_cache_path, 'r', swmr=True) as in_file:
        all_query_identifiers = json.loads(
            in_file["query_gene_names"][()].decode("utf-8"))
        all_query_markers = [
            all_query_identifiers[ii]
            for ii in in_file["all_query_markers"][()]]

    if is_cuda_available():
        gpu_index = 0
        device = torch.device(f'cuda:{gpu_index}')
    else:
        gpu_index = None
        device = torch.device('cpu')

    dataloader = get_torch_dataloader(query_h5ad_path,
                                      chunk_size,
                                      all_query_identifiers,
                                      normalization,
                                      all_query_markers,
                                      device=device,
                                      num_workers=num_workers,
                                      max_gb=max_gb,
                                      tmp_dir=tmp_dir)

    # get a CellByGeneMatrix of average expression
    # profiles for each leaf in the taxonomy
    leaf_node_matrix = get_leaf_means(
        taxonomy_tree=taxonomy_tree,
        precompute_path=precomputed_stats_path)

    type_assignment_model = TypeAssignment()

    config = dict()
    config["leaf_node_matrix"] = leaf_node_matrix
    config["marker_gene_cache_path"] = marker_gene_cache_path
    config["taxonomy_tree"] = taxonomy_tree
    config["bootstrap_factor"] = bootstrap_factor
    config["bootstrap_iteration"] = bootstrap_iteration
    config["rng"] = rng
    config["gpu_index"] = gpu_index

    print("starting type assignment")
    print_freq = 1

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    timers = get_timers()
    config["timers"] = timers

    timerlist = [batch_time, data_time] + list(timers.values())
    progress = ProgressMeter(
        len(dataloader),
        timerlist,
        prefix="Assignment: ")
    end = time.time()
    output_list = []
    for ii, (data, r0, r1) in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        query_cell_names_chunk = query_cell_names[r0:r1]

        t = time.time()
        assignment = type_assignment_model(data, config)
        update_timer("type_assignment", t, timers)

        t = time.time()
        for idx in range(len(assignment)):
            assignment[idx]['cell_id'] = query_cell_names_chunk[idx]
        update_timer("loop", t, timers)

        t = time.time()
        if results_output_path is not None:
            this_output_path = os.path.join(results_output_path,
                                            f"{r0}_{r1}_assignment.json")
            save_results(assignment, this_output_path)
        else:
            output_list += assignment
        update_timer("results", t, timers)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if ii % print_freq == 0:
            progress.display(ii + 1)

    output_list = list(output_list)
    return output_list
