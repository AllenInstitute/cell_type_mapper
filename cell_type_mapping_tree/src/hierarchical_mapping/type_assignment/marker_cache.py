import h5py
import json
import multiprocessing
import numpy as np
import time

from hierarchical_mapping.utils.utils import (
    print_timing)

from hierarchical_mapping.utils.multiprocessing_utils import (
    winnow_process_list)

from hierarchical_mapping.utils.taxonomy_utils import (
    get_all_leaf_pairs)

from hierarchical_mapping.marker_selection.utils import (
    select_marker_genes)


def create_marker_gene_cache(
        cache_path,
        score_path,
        query_gene_names,
        taxonomy_tree,
        marker_genes_per_pair,
        n_processors=6):
    """
    Populate the temporary HDF5 file with the lists of marker
    genes for each parent node.

    Parameters
    ----------
    cache_path:
        The file to be written
    score_path:
        Path to the precomputed scores
    query_gene_names:
        list of gene names in the query dataset
    taxonomy_tree:
        Dict encoding the cell type taxonomy
    marker_genes_per_pair:
        Ideal number of marker genes to choose per leaf pair
    """
    print(f"creating marker gene cache in {cache_path}")
    t0 = time.time()

    parent_node_list = [None]
    hierarchy = taxonomy_tree['hierarchy']
    with h5py.File(cache_path, 'w') as out_file:
        out_file.create_group('None')
        for level in hierarchy[:-1]:
            out_file.create_group(level)
            these_parents = list(taxonomy_tree[level].keys())
            these_parents.sort()
            for parent in these_parents:
                parent_node_list.append((level, parent))
                out_file[level].create_group(parent)

        out_file.create_dataset(
            'parent_node_list',
            data=json.dumps(parent_node_list).encode('utf-8'))

    # shuffle node list so all the big parents
    # don't end up in the same worker;
    # note that the largest parents ought to be at the
    # front of parent_node_list because of hw it was generated
    # (grossest taxonomic level first)
    n_chunks = 6*n_processors
    if n_chunks > len(parent_node_list):
        n_chunks = max(1, len(parent_node_list) // 3)

    chunk_list = []
    for ii in range(n_chunks):
        chunk_list.append([])
    for ii in range(len(parent_node_list)):
        jj = ii % n_chunks
        chunk_list[jj].append(parent_node_list[ii])
    parent_node_list = []
    for chunk in chunk_list:
        parent_node_list += chunk
    del chunk_list

    n_parents = len(parent_node_list)
    n_per_process = max(1, n_parents//n_chunks)
    process_list = []
    mgr = multiprocessing.Manager()
    output_lock = mgr.Lock()

    # keep track of the genes we actually need
    # for all queries
    query_marker_list = mgr.list()
    query_marker_list.append(set())

    ct = 0
    for i0 in range(0, n_parents, n_per_process):
        i1 = i0 + n_per_process
        p = multiprocessing.Process(
                target=_marker_gene_worker,
                kwargs={
                    'parent_node_list': parent_node_list[i0:i1],
                    'taxonomy_tree': taxonomy_tree,
                    'score_path': score_path,
                    'query_gene_names': query_gene_names,
                    'marker_genes_per_pair': marker_genes_per_pair,
                    'query_marker_list': query_marker_list,
                    'output_path': cache_path,
                    'output_lock': output_lock})
        p.start()
        process_list.append(p)
        while len(process_list) >= n_processors:
            n0 = len(process_list)
            process_list = winnow_process_list(process_list)
            n1 = len(process_list)
            if n1 < n0:
                ct += n_per_process*(n0-n1)
                print_timing(
                    t0=t0,
                    tot_chunks=n_parents,
                    i_chunk=ct,
                    unit='hr')

    for p in process_list:
        p.join()

    # list the set of genes we actually need to keep
    # from the query set for all comparisons
    query_genes = np.array(list(query_marker_list[0]))
    query_genes = np.sort(query_genes)
    with h5py.File(cache_path, "a") as in_file:
        in_file.create_dataset(
            "all_query_genes",
            data=query_genes)


def _marker_gene_worker(
        parent_node_list,
        taxonomy_tree,
        score_path,
        query_gene_names,
        marker_genes_per_pair,
        query_marker_list,
        output_path,
        output_lock):

    grp_to_results = dict()
    for parent_node in parent_node_list:
        leaf_pair_list = get_all_leaf_pairs(
            taxonomy_tree=taxonomy_tree,
            parent_node=parent_node)

        if len(leaf_pair_list) > 0:
            marker_genes = select_marker_genes(
                score_path=score_path,
                leaf_pair_list=leaf_pair_list,
                query_genes=query_gene_names,
                genes_per_pair=marker_genes_per_pair,
                n_processors=1,
                rows_at_a_time=1000000)
        else:
            marker_genes = {'reference': np.zeros(0, dtype=int),
                            'query': np.zeros(0, dtype=int)}

        if parent_node is None:
            grp = 'None'
        else:
            grp = f'{parent_node[0]}/{parent_node[1]}'
        grp_to_results[grp] = marker_genes

    with output_lock:
        query_set = query_marker_list[0]

        with h5py.File(output_path, 'a') as out_file:
            for grp in grp_to_results:
                reference = grp_to_results[grp]['reference']
                query = grp_to_results[grp]['query']

                query_set = query_set.union(set(query))

                out_grp = out_file[grp]
                out_grp.create_dataset('reference', data=reference)
                out_grp.create_dataset('query', data=query)

        query_marker_list[0] = query_set
