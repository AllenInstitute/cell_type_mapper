import copy
import h5py
import json
import numpy as np
import pathlib

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)


def truncate_precomputed_stats_file(
        input_path,
        output_path,
        new_hierarchy):
    """
    Read in the precomputed_stats file at input_path.
    Truncate its taxonomy tree so that the hierarchy looks like
    new_hierarchy. Write the result to output_path.
    """

    metadata = {
        'input_path': str(pathlib.Path(input_path).absolute().resolve()),
        'output_path': str(pathlib.Path(output_path).absolute().resolve()),
        'new_hierarchy': list(new_hierarchy)
    }

    old_tree = TaxonomyTree.from_precomputed_stats(input_path)

    if new_hierarchy == old_tree.hierarchy:
        msg = (
            f"{input_path}\nalready conforms to the requested "
            "taxonomic hierarchy."
        )
        raise RuntimeError(msg)

    bad_levels = []
    for level in new_hierarchy:
        if level not in old_tree.hierarchy:
            bad_levels.append(level)
    if len(bad_levels) > 0:
        msg = (
            f"Levels\n{bad_levels}\nare not in "
            f"the taxonomy of\n{input_path}\n"
            f"Unclear how to proceed."
        )
        raise RuntimeError(msg)

    level_to_idx = {
        l: ii for ii, l in enumerate(old_tree.hierarchy)
    }
    new_idx = [level_to_idx[level] for level in new_hierarchy]
    sorted_new_idx = copy.deepcopy(new_idx)
    sorted_new_idx.sort()
    if not new_idx == sorted_new_idx:
        msg = (
            f"You asked for hierarchy\n{new_hierarchy}\n"
            "However, the old taxonomy tree has hierarchy\n"
            f"{old_tree.hierarchy}\n"
            "You cannot shuffle the order of taxonomic levels, "
            "just remove levels from the hierarchy."
        )
        raise RuntimeError(msg)

    to_drop = []
    for level in old_tree.hierarchy:
        if level not in new_hierarchy:
            to_drop.append(level)

    new_tree = None

    for level in to_drop:
        if new_tree is None:
            src_tree = old_tree
        else:
            src_tree = new_tree

        if level == src_tree.leaf_level:
            new_tree = src_tree.drop_leaf_level()
        else:
            new_tree = src_tree.drop_level(level)

    del src_tree

    same_leaves = False
    if new_tree.leaf_level == old_tree.leaf_level:
        same_leaves = True

    if not same_leaves:
        new_leaf_to_row = {
            leaf: ii for ii, leaf in enumerate(new_tree.all_leaves)
        }

        new_leaf_to_old_leaves = dict()
        for old_leaf in old_tree.all_leaves:
            parents = old_tree.parents(
                old_tree.leaf_level,
                node=old_leaf)
            new_leaf = parents[new_tree.leaf_level]
            if new_leaf not in new_leaf_to_old_leaves:
                new_leaf_to_old_leaves[new_leaf] = []
            new_leaf_to_old_leaves[new_leaf].append(old_leaf)

        not_arrays = (
            'metadata',
            'cluster_to_row',
            'col_names',
            'taxonomy_tree'
        )
    else:
        not_to_copy = ('metadata', 'taxonomy_tree', 'col_names')

    with h5py.File(input_path, 'r') as src:

        if not same_leaves:
            old_leaf_to_row = json.loads(
                src['cluster_to_row'][()].decode('utf-8'))

        with h5py.File(output_path, 'w') as dst:
            dst.create_dataset(
                'metadata',
                data=json.dumps(metadata).encode('utf-8'))
            dst.create_dataset(
                'taxonomy_tree',
                data=new_tree.to_str().encode('utf-8'))
            dst.create_dataset(
                'col_names',
                data=src['col_names'][()])

            if same_leaves:
                for dataset in src.keys():
                    if dataset in not_to_copy:
                        continue

                    src_dataset = src[dataset]

                    dst.create_dataset(
                        dataset,
                        data=src_dataset[()],
                        chunks=src_dataset.chunks)
            else:
                dst.create_dataset(
                    'cluster_to_row',
                    data=json.dumps(new_leaf_to_row).encode('utf-8'))

                for dataset in src.keys():
                    if dataset in not_arrays:
                        continue

                    src_dataset = src[dataset]

                    new_data = _convert_to_new_leaves(
                        data_array=src_dataset[()],
                        old_leaf_to_row=old_leaf_to_row,
                        new_leaf_to_row=new_leaf_to_row,
                        new_leaf_to_old_leaves=new_leaf_to_old_leaves)

                    chunks = None
                    if src_dataset.chunks is not None:
                        if len(src_dataset.shape) == 2:
                            chunks = (1, src_dataset.chunks[1])

                    dst.create_dataset(
                        dataset,
                        data=new_data,
                        chunks=chunks)


def _convert_to_new_leaves(
        data_array,
        old_leaf_to_row,
        new_leaf_to_row,
        new_leaf_to_old_leaves):
    """
    Sum together the rows in a precomputed_stats data array
    to reflect the results of truncating data to a new taxonomy.

    Parameters
    -----------
    data_array:
        The array being aggregated
    old_leaf_to_row:
        Old mapping from leaf nodes to row in data_array
    new_leaf_to_row:
        New mapping from leaf_nodes_to_row in data_array
    new_leaf_to_old_leaves:
        Dict mapping new leaf nodes to lists of the old
        leaf nodes that comprise them

    Return
    ------
    A numpy array summed according to the new taxonomy
    """
    if len(data_array.shape) == 2:
        new_array = np.zeros(
            (len(new_leaf_to_row), data_array.shape[1]),
            dtype=data_array.dtype)
    else:
        new_array = np.zeros(
            len(new_leaf_to_row),
            dtype=data_array.dtype)

    for new_leaf in new_leaf_to_old_leaves:
        dst_row = new_leaf_to_row[new_leaf]
        src_rows = [old_leaf_to_row[old_leaf]
                    for old_leaf in new_leaf_to_old_leaves[new_leaf]]
        src_rows.sort()
        src_rows = np.array(src_rows)
        if len(data_array.shape) == 2:
            new_row = data_array[src_rows, :].sum(axis=0)
            new_array[dst_row, :] = new_row
        else:
            new_row = data_array[src_rows].sum()
            new_array[dst_row] = new_row
    return new_array
