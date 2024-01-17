import copy
import h5py

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)


def truncate_precomputed_stats_file(
        src_path,
        dst_path,
        new_hierarchy):
    """
    Read in the precomputed_stats file at src path.
    Truncate its taxonomy tree so that the hierarchy looks like
    new_hierarchy. Write the result to dst_path.
    """
    old_tree = TaxonomyTree.from_precomputed_stats(src_path)

    if new_hierarchy == old_tree.hierarchy:
        msg = (
            f"{src_path}\nalready conforms to the requested "
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
            f"the taxonomy of\n{src_path}\n"
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

    with h5py.File(src_path, 'r') as src:
        with h5py.File(dst_path, 'w') as dst:
            dst.create_dataset(
                'taxonomy_tree',
                data=new_tree.to_str().encode('utf-8'))
            dst.create_dataset(
                'col_names',
                data=src['col_names'][()])
