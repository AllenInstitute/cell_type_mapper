import h5py
import json

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)


def run_leaf_census(
        precompute_path_list):
    """
    Take a list of precomputed_stats_paths describing different
    datasets assigned to the samet taxonomy. Determine which files
    contain how many cells in which leaf nodes.

    Parameters
    ----------
    precompute_path_list:
        List of paths to precomputed stats files

    Returns
    -------
    leaf_to_census:
        Dict mapping leaf nodes in the taxonomy to dicts indicating
        how many cells in those leaf nodes exist in each file.

    taxonomy_tree:
        The TaxonomyTree relevant to these files.

    Notes
    -----
    Raises and error if the files do not contain the same
    taxonomy tree.
    """

    taxonomy_tree = None
    taxonomy_src = None
    leaf_to_census = dict()

    for pth in precompute_path_list:
        this_tree = TaxonomyTree.from_precomputed_stats(
            stats_path=pth)
        if taxonomy_tree is None:
            taxonomy_tree = this_tree
            taxonomy_src = pth
        else:
            if not taxonomy_tree.is_equal_to(this_tree):
                raise RuntimeError(
                    f"{pth}\npoints to a different taxonomy tree than\n"
                    f"{taxonomy_src}")

        with h5py.File(pth, "r") as src:
            cluster_to_row = json.loads(
                src['cluster_to_row'][()].decode('utf-8'))
            n_cells = src['n_cells'][()]

        for node in taxonomy_tree.nodes_at_level(taxonomy_tree.leaf_level):
            idx = cluster_to_row[node]
            if node not in leaf_to_census:
                leaf_to_census[node] = dict()
            leaf_to_census[node][pth] = n_cells[idx]

    return leaf_to_census, taxonomy_tree
