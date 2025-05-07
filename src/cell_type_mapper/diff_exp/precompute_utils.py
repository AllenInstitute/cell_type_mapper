import h5py
import json
import numpy as np
import pathlib

from cell_type_mapper.utils.h5_utils import (
    copy_h5_excluding_data)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad
)

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


def merge_precompute_files(
        precompute_path_list,
        output_path):
    """
    Take a set of precomputed_stats files that describe different
    datasets in the same taxonomy. Create a single precomputed
    stats file in which each cluster only contains the row from the
    original file in which it had the most cells.

    Parameters
    ----------
    precompute_path_list:
        List of paths to the original precomputed stats files
    output_path:
        Where to write the merged precomputed stats file.
    """
    precompute_path_list.sort()

    (census,
     taxonomy_tree) = run_leaf_census(precompute_path_list)

    most_cells = 0
    most_path = None
    for pth in precompute_path_list:
        with h5py.File(pth, 'r') as src:
            ntot = src['n_cells'][()].sum()
            if ntot > most_cells or most_path is None:
                most_path = pth
                most_cells = ntot

    copy_h5_excluding_data(
        src_path=most_path,
        dst_path=output_path,
        excluded_groups=['metadata'],
        excluded_datasets=['metadata'])

    keys_to_skip = set(
        ['taxonomy_tree',
         'n_cells',
         'col_names',
         'cluster_to_row',
         'metadata']
    )

    with h5py.File(output_path, 'a') as dst:
        dst_cluster_lookup = dst['cluster_to_row'][()]
        dst_col_names = dst['col_names'][()]
        for pth in precompute_path_list:
            if pth == most_path:
                continue
            with h5py.File(pth, 'r') as src:
                if src['cluster_to_row'][()] != dst_cluster_lookup:
                    raise RuntimeError(
                        "provided files have different "
                        "cluster_to_row mappings")
                if src['col_names'][()] != dst_col_names:
                    raise RuntimeError(
                        "provided files have different col_names")
                src_n_cells = src['n_cells'][()]
                dst_n_cells = dst['n_cells'][()]
                to_replace = np.where(src_n_cells > dst_n_cells)[0]

                dst['n_cells'][to_replace] = src['n_cells'][to_replace]
                for key in dst.keys():
                    if key in keys_to_skip:
                        continue
                    for idx in to_replace:
                        dst[key][idx, :] = src[key][idx, :]
            dst.flush()


def drop_nodes_from_precomputed_stats(
        src_path,
        dst_path,
        node_list,
        clobber=False):
    """
    Create a new precomputed_stats file by dropping nodes from the
    taxonomy tree of another precomputed_stats file.

    Parameters
    ----------
    src_path:
        path to the original precomputed_stats.h5 file
    dst_path:
        path to the new precomputed_stats.h5 file to be written
    node_list:
        a list of (level, node) tuples indicating which nodes are to
        be dropped from the taxonomy tree
    clobber:
        a boolean. If False and dst_path already exists, throw an exception.
        If True, overwrite pre-existing dst_path.
    """
    src_path = pathlib.Path(src_path)
    dst_path = pathlib.Path(dst_path)
    if dst_path.exists():
        if not clobber:
            raise RuntimeError(
                f"{dst_path} already exists. To overwrite, run "
                "with clobber=True."
            )

    taxonomy_tree = TaxonomyTree.from_precomputed_stats(src_path)
    for node in node_list:
        mapped_node = taxonomy_tree.name_to_node(
            level=node[0],
            node=node[1]
        )

        taxonomy_tree = taxonomy_tree.drop_node(
            level=mapped_node[0],
            node=mapped_node[1]
        )

    new_leaf_list = taxonomy_tree.nodes_at_level(
        taxonomy_tree.leaf_level
    )
    new_leaf_list.sort()
    new_cluster_to_idx = {
        leaf: ii
        for ii, leaf in enumerate(new_leaf_list)
    }

    non_numerical_keys = ('metadata',
                          'cluster_to_row',
                          'col_names',
                          'taxonomy_tree',
                          'n_cells')

    with h5py.File(src_path, 'r') as src:
        with h5py.File(dst_path, 'w') as dst:
            dst.create_dataset(
                'col_names',
                data=src['col_names'][()]
            )
            dst.create_dataset(
                'cluster_to_row',
                data=json.dumps(
                    new_cluster_to_idx
                ).encode('utf-8')
            )
            dst.create_dataset(
                'taxonomy_tree',
                data=taxonomy_tree.to_str(
                    indent=None,
                    drop_cells=True
                ).encode('utf-8')
            )
            src_cluster_to_idx = json.loads(
                src['cluster_to_row'][()].decode('utf-8')
            )
            dst_n_cells = dst.create_dataset(
                'n_cells',
                shape=(len(new_cluster_to_idx),),
                dtype=int
            )
            src_n_cells = src['n_cells'][()]
            for leaf in new_leaf_list:
                src_idx = src_cluster_to_idx[leaf]
                dst_idx = new_cluster_to_idx[leaf]
                dst_n_cells[dst_idx] = src_n_cells[src_idx]

            for data_key in src.keys():
                if data_key in non_numerical_keys:
                    continue
                dst_data = dst.create_dataset(
                    data_key,
                    dtype=src[data_key].dtype,
                    shape=(len(new_cluster_to_idx), src[data_key].shape[1]),
                    chunks=True
                )
                src_data = src[data_key][()]
                for leaf in new_leaf_list:
                    src_idx = src_cluster_to_idx[leaf]
                    dst_idx = new_cluster_to_idx[leaf]
                    dst_data[dst_idx, :] = src_data[src_idx, :]
