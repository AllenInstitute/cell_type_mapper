"""
Define functions needed to take several TaxonomyTrees and combine
them into one tree, in which each of the input Trees is its own
node at the top level of the new taxonomy.
"""
import pandas as pd
import warnings

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree
)


def merge_taxonomy_trees(tree_lookup):
    """
    Parameters
    ----------
    tree_lookup:
        a dict whose keys are the names of the individual
        taxonomies and whose values are the TaxonomyTrees
        associated with those names.

    Returns
    --------
    merged_tree:
        a TaxonomyTree whose top level is the taxonomies stored
        in tree_lookup. Trees will be filled out with dummy levels
        as necessary to give the merged tree a self-consistent
        number of levels.


    name_map:
        a dict keyed on
            input_taxonomy
            input_taxonomy_level
            input_taxonomy_node
        which maps to the level and label of the nodes in the
        merged_tree
    """
    taxonomy_name_list = sorted(tree_lookup.keys())
    name_map = dict()

    n_levels = max(
        [len(tree.hierarchy) for tree in tree_lookup.values()]
    ) + 1

    output_hierarchy = [
        f'level_{ii}' for ii in range(n_levels)
    ]

    # create a fake cell_metadata dataframe from which to instantiate
    # the merge cell_type_taxonomy
    cell_ct = 0
    dummy_ct = 0
    cell_records = []
    used_taxons = set()
    for taxonomy_name in taxonomy_name_list:
        tree = tree_lookup[taxonomy_name]

        level_map = {
            ilevel: olevel
            for ilevel, olevel in zip(tree.hierarchy[-1::-1],
                                      output_hierarchy[-1::-1])
        }

        taxon_map = dict()
        for level in tree.hierarchy:
            taxon_map[level] = dict()
            for node in tree.nodes_at_level(level):
                new_taxon = f'{taxonomy_name}:{level}:{node}'
                if new_taxon in used_taxons:
                    raise RuntimeError(
                        f"reusing taxon {new_taxon}"
                    )
                used_taxons.add(new_taxon)
                taxon_map[level][node] = {
                    'level': level_map[level],
                    'node': new_taxon
                }

        name_map[taxonomy_name] = taxon_map

        parent_to_child = dict()

        for leaf in tree.nodes_at_level(tree.leaf_level):
            parentage = tree.parents(
                level=tree.leaf_level,
                node=leaf
            )
            cell = {
                'cell_id': f'fake_cell_{cell_ct}',
                level_map[tree.leaf_level]: (
                    taxon_map[tree.leaf_level][leaf]['node']
                )
            }
            cell_ct += 1
            for level in parentage:
                cell[level_map[level]] = (
                    taxon_map[level][parentage[level]]['node']
                )

            cell[output_hierarchy[0]] = taxonomy_name

            # backfill cell with any placeholder types that are needed
            for parent_level, child_level in zip(output_hierarchy[:-1],
                                                 output_hierarchy[1:]):
                if child_level in cell:
                    break

                parent_node = cell[parent_level]
                if parent_node not in parent_to_child:
                    parent_to_child[parent_node] = (
                        f'{taxonomy_name}:PLACEHOLDER_{dummy_ct}'
                    )
                    dummy_ct += 1
                child_node = parent_to_child[parent_node]
                cell[child_level] = child_node
            cell_records.append(cell)

    cell_df = pd.DataFrame(cell_records).set_index('cell_id')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        output_tree = TaxonomyTree.from_dataframe(
            dataframe=cell_df,
            column_hierarchy=output_hierarchy,
            drop_rows=True
        )

    return output_tree, name_map
