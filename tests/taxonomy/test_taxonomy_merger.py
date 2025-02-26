"""
Test the merging of taxonomies with different numbers of levels
"""
from cell_type_mapper.taxonomy.taxonomy_tree import TaxonomyTree
from cell_type_mapper.taxonomy.merger import merge_taxonomy_trees


def test_tree_merger():

    dataA = {
        'hierarchy': ['class', 'subclass', 'cluster'],
        'class': {'A': ['C'], 'B': ['D', 'E']},
        'subclass': {'C': ['F', 'G'],
                     'D': ['H'],
                     'E': ['I', 'J']},
        'cluster': {k: [f'c_{ii}'] for ii, k in enumerate('FGHIJ')}
    }
    treeA = TaxonomyTree(data=dataA)

    dataB = {
        'hierarchy': ['cluster', 'subclass'],
        'cluster': {'A': ['B'], 'c': ['d', 'e']},
        'subclass': {
            'B': ['x', 'y'],
            'd': ['z'],
            'e': ['u', 'v']
        }
    }
    treeB = TaxonomyTree(data=dataB)

    dataC = {
        'hierarchy': ['subclass', 'supercluster', 'cluster', 'subcluster'],
        'subclass': {'A': ['B'], 'C': ['D']},
        'supercluster': {'B': ['E', 'F'], 'D': ['G']},
        'cluster': {'E': ['H'], 'F': ['i', 'j'], 'G': ['K', 'L']},
        'subcluster': {
            k: [f'{k}_{ii}' for ii in range(3)]
            for k in 'HijKL'
        }
    }
    treeC = TaxonomyTree(data=dataC)

    tree_lookup = {
        'taxonomyA': treeA,
        'taxonomyB': treeB,
        'taxonomyC': treeC
    }

    merged_tree, name_map = merge_taxonomy_trees(
        tree_lookup=tree_lookup
    )

    for taxonomy_name in tree_lookup:
        base_tree = tree_lookup[taxonomy_name]

        # make sure leaves were all assigned to leaf level
        # of new tree
        for leaf in base_tree.nodes_at_level(base_tree.leaf_level):
            new_level = (
                name_map[taxonomy_name][base_tree.leaf_level][leaf]['level']
            )

            new_name = (
                name_map[taxonomy_name][base_tree.leaf_level][leaf]['node']
            )

            assert new_level == merged_tree.leaf_level

            assert (
                new_name in merged_tree.nodes_at_level(merged_tree.leaf_level)
            )

        # verify consistent parentage
        for base_level, merged_level in zip(base_tree.hierarchy[-1::-1],
                                            merged_tree.hierarchy[-1::-1]):

            for node in base_tree.nodes_at_level(base_level):

                base_parents = base_tree.parents(level=base_level, node=node)
                new_node = name_map[taxonomy_name][base_level][node]

                assert new_node['level'] == merged_level

                merged_parents = merged_tree.parents(
                    level=new_node['level'],
                    node=new_node['node']
                )
                assert (
                    merged_parents[merged_tree.hierarchy[0]] == taxonomy_name
                )

                for parent_level in base_parents:
                    expected = (
                        name_map[taxonomy_name][parent_level][
                                                    base_parents[parent_level]]
                    )

                    assert (
                        merged_parents[expected['level']] == expected['node']
                    )
