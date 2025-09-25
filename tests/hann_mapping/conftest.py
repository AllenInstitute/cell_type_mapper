import pytest
import warnings

import cell_type_mapper.taxonomy.taxonomy_tree as tree_module


@pytest.fixture
def tree_fixture():
    tree_data = {
        'hierarchy': ['class', 'subclass', 'cluster'],
        'class': {'A': ['a', 'b'], 'B': ['c']},
        'subclass': {'a': ['a1'],
                     'b': ['b1', 'b2'],
                     'c': ['c1', 'c2']},
        'cluster': {
            'a1': [],
            'b1': [],
            'b2': [],
            'c1': [],
            'c2': []}
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        taxonomy_tree = tree_module.TaxonomyTree(data=tree_data)
    return taxonomy_tree
