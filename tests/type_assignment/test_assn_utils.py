import pytest

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.type_assignment.utils import (
    validate_bootstrap_factor_lookup)

from cell_type_mapper.cli.cli_log import (
    CommandLog)


@pytest.mark.parametrize('use_log', [True, False])
def test_validate_bootstrap_lookup(
        tree_fixture,
        use_log):

    if use_log:
        log = CommandLog()
    else:
        log = None

    taxonomy_tree = TaxonomyTree(data=tree_fixture)

    good_lookup = {
        'None': 0.5,
        'class': 1.0,
        'subclass': 1.0e-5
    }

    validate_bootstrap_factor_lookup(
        taxonomy_tree=taxonomy_tree,
        bootstrap_factor_lookup=good_lookup,
        log=log)

    bad_lookup = {
        'None': 0.5,
        'class': 1.0
    }

    msg = "missing level 'subclass'"
    with pytest.raises(RuntimeError, match=msg):
        validate_bootstrap_factor_lookup(
            taxonomy_tree=taxonomy_tree,
            bootstrap_factor_lookup=bad_lookup,
            log=log)

    bad_lookup = {
        'None': 0.5,
        'class': 1.0,
        'subclass': 1.1
    }

    msg = "not >0.0 and <=1.0"
    with pytest.raises(RuntimeError, match=msg):
        validate_bootstrap_factor_lookup(
            taxonomy_tree=taxonomy_tree,
            bootstrap_factor_lookup=bad_lookup,
            log=log)

    bad_lookup = {
        'None': 0.5,
        'class': 1.0,
        'subclass': -0.1
    }

    msg = "not >0.0 and <=1.0"
    with pytest.raises(RuntimeError, match=msg):
        validate_bootstrap_factor_lookup(
            taxonomy_tree=taxonomy_tree,
            bootstrap_factor_lookup=bad_lookup,
            log=log)

    bad_lookup = {
        'None': 0.5,
        'class': 'xyz',
        'subclass': 0.1
    }

    msg = "not a number"
    with pytest.raises(RuntimeError, match=msg):
        validate_bootstrap_factor_lookup(
            taxonomy_tree=taxonomy_tree,
            bootstrap_factor_lookup=bad_lookup,
            log=log)

    msg = "is not dict"
    with pytest.raises(RuntimeError, match=msg):
        validate_bootstrap_factor_lookup(
            taxonomy_tree=taxonomy_tree,
            bootstrap_factor_lookup=None,
            log=log)
