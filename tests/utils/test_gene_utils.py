import pytest

from cell_type_mapper.cli.cli_log import (
    CommandLog
)

from cell_type_mapper.utils.gene_utils import(
     mask_duplicate_gene_identifiers,
     DuplicateGeneIDWarning
)


@pytest.mark.parametrize("use_log", [True, False])
def test_mask_gene_id(use_log):
    if use_log:
        log = CommandLog()
    else:
        log = None

    prefix = 'DUMMY'
    gene_id_list = ['a', 'b', 'c', 'd']
    actual = mask_duplicate_gene_identifiers(
        gene_identifier_list=gene_id_list,
        mask_prefix=prefix,
        log=log
    )
    assert actual == gene_id_list

    with_dup = ['a', 'b', 'a', 'c', 'd', 'c', 'c', 'e']
    expected = [
        'DUMMY_0_0', 'b', 'DUMMY_0_1', 'DUMMY_1_0',
        'd', 'DUMMY_1_1', 'DUMMY_1_2', 'e'
    ]

    msg = "The following gene identifiers occurred more than once"
    if use_log:
        flavor = UserWarning
    else:
        flavor = DuplicateGeneIDWarning

    with pytest.warns(flavor, match=msg):
        actual = mask_duplicate_gene_identifiers(
            gene_identifier_list=with_dup,
            mask_prefix=prefix,
            log=log)

    assert actual == expected
