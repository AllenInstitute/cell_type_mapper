import pytest

import anndata
import numpy as np
import pandas as pd

from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.cli.cli_log import (
    CommandLog
)

from cell_type_mapper.utils.gene_utils import (
     mask_duplicate_gene_identifiers,
     get_gene_identifier_list,
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
        'DUMMY_a_0', 'b', 'DUMMY_a_1', 'DUMMY_c_0',
        'd', 'DUMMY_c_1', 'DUMMY_c_2', 'e'
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


@pytest.mark.parametrize('gene_id_col', [None, 'blah'])
def test_get_gene_identifier_list(
        tmp_dir_fixture,
        gene_id_col):

    gene_names = ['albert', 'jackie', 'tallulah', 'fred']
    var = pd.DataFrame(
        [{'idx': ii**2, 'blah': gene_names[ii]}
         for ii in range(len(gene_names))]
    )

    if gene_id_col is None:
        var = var.set_index('blah')
    else:
        var = var.set_index('idx')

    h5ad_path_list = []
    for ii in range(3):
        xx = np.random.random_sample((7, len(gene_names)))
        aa = anndata.AnnData(X=xx, var=var)
        pth = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5ad'
        )
        aa.write_h5ad(pth)
        h5ad_path_list.append(pth)

    actual = get_gene_identifier_list(
        h5ad_path_list=h5ad_path_list,
        gene_id_col=gene_id_col
    )
    assert actual == gene_names

    # test case where the h5ad files do not have the
    # same gene name list
    bad_var = pd.DataFrame(
        [{'idx': ii**2, 'blah': f'g_{ii}'}
         for ii in range(len(gene_names))]
    )

    if gene_id_col is None:
        bad_var = bad_var.set_index('blah')
    else:
        bad_var = bad_var.set_index('idx')

    bad_pth = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad'
    )
    xx = np.random.random_sample((4, len(gene_names)))
    aa = anndata.AnnData(X=xx, var=bad_var)
    aa.write_h5ad(bad_pth)
    h5ad_path_list.append(bad_pth)
    msg = "Inconsistent gene names list"
    with pytest.raises(RuntimeError, match=msg):
        get_gene_identifier_list(
            h5ad_path_list=h5ad_path_list,
            gene_id_col=gene_id_col
        )


@pytest.mark.parametrize('gene_id_col', [None, 'blah'])
def test_get_gene_duplicated_identifier_list(
        tmp_dir_fixture,
        gene_id_col):

    gene_names = [
        'albert',
        'jackie',
        'fred',
        'tallulah',
        'fred',
        'jackie',
        'bob',
        'jackie']

    expected = [
        'albert',
        'INVALID_MARKER_jackie_0',
        'INVALID_MARKER_fred_0',
        'tallulah',
        'INVALID_MARKER_fred_1',
        'INVALID_MARKER_jackie_1',
        'bob',
        'INVALID_MARKER_jackie_2'
    ]

    var = pd.DataFrame(
        [{'idx': ii**2, 'blah': gene_names[ii]}
         for ii in range(len(gene_names))]
    )

    if gene_id_col is None:
        var = var.set_index('blah')
    else:
        var = var.set_index('idx')

    h5ad_path_list = []
    for ii in range(3):
        xx = np.random.random_sample((7, len(gene_names)))
        aa = anndata.AnnData(X=xx, var=var)
        pth = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5ad'
        )
        aa.write_h5ad(pth)
        h5ad_path_list.append(pth)

    msg = "The following gene identifiers occurred more than once"
    with pytest.warns(DuplicateGeneIDWarning, match=msg):
        actual = get_gene_identifier_list(
            h5ad_path_list=h5ad_path_list,
            gene_id_col=gene_id_col
        )

    assert actual == expected
