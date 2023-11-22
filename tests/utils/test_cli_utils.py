import pytest

import anndata
import numpy as np
import pandas as pd

from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.utils.cli_utils import _get_query_gene_names


@pytest.mark.parametrize('as_ensembl', [True, False])
def test_get_query_gene_names(tmp_dir_fixture, as_ensembl):

    input_names = [
        'Xkr4',
        'Rrs1',
        'bob',
        'NCBIGene:73261']

    expected_ensembl = [
        'ENSMUSG00000051951',
        'ENSMUSG00000061024',
        None,
        'ENSMUSG00000005983']

    var = pd.DataFrame(
        [{'gene_id': n} for n in input_names]).set_index('gene_id')

    obs = pd.DataFrame([{'cell_id': f'cell_{ii}'}
                        for ii in range(10)]).set_index('cell_id')

    a_data = anndata.AnnData(
        X=np.zeros((len(obs), len(var)), dtype=np.float32),
        obs=obs,
        var=var,
        dtype=np.float32)

    src_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='var_names_',
        suffix='.h5ad')

    a_data.write_h5ad(src_path)

    actual = _get_query_gene_names(src_path, map_to_ensembl=as_ensembl)
    if as_ensembl:
        for idx in (0, 1, 3):
            assert actual[0][idx] == expected_ensembl[idx]
        assert 'unmapped' in actual[0][2]
        assert len(actual[0]) == 4
        assert actual[1] == 1
    else:
        assert actual == (input_names, 0)
