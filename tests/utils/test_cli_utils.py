import pytest

import anndata
import itertools
import numpy as np
import pandas as pd
import warnings

from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.utils.cli_utils import (
     _get_query_gene_names
)


@pytest.mark.parametrize(
    'as_ensembl,species',
    itertools.product(
        [True, False],
        ['human', 'mouse', 'nonsense']
    )
)
def test_get_query_gene_names(tmp_dir_fixture, as_ensembl, species):

    if species == 'mouse':
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
    elif species == 'human':
        input_names = [
            'A1BG',
            'A1CF',
            'alice',
            'A4GALT']
        expected_ensembl = [
            "ENSG00000121410",
            "ENSG00000148584",
            None,
            "ENSG00000128274"
        ]
    elif species == 'nonsense':
        input_names = ['alice', 'bob', 'cheryl', 'dan']
    else:
        raise RuntimeError(
            f"Unclear how to handle species {species}")

    var = pd.DataFrame(
        [{'gene_id': n} for n in input_names]).set_index('gene_id')

    obs = pd.DataFrame([{'cell_id': f'cell_{ii}'}
                        for ii in range(10)]).set_index('cell_id')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        if species == 'nonsense' and as_ensembl:
            with pytest.raises(RuntimeError, match="Could not find a species"):
                _get_query_gene_names(src_path, map_to_ensembl=as_ensembl)
        else:
            actual = _get_query_gene_names(src_path, map_to_ensembl=as_ensembl)
            if as_ensembl:
                for idx in (0, 1, 3):
                    assert actual[0][idx] == expected_ensembl[idx]
                assert 'unmapped' in actual[0][2]
                assert len(actual[0]) == 4
                assert actual[1] == 1
            else:
                assert actual[0] == input_names
                assert actual[1] == 0


def test_flag_in_get_query_gene_names(
        tmp_dir_fixture):
    """
    Test if the boolean flag at the end of get_query_gene_names' output
    is correct
    """

    # case when some genes are mapped to ENSEMBL IDs
    symbol_input_names = [
        'Xkr4',
        'Rrs1',
        'bob',
        'NCBIGene:73261']

    ensembl_input_names = [
        'ENSMUSG00000051951',
        'ENSMUSG00000061024',
        'bob',
        'ENSMUSG00000005983']

    hybrid_input_names = [
        'Xkr4',
        'ENSMUSG00000061024',
        'bob',
        'NCBIGene:73261']

    for input_names, expected in [(symbol_input_names, True),
                                  (ensembl_input_names, False),
                                  (hybrid_input_names, True)]:

        var = pd.DataFrame(
            [{'gene_id': n} for n in input_names]).set_index('gene_id')

        obs = pd.DataFrame([{'cell_id': f'cell_{ii}'}
                            for ii in range(10)]).set_index('cell_id')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

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

            result = _get_query_gene_names(
                query_gene_path=src_path,
                map_to_ensembl=True)

        if expected:
            assert result[2]
        else:
            assert not result[2]
