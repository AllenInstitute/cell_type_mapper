import pytest

import anndata
import h5py
import json
import numpy as np
import pandas as pd
import warnings

from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.utils.cli_utils import (
     align_query_gene_names
)

import cell_type_mapper.test_utils.gene_mapping.mappers as legacy_mappers


@pytest.fixture(scope='module')
def precomputed_stats_fixture(tmp_dir_fixture):
    """
    Return dict. Keys are 'mouse' and 'human'. Values will be
    paths to hdf5 files whose col_names point to lists of ENSEMBL IDs
    from the appropriate species
    """
    result = dict()
    for species in ('mouse', 'human'):
        file_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix=f'precomputed_stats_for_gene_alignment_{species}_',
            suffix='.h5'
        )
        if species == 'mouse':
            src = legacy_mappers.get_mouse_gene_id_mapping()
        elif species == 'human':
            src = legacy_mappers.get_human_gene_id_mapping()
        else:
            raise RuntimeError(f"cannot handle species '{species}'")
        values = []
        for ii, val in enumerate(src.values()):
            if ii > 10:
                break
            values.append(val)

        print(f'values {values}')
        with h5py.File(file_path, 'w') as dst:
            dst = dst.create_dataset(
                'col_names',
                data=json.dumps(values).encode('utf-8')
            )

        result[species] = file_path
    return result


@pytest.mark.parametrize(
    'as_ensembl, species, ensembl_version',
    [(True, 'human', False),
     (False, 'human', False),
     (True, 'mouse', False),
     (False, 'mouse', False),
     (True, 'nonsense', False),
     (False, 'nonsense', False),
     (True, 'mouse', True)
     ]
)
def test_align_query_gene_names(
        tmp_dir_fixture,
        as_ensembl,
        species,
        ensembl_version,
        precomputed_stats_fixture,
        legacy_gene_mapper_db_path_fixture):
    """
    as_ensembl toggles whether or not we are mapping the genes
    to ENSEMBL IDs (false means we perform no mapping by passing
    gene_maper_db=precomputed_stats=None)

    species is the species we are mapping to

    if ensembl_version == True, pass in ENSEMBL IDs with versions
    appended to them
    """

    if species == 'mouse':
        precomputed_stats_path = precomputed_stats_fixture[species]
    else:
        precomputed_stats_path = precomputed_stats_fixture['human']

    db_path = legacy_gene_mapper_db_path_fixture
    if not as_ensembl:
        precomputed_stats_path = None
        db_path = None

    if species == 'mouse':
        if ensembl_version:
            input_names = [
                'ENSMUSG00000051951.123',
                'ENSMUSG00000061024.456',
                'silly_gene',
                'ENSMUSG00000005983.789'
            ]
        else:
            input_names = [
                'Xkr4',
                'Rrs1',
                'bob',
                'NCBIGene:73261'
            ]

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

        actual = align_query_gene_names(
            src_path,
            precomputed_stats_path=precomputed_stats_path,
            gene_mapper_db_path=db_path)

        if as_ensembl:
            if species == 'nonsense':
                for gene in actual[0]:
                    assert 'UNMAPPABLE' in gene
                assert actual[1] == 4
                assert not actual[2]
            else:
                for idx in (0, 1, 3):
                    assert actual[0][idx] == expected_ensembl[idx]
                assert 'UNMAPPABLE' in actual[0][2]
                assert actual[1] == 1
                assert actual[2]
            assert len(actual[0]) == 4
            expected_mapping = {
                g0: g1
                for g0, g1 in zip(input_names, actual[0])
            }

            # check that metadata dict records the mapping
            assert actual[3]['mapping'] == expected_mapping
            assert 'provenance' in actual[3]

        else:
            assert actual[0] == input_names
            assert actual[1] == 0
            assert not actual[2]
            assert actual[3] == dict()


def test_flag_in_align_query_gene_names(
        tmp_dir_fixture,
        precomputed_stats_fixture,
        legacy_gene_mapper_db_path_fixture):
    """
    Test if the boolean flag at the end of align_query_gene_names' output
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

            result = align_query_gene_names(
                query_gene_path=src_path,
                precomputed_stats_path=precomputed_stats_fixture['mouse'],
                gene_mapper_db_path=legacy_gene_mapper_db_path_fixture)

        if expected:
            assert result[2]
        else:
            assert not result[2]


def test_no_species_gene_alignment(
        tmp_dir_fixture,
        legacy_gene_mapper_db_path_fixture):
    """
    Test that, when no species could be found for the
    reference data, the input genes remain unchanged.
    """
    precomputed_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='nonsense_species',
        suffix='.h5'
    )
    gene_names = [
        'not',
        'real',
        'genes'
    ]
    with h5py.File(precomputed_path, 'w') as dst:
        dst.create_dataset(
            'col_names',
            data=json.dumps(gene_names).encode('utf-8')
        )

    input_names = [
        'Xkr4',
        'Rrs1',
        'bob',
        'NCBIGene:73261']

    var = pd.DataFrame(
        {'gene': g} for g in input_names
    ).set_index('gene')
    adata_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')
    adata = anndata.AnnData(var=var)
    adata.write_h5ad(adata_path)

    actual = align_query_gene_names(
        query_gene_path=adata_path,
        gene_id_col=None,
        precomputed_stats_path=precomputed_path,
        gene_mapper_db_path=legacy_gene_mapper_db_path_fixture,
        log=None)

    assert actual[0] == input_names
    assert actual[1] == 0
    assert not actual[2]
    assert actual[3] == dict()
