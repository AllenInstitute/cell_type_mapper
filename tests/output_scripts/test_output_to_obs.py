import pytest

import anndata
import copy
import json
import numpy as np
import os
import pandas as pd

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.cli.transcribe_to_obs import (
    TranscribeToObsRunner)

from cell_type_mapper.utils.output_utils import (
    blob_to_df)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)


@pytest.fixture(scope='module')
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp(
        'output_formatting_')
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture(scope='module')
def cell_id_fixture():
    return [f'cell_{ii}' for ii in range(17)]

@pytest.fixture(scope='module')
def taxonomy_data_fixture():
    result = {
        'hierarchy': ['class', 'subclass', 'cluster'],
        'class': {
            'A': ['a', 'c'],
            'B': ['b'],
            'C': ['d', 'e']
        },
        'subclass': {
            'a': ['1'],
            'b': ['2', '4'],
            'c': ['3', '5'],
            'd': ['6'],
            'e': ['7']
        },
        'cluster': {
             str(l): [] for l in range(1, 8, 1)
        }
    }
    return result



@pytest.fixture(scope='module')
def output_json_data_fixture(
         cell_id_fixture,
         taxonomy_data_fixture):

    cell_id_list = copy.deepcopy(cell_id_fixture)
    rng = np.random.default_rng(8712311)
    rng.shuffle(cell_id_list)
    data = []
    for cell_id in cell_id_list:
        this = {'cell_id': cell_id}
        for level in taxonomy_data_fixture:
            if level == 'hierarchy':
                continue
            this_level = dict()
            this_level['assignment'] = rng.choice(
                list(taxonomy_data_fixture[level].keys()))
            this_level['probability'] = float(rng.random())
            this[level] = this_level
        data.append(this)
    return data


@pytest.fixture(scope='module')
def output_json_fixture(
         tmp_dir_fixture,
         output_json_data_fixture,
         taxonomy_data_fixture):

    json_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='extended_output_',
        suffix='.json')

    result = {'results': output_json_data_fixture,
              'taxonomy_tree': taxonomy_data_fixture}

    with open(json_path, 'w') as dst:
        dst.write(json.dumps(result, indent=2))
    return json_path


@pytest.fixture(scope='module')
def expected_mapping_df_fixture(
        output_json_data_fixture,
        taxonomy_data_fixture):

    taxonomy_tree = TaxonomyTree(data=taxonomy_data_fixture)
    df = blob_to_df(
        results_blob=output_json_data_fixture,
        taxonomy_tree=taxonomy_tree)

    col_name_list = list(df.columns)
    for col in col_name_list:
        if col == 'cell_id':
            continue
        new_col = f'CDM_{col}'
        df[new_col] = df[col]
        df.drop(col, axis=1, inplace=True)

    return df


@pytest.fixture(scope='module')
def h5ad_fixture(
        tmp_dir_fixture,
        cell_id_fixture):

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='src_data_',
        suffix='.h5ad')

    rng = np.random.default_rng(712321)

    n_cells = len(cell_id_fixture)
    n_genes = 13

    obs_data = [
        {'cell_id': c, 'junk': int(rng.integers(8, 17))}
        for c in cell_id_fixture
    ]
    obs = pd.DataFrame(obs_data).set_index('cell_id')

    var_data= [
        {'gene_id': f'gene_{ii}', 'silly': int(rng.integers(2, 104))}
        for ii in range(n_genes)
    ]
    var = pd.DataFrame(var_data).set_index('gene_id')

    X = rng.random((n_cells, n_genes))
    obsm = {'a': rng.random((n_cells, 2)),
            'b': rng.random(n_cells)}
    varm = {'y': rng.random((n_genes, 5))}
    uns = {'foo': 'bar', 'baz': 77}

    src = anndata.AnnData(
        X=X,
        obs=obs,
        var=var,
        obsm=obsm,
        varm=varm,
        uns=uns)

    src.write_h5ad(h5ad_path)
    return h5ad_path


def test_transcription_to_obs(
        output_json_fixture,
        h5ad_fixture,
        expected_mapping_df_fixture,
        tmp_dir_fixture):

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='merged_',
        suffix='.h5ad')

    os.unlink(output_path)

    config = {
        'result_path': output_json_fixture,
        'h5ad_path': h5ad_fixture,
        'new_h5ad_path': output_path}

    runner = TranscribeToObsRunner(
        args=[],
        input_data=config)

    runner.run()

    baseline = anndata.read_h5ad(h5ad_fixture, backed='r')
    test = anndata.read_h5ad(output_path, backed='r')

    np.testing.assert_allclose(
        baseline.X[()],
        test.X[()],
        atol=0.0,
        rtol=1.0e-6)

    assert set(baseline.obsm.keys()) == set(test.obsm.keys())
    for k in baseline.obsm:
        np.testing.assert_allclose(
            test.obsm[k],
            baseline.obsm[k],
            atol=0.0,
            rtol=1.0e-6)

    assert set(baseline.varm.keys()) == set(test.varm.keys())
    for k in baseline.varm:
        np.testing.assert_allclose(
            test.varm[k],
            baseline.varm[k],
            atol=0.0,
            rtol=1.0e-6)

    for k in baseline.uns:
        assert k in test.uns
        assert baseline.uns[k] == test.uns[k]

    pd.testing.assert_frame_equal(baseline.var, test.var)

    baseline_obs = baseline.obs
    test_obs = test.obs
    assert set(baseline_obs.index.values) == set(test_obs.index.values)
    assert len(test_obs.columns) > len(baseline_obs.columns)
    test_subset = test_obs[baseline_obs.columns]
    pd.testing.assert_frame_equal(
        baseline_obs,
        test_subset)

    new_col = [c for c in test_obs.columns if c not in baseline_obs.columns]

    test_mapping = test_obs[new_col]
    expected = expected_mapping_df_fixture.set_index('cell_id')
    expected = expected.loc[test_mapping.index.values]
    pd.testing.assert_frame_equal(
        test_mapping,
        expected)
