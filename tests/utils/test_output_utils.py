import pytest

import anndata
import h5py
import itertools
import json
import numpy as np
import pandas as pd
import pathlib

from cell_type_mapper.utils.utils import (
    _clean_up,
    mkstemp_clean)

from cell_type_mapper.utils.output_utils import (
    re_order_blob,
    blob_to_df,
    blob_to_csv,
    precomputed_stats_to_uns,
    uns_to_precomputed_stats)


@pytest.fixture(scope='module')
def tmp_dir_fixture(tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('output'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture(scope='module')
def precomputed_stats_fixture(tmp_dir_fixture):
    """
    Write an HDF5 file that has the same schema as a
    precomputed_stats file. Save it to disk. Return
    the path to the saved file.
    """
    h5_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precomputed_stats_',
        suffix='.h5')

    # the contents of this file will just be nonsense;
    # all we need to test for is that they get serialized
    # and deserialized correctly
    rng = np.random.default_rng(77112)
    with h5py.File(h5_path, 'w') as dst:
        dst.create_dataset(
            'metadata',
            data=json.dumps({'x': 1, 'y': 3}).encode('utf-8'))
        dst.create_dataset(
            'cluster_to_row',
            data=json.dumps({'a': 'bcd', 'b': 5}).encode('utf-8'))
        dst.create_dataset(
            'col_names',
            data=json.dumps(['a', 'b', 'c', 'd']).encode('utf-8'))
        dst.create_dataset(
            'taxonomy_tree',
            data=json.dumps({'x': [[1, 2, 3], [6, 7, 8]],
                             'y': [8,],
                             'z/ab': [4, 7]}).encode('utf-8'))
        dst.create_dataset(
            'n_cells',
            data=np.arange(5))
        for k in ('sum', 'sumsq', 'ge1', 'gt1', 'gt0'):
            dst.create_dataset(
                k,
                data=rng.random((37, 29)))

    return h5_path


@pytest.fixture(scope='module')
def results_fixture():
    results = [
        {'cell_id': 'a',
         'level1': {'assignment': 'alice',
                    'bootstrapping_probability': 0.01234567,
                    'corr': 0.112253,
                    'runners_up': ['a', 'b', 'd']},
         'level3': {'assignment': 'bob',
                    'bootstrapping_probability': 0.2,
                    'corr': 0.4,
                    'runners_up': ['c']},
         'level7': {'assignment': 'cheryl',
                    'bootstrapping_probability': 0.245,
                    'corr': 0.33332}
         },
        {'cell_id': 'b',
         'level1': {'assignment': 'roger',
                    'bootstrapping_probability': 0.11119,
                    'corr': 0.1},
         'level3': {'assignment': 'dodger',
                    'bootstrapping_probability': 0.3,
                    'corr': 0.9,
                    'runners_up': ['a', 'f', 'b', 'b'],
                    'runner_up_probability': [0.5, 0.1, 0.05, 0.05]},
         'level7': {'assignment': 'brooklyn',
                    'bootstrapping_probability': 0.5,
                    'corr': 0.11723}
         }
    ]
    return results


@pytest.mark.parametrize('check_consistency', [True, False])
def test_blob_to_df(results_fixture, check_consistency):

    class DummyTree(object):
        hierarchy = ['level1', 'level3', 'level7']

        @property
        def leaf_level(self):
            return 'level7'

        def label_to_name(self, level, label, name_key='gar'):
            return label

        def level_to_name(self, level_label):
            return level_label

    actual_df = blob_to_df(
        results_blob=results_fixture,
        taxonomy_tree=DummyTree(),
        check_consistency=check_consistency)

    assert len(actual_df) == 2

    expected_columns = set([
        "cell_id",
        "level1_name",
        "level1_label",
        "level1_bootstrapping_probability",
        "level1_corr",
        "level1_runners_up_0",
        "level1_runners_up_1",
        "level1_runners_up_2",
        "level3_name",
        "level3_label",
        "level3_bootstrapping_probability",
        "level3_corr",
        "level3_runners_up_0",
        "level3_runners_up_1",
        "level3_runners_up_2",
        "level3_runners_up_3",
        "level3_runner_up_probability_0",
        "level3_runner_up_probability_1",
        "level3_runner_up_probability_2",
        "level3_runner_up_probability_3",
        "level7_name",
        "level7_label",
        "level7_alias",
        "level7_bootstrapping_probability",
        "level7_corr"])

    if check_consistency:
        expected_columns.add('hierarchy_consistent')

    assert set(actual_df.columns) == expected_columns

    for record in results_fixture:
        sub_df = actual_df[actual_df['cell_id'] == record['cell_id']]
        assert len(sub_df) == 1

        if check_consistency:
            if record['cell_id'] == 'a':
                assert sub_df.hierarchy_consistent.values[0]
            else:
                assert not sub_df.hierarchy_consistent.values[0]

        for level in ('level1', 'level3', 'level7'):
            for k in ('name', 'label'):
                assert (
                    sub_df[f'{level}_{k}'].values
                    == record[level]['assignment']
                )
            if level == 'level7':
                assert (
                    sub_df[f'{level}_alias'].values
                    == record[level]['assignment']
                )
            for k in ('bootstrapping_probability', 'corr'):
                np.testing.assert_allclose(
                    sub_df[f'{level}_{k}'],
                    record[level][k])

            if level == 'level7':
                continue
            if 'runners_up' in record[level]:
                expected_runners_up = record[level]['runners_up']
            else:
                expected_runners_up = []
            for idx in range(len(expected_runners_up)):
                assert (
                    sub_df[f'{level}_runners_up_{idx}'].values
                    == expected_runners_up[idx]
                )

            if level == 'level1' and len(expected_runners_up) < 3:
                for idx in range(len(expected_runners_up), 3):
                    sub_df[f'{level}_runners_up_{idx}'].values is None
            elif level == 'level3' and len(expected_runners_up) < 4:
                for idx in range(len(expected_runners_up), 4):
                    sub_df[f'{level}_runners_up_{idx}'] is None


@pytest.mark.parametrize(
    'with_metadata,check_consistency,rows_at_a_time',
    itertools.product(
        [True, False],
        [True, False],
        [None, 1]
    )
)
def test_blob_to_csv(
        tmp_dir_fixture,
        with_metadata,
        results_fixture,
        check_consistency,
        rows_at_a_time):

    class DummyTree(object):
        hierarchy = ['level1', 'level3', 'level7']

        @property
        def leaf_level(self):
            return 'level7'

        def label_to_name(self, level, label, name_key='gar'):
            return label

        def level_to_name(self, level_label):
            return level_label

    csv_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.csv')

    if with_metadata:
        metadata_path = 'path/to/my/cool_file.txt'
        n_expected = 6
        n_offset = 1
    else:
        metadata_path = None
        n_expected = 5
        n_offset = 0

    blob_to_csv(
        results_blob=results_fixture,
        taxonomy_tree=DummyTree(),
        output_path=csv_path,
        metadata_path=metadata_path,
        check_consistency=check_consistency,
        confidence_key='bootstrapping_probability',
        confidence_label='bootstrapping_probability',
        rows_at_a_time=rows_at_a_time)

    with open(csv_path, 'r') as src:
        actual_lines = src.readlines()
    assert len(actual_lines) == n_expected

    if with_metadata:
        metadata_line = '# metadata = cool_file.txt\n'
        assert actual_lines[0] == metadata_line

    taxonomy_line = '# taxonomy hierarchy = ["level1", "level3", "level7"]\n'
    assert actual_lines[0+n_offset] == taxonomy_line

    if check_consistency:
        header_line = ('cell_id,hierarchy_consistent,level1_label,level1_name,'
                       'level1_bootstrapping_probability,'
                       'level3_label,level3_name,'
                       'level3_bootstrapping_probability,'
                       'level7_label,level7_name,'
                       'level7_alias,level7_bootstrapping_probability\n')
    else:
        header_line = ('cell_id,level1_label,level1_name,'
                       'level1_bootstrapping_probability,'
                       'level3_label,level3_name,'
                       'level3_bootstrapping_probability,'
                       'level7_label,level7_name,level7_alias,'
                       'level7_bootstrapping_probability\n')
    assert actual_lines[2+n_offset] == header_line

    if check_consistency:
        cell0 = (
            'a,True,alice,alice,0.0123,bob,bob,0.2000,'
            'cheryl,cheryl,cheryl,0.2450\n'
        )
    else:
        cell0 = (
            'a,alice,alice,0.0123,bob,bob,0.2000,'
            'cheryl,cheryl,cheryl,0.2450\n'
        )
    assert actual_lines[3+n_offset] == cell0

    if check_consistency:
        cell1 = (
            'b,False,roger,roger,0.1112,dodger,dodger,0.3000,'
            'brooklyn,brooklyn,brooklyn,0.5000\n'
        )
    else:
        cell1 = (
            'b,roger,roger,0.1112,dodger,dodger,0.3000,'
            'brooklyn,brooklyn,brooklyn,0.5000\n'
        )
    assert actual_lines[4+n_offset] == cell1

    # test again using a different confidence stat

    blob_to_csv(
        results_blob=results_fixture,
        taxonomy_tree=DummyTree(),
        output_path=csv_path,
        metadata_path=metadata_path,
        confidence_key='corr',
        confidence_label='number',
        rows_at_a_time=rows_at_a_time)

    with open(csv_path, 'r') as src:
        actual_lines = src.readlines()
    assert len(actual_lines) == n_expected

    if with_metadata:
        metadata_line = '# metadata = cool_file.txt\n'
        assert actual_lines[0] == metadata_line

    taxonomy_line = '# taxonomy hierarchy = ["level1", "level3", "level7"]\n'
    assert actual_lines[0+n_offset] == taxonomy_line

    header_line = ('cell_id,level1_label,level1_name,level1_number,'
                   'level3_label,level3_name,level3_number,level7_label,'
                   'level7_name,level7_alias,level7_number\n')
    assert actual_lines[2+n_offset] == header_line

    cell0 = 'a,alice,alice,0.1123,bob,bob,0.4000,cheryl,cheryl,cheryl,0.3333\n'
    assert actual_lines[3+n_offset] == cell0

    cell1 = (
        'b,roger,roger,0.1000,dodger,dodger,0.9000,'
        'brooklyn,brooklyn,brooklyn,0.1172\n'
    )
    assert actual_lines[4+n_offset] == cell1


@pytest.mark.parametrize(
    'with_metadata,rows_at_a_time',
    itertools.product(
        [True, False],
        [None, 1]
    )
)
def test_blob_to_csv_with_mapping(
        tmp_dir_fixture,
        with_metadata,
        results_fixture,
        rows_at_a_time):
    """
    Test with a name mapping
    """

    name_mapper = {
        'level1': {
            'alice': {
                'name': 'beverly'
            },
            'roger': {
                'name': 'jane'
            }
        },
        'level3': {
            'bob': {
                'name': 'X'
            },
            'dodger': {
                'name': 'Y'
            }
        },
        'level7': {
            'cheryl': {
                'name': 'tom',
                'alias': '77'
            },
            'brooklyn': {
                'name': 'cleveland',
                'alias': '88'
            }
        }
    }

    class DummyTree(object):
        hierarchy = ['level1', 'level3', 'level7']

        @property
        def leaf_level(self):
            return 'level7'

        def label_to_name(self, level, label, name_key='gar'):
            return name_mapper[level][label][name_key]

        def level_to_name(self, level_label):
            return level_label

    csv_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.csv')

    if with_metadata:
        metadata_path = 'path/to/my/cool_file.txt'
        n_expected = 6
        n_offset = 1
    else:
        metadata_path = None
        n_expected = 5
        n_offset = 0

    blob_to_csv(
        results_blob=results_fixture,
        taxonomy_tree=DummyTree(),
        output_path=csv_path,
        metadata_path=metadata_path,
        confidence_key='bootstrapping_probability',
        confidence_label='bootstrapping_probability',
        rows_at_a_time=rows_at_a_time)

    with open(csv_path, 'r') as src:
        actual_lines = src.readlines()
    assert len(actual_lines) == n_expected

    if with_metadata:
        metadata_line = '# metadata = cool_file.txt\n'
        assert actual_lines[0] == metadata_line

    taxonomy_line = '# taxonomy hierarchy = ["level1", "level3", "level7"]\n'
    assert actual_lines[0+n_offset] == taxonomy_line

    header_line = ('cell_id,level1_label,level1_name,'
                   'level1_bootstrapping_probability,'
                   'level3_label,level3_name,'
                   'level3_bootstrapping_probability,level7_label,'
                   'level7_name,level7_alias,'
                   'level7_bootstrapping_probability\n')
    assert actual_lines[2+n_offset] == header_line

    cell0 = 'a,alice,beverly,0.0123,bob,X,0.2000,cheryl,tom,77,0.2450\n'
    assert actual_lines[3+n_offset] == cell0

    cell1 = (
        'b,roger,jane,0.1112,dodger,Y,'
        '0.3000,brooklyn,cleveland,88,0.5000\n'
    )
    assert actual_lines[4+n_offset] == cell1

    # test again using a different confidence stat

    blob_to_csv(
        results_blob=results_fixture,
        taxonomy_tree=DummyTree(),
        output_path=csv_path,
        metadata_path=metadata_path,
        confidence_key='corr',
        confidence_label='number')

    with open(csv_path, 'r') as src:
        actual_lines = src.readlines()
    assert len(actual_lines) == n_expected

    if with_metadata:
        metadata_line = '# metadata = cool_file.txt\n'
        assert actual_lines[0] == metadata_line

    taxonomy_line = '# taxonomy hierarchy = ["level1", "level3", "level7"]\n'
    assert actual_lines[0+n_offset] == taxonomy_line

    header_line = ('cell_id,level1_label,level1_name,level1_number,'
                   'level3_label,level3_name,level3_number,level7_label,'
                   'level7_name,level7_alias,level7_number\n')
    assert actual_lines[2+n_offset] == header_line

    cell0 = 'a,alice,beverly,0.1123,bob,X,0.4000,cheryl,tom,77,0.3333\n'
    assert actual_lines[3+n_offset] == cell0

    cell1 = (
        'b,roger,jane,0.1000,dodger,Y,'
        '0.9000,brooklyn,cleveland,88,0.1172\n'
    )
    assert actual_lines[4+n_offset] == cell1


@pytest.mark.parametrize('with_metadata', [True, False])
def test_blob_to_csv_level_map(
        tmp_dir_fixture,
        with_metadata,
        results_fixture):

    class DummyTree(object):
        hierarchy = ['level1', 'level3', 'level7']

        @property
        def leaf_level(self):
            return 'level7'

        def label_to_name(self, level, label, name_key='gar'):
            return label

        def level_to_name(self, level_label):
            n = level_label.replace('level', '')
            n = int(n)**2
            return f'salted_{n}'

    csv_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.csv')

    if with_metadata:
        metadata_path = 'path/to/my/cool_file.txt'
        n_expected = 7
        n_offset = 1
    else:
        metadata_path = None
        n_expected = 6
        n_offset = 0

    blob_to_csv(
        results_blob=results_fixture,
        taxonomy_tree=DummyTree(),
        output_path=csv_path,
        confidence_key='bootstrapping_probability',
        confidence_label='bootstrapping_probability',
        metadata_path=metadata_path)

    with open(csv_path, 'r') as src:
        actual_lines = src.readlines()
    assert len(actual_lines) == n_expected

    if with_metadata:
        metadata_line = '# metadata = cool_file.txt\n'
        assert actual_lines[0] == metadata_line

    taxonomy_line = '# taxonomy hierarchy = ["level1", "level3", "level7"]\n'
    assert actual_lines[0+n_offset] == taxonomy_line

    readable_taxonomy_line = ('# readable taxonomy hierarchy = '
                              '["salted_1", '
                              '"salted_9", '
                              '"salted_49"]\n')
    assert actual_lines[1+n_offset] == readable_taxonomy_line

    header_line = ('cell_id,'
                   'salted_1_label,salted_1_name,'
                   'salted_1_bootstrapping_probability,'
                   'salted_9_label,salted_9_name,'
                   'salted_9_bootstrapping_probability,'
                   'salted_49_label,salted_49_name,salted_49_alias,'
                   'salted_49_bootstrapping_probability\n')
    # version line is at 2+n_offset
    assert actual_lines[3+n_offset] == header_line


def test_re_order_blob(tmp_dir_fixture):
    rng = np.random.default_rng(221312)
    cell_id_list = [f'c_{ii}' for ii in range(11)]
    obs = pd.DataFrame(
        [{'cell_id': c} for c in cell_id_list]).set_index('cell_id')
    a_data = anndata.AnnData(obs=obs)
    h5ad_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.h5ad')
    a_data.write_h5ad(h5ad_path)

    rng.shuffle(cell_id_list)
    results_blob = [
        {'cell_id': c, 'data': ii**2}
        for ii, c in enumerate(cell_id_list)
    ]
    results_lookup = {c['cell_id']: c for c in results_blob}

    new_blob = re_order_blob(
        results_blob=results_blob,
        query_path=h5ad_path)

    assert not (
        list([c['cell_id'] for c in results_blob])
        == list(obs.index.values)
    )

    assert (
        list([c['cell_id'] for c in new_blob])
        == list(obs.index.values)
    )

    for c in new_blob:
        assert c == results_lookup[c['cell_id']]

    assert len(new_blob) == len(results_lookup)


def test_precomputed_stats_to_uns(
        tmp_dir_fixture,
        precomputed_stats_fixture):
    """
    Test utility functions to move precomputed stats data from HDF5 to
    the uns element of an h5ad file and back.
    """
    rng = np.random.default_rng(812331)
    n_cells = 15
    n_genes = 21
    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='uns_anndata_',
        suffix='.h5ad')

    var = pd.DataFrame(
        [{'gene': f'g_{ii}'}
         for ii in range(n_genes)]
    ).set_index('gene')

    obs = pd.DataFrame(
        [{'cell': f'c_{ii}'}
         for ii in range(n_cells)]
    ).set_index('cell')

    original_uns = {
        'garbage': 'yes',
        'maybe': [1, 2, 3, 4]
    }

    a_data = anndata.AnnData(
        X=rng.random((n_cells, n_genes)),
        obs=obs,
        var=var,
        uns=original_uns)

    a_data.write_h5ad(h5ad_path)

    uns_key = 'serialization_test'

    precomputed_stats_to_uns(
        precomputed_stats_path=precomputed_stats_fixture,
        h5ad_path=h5ad_path,
        uns_key=uns_key)

    roundtrip_path = uns_to_precomputed_stats(
        uns_key=uns_key,
        h5ad_path=h5ad_path,
        tmp_dir=tmp_dir_fixture)

    with h5py.File(precomputed_stats_fixture, 'r') as expected_src:
        with h5py.File(roundtrip_path, 'r') as actual_src:
            assert set(expected_src.keys()) == set(actual_src.keys())
            for k in expected_src.keys():
                expected = expected_src[k][()]
                actual = actual_src[k][()]
                if not isinstance(expected, np.ndarray):
                    assert expected == actual
                else:
                    np.testing.assert_allclose(
                        expected,
                        actual,
                        atol=0.0,
                        rtol=1.0e-6)

    # make sure that original uns is unspoiled
    roundtrip_h5ad = anndata.read_h5ad(h5ad_path, backed='r')
    assert roundtrip_h5ad.uns['garbage'] == original_uns['garbage']
    np.testing.assert_array_equal(
        original_uns['maybe'],
        roundtrip_h5ad.uns['maybe']
    )
