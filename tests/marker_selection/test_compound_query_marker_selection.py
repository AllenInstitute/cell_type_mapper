"""
Test the utility functions for selecting query markers
from several reference markers, which point to different
precomputed stats files.
"""

import pytest

import h5py
import itertools
import json
import numpy as np
import pathlib
import tempfile
import warnings

from cell_type_mapper.test_utils.reference_markers import (
    move_precomputed_stats_from_reference_markers
)

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.cli.reference_markers import (
    ReferenceMarkerRunner)

from cell_type_mapper.type_assignment.marker_cache_v2 import (
    create_raw_marker_gene_lookup)

from cell_type_mapper.type_assignment.marker_cache_v2 import (
    create_marker_gene_lookup_from_ref_list)


@pytest.fixture(scope='module')
def tmp_dir_fixture(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp('compound_query_')
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture(scope='module')
def taxonomy_tree_fixture():
    data = {
        'hierarchy': ['class', 'subclass', 'cluster'],
        'class': {
            'classA': ['subclassA', 'subclassB'],
            'classB': ['subclassC', 'subclassD']
        },
        'subclass': {
            'subclassA': ['c0', 'c1'],
            'subclassB': ['c2', 'c3'],
            'subclassC': ['c4', 'c5'],
            'subclassD': ['c6', 'c7']
        },
        'cluster': {
            f'c{ii}': [] for ii in range(8)
        }
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return TaxonomyTree(data=data)


@pytest.fixture(scope='module')
def precompute_path_to_n_cluster(
        tmp_dir_fixture):

    pth0 = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precomputedAA_.AAA',
        suffix='.0.h5')

    pth1 = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precomputedBB_.BBB',
        suffix='.1.h5')

    pth2 = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precomputedCC_.CCC',
        suffix='.2.h5')

    result = {
        'c0': {
            pth0: 100,
            pth1: 10,
            pth2: 3
        },
        'c1': {
            pth0: 110,
            pth1: 15,
            pth2: 4
        },
        'c2': {
            pth0: 20,
            pth1: 475,
            pth2: 5
        },
        'c3': {
            pth0: 35,
            pth1: 588,
            pth2: 4
        },
        'c4': {
            pth0: 24,
            pth1: 18,
            pth2: 9
        },
        'c5': {
            pth0: 25,
            pth1: 69,
            pth2: 32
        },
        'c6': {
            pth0: 245,
            pth1: 5,
            pth2: 131
        },
        'c7': {
            pth0: 6,
            pth1: 66,
            pth2: 121
        }
    }
    return result


@pytest.fixture(scope='module')
def gene_fixture():
    return [f'g_{ii}' for ii in range(48)]


@pytest.fixture(scope='module')
def precomputed_stats_files(
        precompute_path_to_n_cluster,
        gene_fixture,
        taxonomy_tree_fixture):

    rng = np.random.default_rng(221312)

    cluster_to_row = {
        f'c{ii}': ii for ii in range(8)
    }

    n_clusters = len(cluster_to_row)
    n_genes = len(gene_fixture)

    path_list = list(precompute_path_to_n_cluster['c0'].keys())
    path_list.sort()

    invalid_genes = np.zeros(n_genes, dtype=bool)

    for i_path, path in enumerate(path_list):

        invalid_genes[:] = False
        invalid_genes[i_path::3] = True
        invalid_genes = np.logical_not(invalid_genes)

        sum_arr = 2.0+10.0*rng.random((n_clusters, n_genes))
        sum_arr[:, invalid_genes] = 0.0
        sumsq_arr = (2.0*(rng.random((n_clusters, n_genes))+1.0))*sum_arr
        ge1_arr = np.zeros((n_clusters, n_genes), dtype=int)
        n_cells = np.zeros(len(cluster_to_row), dtype=int)
        for cc in cluster_to_row:
            idx = cluster_to_row[cc]
            n_cells[idx] = precompute_path_to_n_cluster[cc][path]
            ge1_arr[idx, :] = rng.integers(2, n_cells[idx], n_genes)
            ge1_arr[idx, invalid_genes] = 0

        with h5py.File(path, 'w') as dst:
            dst.create_dataset(
                'taxonomy_tree',
                data=taxonomy_tree_fixture.to_str().encode('utf-8'))
            dst.create_dataset(
                'n_cells', data=n_cells)
            dst.create_dataset(
                'col_names',
                data=json.dumps(gene_fixture).encode('utf-8')
            )
            dst.create_dataset(
                'cluster_to_row',
                data=json.dumps(cluster_to_row).encode('utf-8'))
            dst.create_dataset(
                'sum', data=sum_arr)
            dst.create_dataset(
                'sumsq', data=sumsq_arr)
            dst.create_dataset(
                'ge1', data=ge1_arr)
    return path_list


@pytest.fixture(scope='module')
def reference_marker_fixture(
        precomputed_stats_files,
        taxonomy_tree_fixture,
        gene_fixture,
        tmp_dir_fixture):

    output_dir = tempfile.mkdtemp(dir=tmp_dir_fixture)

    path_list = []
    for precompute_path in precomputed_stats_files:
        precompute_name = pathlib.Path(precompute_path).name
        old_str = precompute_name.split('.')[0]
        new_str = precompute_name.replace(old_str, 'reference_markers', 1)
        expected_path = f'{output_dir}/{new_str}'

        # p_th is set to a nonsense value below
        # because we are actually testing that
        # markers get pulled from the correct
        # reference file based on the number of
        # cells represented by each
        config = {
            'precomputed_path_list': [precompute_path],
            'output_dir': output_dir,
            'clobber': True,
            'drop_level': None,
            'tmp_dir': str(tmp_dir_fixture),
            'n_processors': 1,
            'exact_penetrance': False,
            'p_th': 1.5,
            'q1_th': 0.5,
            'q1_min_th': 0.0,
            'qdiff_th': 0.7,
            'qdiff_min_th': 0.0,
            'log2_fold_th': 0.5,
            'log2_fold_min_th': 0.0,
            'n_valid': 8
        }

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            runner = ReferenceMarkerRunner(
                args=[], input_data=config)
            runner.run()

        path_list.append(expected_path)

        with h5py.File(expected_path, 'r') as src:
            diff = np.diff(
                src['sparse_by_pair/up_pair_idx'][()])
            diff += np.diff(
                src['sparse_by_pair/down_pair_idx'][()])
            assert diff.max() > 0

    return path_list


@pytest.fixture(scope='module')
def expected_query_marker_fixture(
        reference_marker_fixture,
        gene_fixture,
        taxonomy_tree_fixture,
        tmp_dir_fixture):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        result = dict()
        for ref_path in reference_marker_fixture:
            lookup = create_raw_marker_gene_lookup(
                input_cache_path=ref_path,
                query_gene_names=gene_fixture,
                taxonomy_tree=taxonomy_tree_fixture,
                n_per_utility=3,
                tmp_dir=tmp_dir_fixture,
                n_processors=3)
            lookup.pop('log')
            result[ref_path] = lookup

    # make sure different reference files give different
    # sets of query markers
    for pair in itertools.combinations(reference_marker_fixture, 2):
        l0 = result[pair[0]]
        l1 = result[pair[1]]
        key_list = set(l0.keys()).union(set(l1.keys()))
        ct_compare = 0
        for k in key_list:
            if k not in l0 or k not in l1:
                continue
            if len(l0[k]) == 0 and len(l1[k]) == 0:
                continue
            assert set(l0[k]) != set(l1[k])
            ct_compare += 1
        assert ct_compare > 0

    return result


def test_query_infrastructure(
        reference_marker_fixture,
        expected_query_marker_fixture,
        gene_fixture,
        tmp_dir_fixture):
    """
    Make sure that query markers are chosen from the correct
    dataset based on how many cells are in each leaf node.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        actual = create_marker_gene_lookup_from_ref_list(
            reference_marker_path_list=reference_marker_fixture,
            query_gene_names=gene_fixture,
            n_per_utility=3,
            n_per_utility_override=None,
            n_processors=3,
            behemoth_cutoff=1000,
            tmp_dir=tmp_dir_fixture)

    pth0 = reference_marker_fixture[0]
    pth1 = reference_marker_fixture[1]
    pth2 = reference_marker_fixture[2]
    expected = expected_query_marker_fixture

    # these expected values are based on the number cells in
    # the n_cells array of the precomputed stats files
    assert set(actual['class/classA']) == set(expected[pth1]['class/classA'])
    assert set(actual['class/classB']) == set(expected[pth0]['class/classB'])
    assert set(actual['subclass/subclassA']) == set(
        expected[pth0]['subclass/subclassA'])
    assert set(actual['subclass/subclassB']) == set(
        expected[pth1]['subclass/subclassB'])
    assert set(actual['subclass/subclassC']) == set(
        expected[pth1]['subclass/subclassC'])
    assert set(actual['subclass/subclassD']) == set(
        expected[pth2]['subclass/subclassD'])


@pytest.mark.parametrize('search_for_stats_file', (True, False))
def test_misplaced_stats_infrastructure(
        reference_marker_fixture,
        expected_query_marker_fixture,
        precomputed_stats_files,
        gene_fixture,
        tmp_dir_fixture,
        search_for_stats_file):
    """
    Make sure that the query marker finder works, even when the
    precomputed stats file is not in the expected place (as long as
    search_for_stats is set to True)
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        tmp_dir = pathlib.Path(
            tempfile.mkdtemp(
                dir=tmp_dir_fixture
            )
        )

        new_ref_marker_list = move_precomputed_stats_from_reference_markers(
            reference_marker_path_list=reference_marker_fixture,
            tmp_dir=tmp_dir)

        if not search_for_stats_file:
            match = "search_for_stats_file=True"
            with pytest.raises(FileNotFoundError, match=match):
                actual = create_marker_gene_lookup_from_ref_list(
                    reference_marker_path_list=new_ref_marker_list,
                    query_gene_names=gene_fixture,
                    n_per_utility=3,
                    n_per_utility_override=None,
                    n_processors=3,
                    behemoth_cutoff=1000,
                    tmp_dir=tmp_dir_fixture,
                    search_for_stats_file=search_for_stats_file)

        else:
            actual = create_marker_gene_lookup_from_ref_list(
                reference_marker_path_list=new_ref_marker_list,
                query_gene_names=gene_fixture,
                n_per_utility=3,
                n_per_utility_override=None,
                n_processors=3,
                behemoth_cutoff=1000,
                tmp_dir=tmp_dir_fixture,
                search_for_stats_file=search_for_stats_file)

            pth0 = reference_marker_fixture[0]
            pth1 = reference_marker_fixture[1]
            pth2 = reference_marker_fixture[2]
            expected = expected_query_marker_fixture

            # these expected values are based on the number cells in
            # the n_cells array of the precomputed stats files
            assert (
                set(actual['class/classA'])
                == set(expected[pth1]['class/classA'])
            )
            assert (
                set(actual['class/classB'])
                == set(expected[pth0]['class/classB'])
            )
            assert set(actual['subclass/subclassA']) == set(
                expected[pth0]['subclass/subclassA'])
            assert set(actual['subclass/subclassB']) == set(
                expected[pth1]['subclass/subclassB'])
            assert set(actual['subclass/subclassC']) == set(
                expected[pth1]['subclass/subclassC'])
            assert set(actual['subclass/subclassD']) == set(
                expected[pth2]['subclass/subclassD'])

    _clean_up(tmp_dir)
