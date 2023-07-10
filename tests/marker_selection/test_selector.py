import pytest

import copy
from itertools import combinations
import h5py
import json
import numpy as np
import pathlib

from hierarchical_mapping.utils.utils import (
    _clean_up,
    mkstemp_clean)

from hierarchical_mapping.utils.multiprocessing_utils import (
    DummyLock)

from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from hierarchical_mapping.taxonomy.utils import (
    get_all_leaf_pairs)

from hierarchical_mapping.binary_array.binary_array import (
    BinarizedBooleanArray)

from hierarchical_mapping.marker_selection.marker_array import (
    MarkerGeneArray)

from hierarchical_mapping.marker_selection.selection import (
    select_marker_genes_v2)

from hierarchical_mapping.marker_selection.selection_pipeline import (
    _marker_selection_worker,
    select_all_markers)

from hierarchical_mapping.type_assignment.marker_cache_v2 import (
    create_marker_cache_from_reference_markers,
    create_marker_cache_from_specified_markers,
    serialize_markers,
    write_query_markers_to_h5)

from hierarchical_mapping.cli.cli_log import CommandLog


@pytest.fixture(scope='module')
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('selector'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def taxonomy_tree_fixture():

    rng = np.random.default_rng(77123)

    tree = dict()
    tree['hierarchy'] = ['class', 'subclass', 'cluster']
    tree['class']  = {
        'aa': ['a', 'b', 'c', 'd'],
        'bb': ['e'],
        'cc': ['f', 'g', 'h']}

    cluster_list = []
    name_ct = 0
    tree['subclass'] = dict()
    for subclass in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
        n_clusters = rng.integers(3, 8)
        if subclass == 'a':
            n_clusters = 1
        tree['subclass'][subclass] = []
        for ii in range(n_clusters):
            n = f'cluster_{name_ct}'
            cluster_list.append(n)
            tree['subclass'][subclass].append(n)
            name_ct += 1

    tree['cluster'] = dict()
    c0 = 0
    for cluster in cluster_list:
        rows = rng.integers(2, 6)
        tree['cluster'][cluster] = []
        for ii in range(rows):
            tree['cluster'][cluster].append(c0+ii)
        c0 += rows

    return tree

@pytest.fixture
def pair_to_idx_fixture(
        taxonomy_tree_fixture):

    leaf = taxonomy_tree_fixture['hierarchy'][-1]
    cluster_list = list(taxonomy_tree_fixture[leaf].keys())
    cluster_list.sort()
    pair_to_idx = dict()
    pair_to_idx['cluster'] = dict()
    for idx, pair in enumerate(combinations(cluster_list, 2)):
        assert pair[0] < pair[1]
        if pair[0] not in pair_to_idx['cluster']:
            pair_to_idx['cluster'][pair[0]] = dict()
        pair_to_idx['cluster'][pair[0]][pair[1]] = idx
    pair_to_idx['n_pairs'] = idx + 1
    return pair_to_idx


@pytest.fixture
def gene_names_fixture():
    n_genes = 47
    gene_names = [f'g_{ii}' for ii in range(n_genes)]
    return gene_names


@pytest.fixture
def is_marker_fixture(
        pair_to_idx_fixture,
        gene_names_fixture):
    n_pairs = pair_to_idx_fixture['n_pairs']
    n_genes = len(gene_names_fixture)
    rng = np.random.default_rng(876543)
    data = rng.integers(0, 2, (n_genes, n_pairs), dtype=bool)
    return data


@pytest.fixture
def up_reg_fixture(
        pair_to_idx_fixture,
        gene_names_fixture):
    n_pairs = pair_to_idx_fixture['n_pairs']
    n_genes = len(gene_names_fixture)
    rng = np.random.default_rng(25789)
    data = rng.integers(0, 2, (n_genes, n_pairs), dtype=bool)
    return data


@pytest.fixture
def marker_cache_fixture(
         tmp_dir_fixture,
         is_marker_fixture,
         up_reg_fixture,
         gene_names_fixture,
         pair_to_idx_fixture):

    out_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='markers_',
            suffix='.h5'))

    n_rows = len(gene_names_fixture)
    n_cols = pair_to_idx_fixture['n_pairs']

    is_marker = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)

    up_reg = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)

    for i_row in range(n_rows):
        is_marker.set_row(i_row, is_marker_fixture[i_row, :])
        up_reg.set_row(i_row, up_reg_fixture[i_row, :])

    is_marker.write_to_h5(
        h5_path=out_path,
        h5_group='markers')

    up_reg.write_to_h5(
        h5_path=out_path,
        h5_group='up_regulated')

    # Add some nonsense genes to full_gene_names and shuffle.
    # This is meant to simulate the case where the reference
    # dataset contains genes that are not markers for any
    # taxon pairs.
    rng = np.random.default_rng(887123)
    full_gene_names = copy.deepcopy(gene_names_fixture)
    for ii in range(7):
        full_gene_names.append(f'full_{ii}')
    rng.shuffle(full_gene_names)

    with h5py.File(out_path, 'a') as dst:
        pair_to_idx = copy.deepcopy(pair_to_idx_fixture)
        pair_to_idx.pop('n_pairs')
        dst.create_dataset(
            'pair_to_idx',
            data=json.dumps(pair_to_idx).encode('utf-8'))
        dst.create_dataset(
            'gene_names',
            data=json.dumps(gene_names_fixture).encode('utf-8'))
        dst.create_dataset(
            'full_gene_names',
            data=json.dumps(full_gene_names).encode('utf-8'))
        dst.create_dataset(
            'n_pairs',
            data=n_cols)

    return out_path


@pytest.fixture
def blank_marker_cache_fixture(
         tmp_dir_fixture,
         gene_names_fixture,
         pair_to_idx_fixture):
    """
    Case where there are no marker genes
    """
    out_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='blank_markers_',
            suffix='.h5'))

    n_rows = len(gene_names_fixture)
    n_cols = pair_to_idx_fixture['n_pairs']

    is_marker = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)

    up_reg = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)

    is_marker.write_to_h5(
        h5_path=out_path,
        h5_group='markers')

    up_reg.write_to_h5(
        h5_path=out_path,
        h5_group='up_regulated')

    with h5py.File(out_path, 'a') as dst:
        pair_to_idx = copy.deepcopy(pair_to_idx_fixture)
        pair_to_idx.pop('n_pairs')
        dst.create_dataset(
            'pair_to_idx',
            data=json.dumps(pair_to_idx).encode('utf-8'))
        dst.create_dataset(
            'gene_names',
            data=json.dumps(gene_names_fixture).encode('utf-8'))
        dst.create_dataset(
            'n_pairs',
            data=n_cols)

    return out_path


def test_selecting_from_blank_markers(
        gene_names_fixture,
        taxonomy_tree_fixture,
        blank_marker_cache_fixture):

    marker_array = MarkerGeneArray.from_cache_path(
        cache_path=blank_marker_cache_fixture)

    marker_genes = select_marker_genes_v2(
        marker_gene_array=marker_array,
        query_gene_names=gene_names_fixture,
        taxonomy_tree=TaxonomyTree(data=taxonomy_tree_fixture),
        parent_node=None,
        n_per_utility=5)

    assert marker_genes == []


def test_selecting_from_no_matched_genes(
         marker_cache_fixture,
         taxonomy_tree_fixture):
    """
    Test that case where no genes match raises an error
    """
    marker_array = MarkerGeneArray.from_cache_path(
        cache_path=marker_cache_fixture)

    with pytest.raises(RuntimeError, match='No gene overlap'):
        select_marker_genes_v2(
            marker_gene_array=marker_array,
            query_gene_names=['nope_1', 'nope_2'],
            taxonomy_tree=TaxonomyTree(data=taxonomy_tree_fixture),
            parent_node=None,
            n_per_utility=5)


def test_selection_worker_smoke(
         marker_cache_fixture,
         gene_names_fixture,
         taxonomy_tree_fixture):
    """
    Run a smoketest of _marker_selection_worker
    """
    rng = np.random.default_rng(2231)
    query_gene_names = rng.choice(gene_names_fixture, 40, replace=False)
    output_dict = dict()

    parent_list = [None,
                   ('subclass', 'e'),
                   ('class', 'aa'),
                   ('class', 'bb')]

    for parent in parent_list:
        marker_gene_array = MarkerGeneArray.from_cache_path(
            cache_path=marker_cache_fixture)

        _marker_selection_worker(
            marker_gene_array=marker_gene_array,
            query_gene_names=query_gene_names,
            taxonomy_tree=TaxonomyTree(data=taxonomy_tree_fixture),
            parent_node=parent,
            n_per_utility=5,
            output_dict=output_dict,
            stdout_lock=DummyLock())

    for parent in parent_list:
        if parent == ('class', 'bb'):
            assert len(output_dict[parent]) == 0
        else:
            assert len(output_dict[parent]) > 0
            for g in output_dict[parent]:
                assert g in gene_names_fixture


@pytest.mark.parametrize("behemoth_cutoff", [1000000, 'adaptive', -1])
def test_full_marker_selection_smoke(
         marker_cache_fixture,
         gene_names_fixture,
         taxonomy_tree_fixture,
         behemoth_cutoff,
         tmp_dir_fixture):
    """
    Run a smoketest of select_all_markers

    (testing behemoth_cutoff = -1 is important to make sure
    the code can handle a pile-up in the "behemoth_parents"
    queue)
    """

    # select a behemoth cutoff that affects some but not all
    # parents
    if behemoth_cutoff == 'adaptive':
        n_list = []
        for level in taxonomy_tree_fixture['hierarchy'][:-1]:
            for node in taxonomy_tree_fixture[level]:
                parent = (level, node)
                n_list.append(
                    len(get_all_leaf_pairs(
                            taxonomy_tree=taxonomy_tree_fixture,
                            parent_node=parent)))
        n_list.sort()
        behemoth_cutoff = n_list[len(n_list)//2]
        assert n_list[len(n_list)//4] < behemoth_cutoff
        assert n_list[3*len(n_list)//4] > behemoth_cutoff

    query_gene_names = gene_names_fixture

    result = select_all_markers(
        marker_cache_path=marker_cache_fixture,
        query_gene_names=query_gene_names,
        taxonomy_tree=TaxonomyTree(data=taxonomy_tree_fixture),
        n_per_utility=7,
        n_processors=3,
        behemoth_cutoff=behemoth_cutoff)

    # class bb and subclass a should have no markers
    null_parents = [('class', 'bb'),
                    ('subclass', 'a')]

    parent_list = []
    for level in taxonomy_tree_fixture['hierarchy'][:-1]:
        for node in taxonomy_tree_fixture[level]:
            parent = (level, node)
            parent_list.append(parent)
    parent_list.append(None)

    for parent in parent_list:
        assert parent in result
        if parent in null_parents:
            assert len(result[parent]) == 0
        else:
            assert len(result[parent]) > 0
            for g in result[parent]:
                assert g in query_gene_names


@pytest.mark.parametrize("n_clip", [0, 5, 15])
def test_full_marker_cache_creation_smoke(
         marker_cache_fixture,
         gene_names_fixture,
         taxonomy_tree_fixture,
         tmp_dir_fixture,
         n_clip):
    """
    Run a smoketest of create_marker_cache_from_reference_markers
    """

    taxonomy_tree = TaxonomyTree(data=taxonomy_tree_fixture)

    # select a behemoth cut-off that puts several
    # parents on either side of the divide
    n_list = []
    for level in taxonomy_tree_fixture['hierarchy'][:-1]:
        for node in taxonomy_tree_fixture[level]:
            parent = (level, node)
            ct = len(get_all_leaf_pairs(
                        taxonomy_tree=taxonomy_tree_fixture,
                        parent_node=parent))
            if ct > 0:
                n_list.append(ct)
    n_list.sort()
    behemoth_cutoff = n_list[len(n_list)//2]
    assert n_list[len(n_list)//4] < behemoth_cutoff
    assert n_list[3*len(n_list)//4] > behemoth_cutoff

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='marker_cache_',
        suffix='.h5')

    rng = np.random.default_rng(62234)
    query_gene_names = copy.deepcopy(gene_names_fixture)
    rng.shuffle(query_gene_names)

    # maybe we do not have all the available genes
    if n_clip > 0:
        query_gene_names = query_gene_names[:n_clip]

    # these are the results that should be recorded
    expected = select_all_markers(
        marker_cache_path=marker_cache_fixture,
        query_gene_names=query_gene_names,
        taxonomy_tree=taxonomy_tree,
        n_per_utility=7,
        n_processors=3,
        behemoth_cutoff=behemoth_cutoff)

    create_marker_cache_from_reference_markers(
        output_cache_path=output_path,
        input_cache_path=marker_cache_fixture,
        query_gene_names=query_gene_names,
        taxonomy_tree=taxonomy_tree,
        n_per_utility=7,
        n_processors=3,
        behemoth_cutoff=behemoth_cutoff)

    # because I added some nonsense genes to full_gene_names
    # in the test fixture, to simulate the case where there
    # are genes in the reference data that do not make it
    # through reference marker cache creation
    with h5py.File(marker_cache_fixture, 'r') as src:
        full_gene_names = json.loads(
            src['full_gene_names'][()].decode('utf-8'))

    with h5py.File(output_path, 'r') as actual:

        ref_names = json.loads(
            actual['reference_gene_names'][()].decode('utf-8'))
        assert ref_names == full_gene_names
        query_names = json.loads(
            actual['query_gene_names'][()].decode('utf-8'))
        assert query_names == query_gene_names

        actual_all_ref_idx = actual['all_reference_markers'][()]
        actual_all_query_idx = actual['all_query_markers'][()]
        all_ref_idx = set()
        all_query_idx = set()
        for parent in expected:
            if parent is None:
                actual_grp = 'None'
            else:
                actual_grp = f"{parent[0]}/{parent[1]}"
            assert actual_grp in actual
            actual_reference = []
            for ii in actual[actual_grp]['reference'][()]:
                actual_reference.append(full_gene_names[ii])
                all_ref_idx.add(ii)
            actual_query = []
            for ii in actual[actual_grp]['query'][()]:
                actual_query.append(query_gene_names[ii])
                all_query_idx.add(ii)
            assert actual_reference == actual_query
            assert set(actual_reference) == set(expected[parent])
        assert all_ref_idx == set(actual_all_ref_idx)
        assert all_query_idx == set(actual_all_query_idx)


@pytest.mark.parametrize("n_clip", [0, 5, 15])
def test_marker_serialization(
         marker_cache_fixture,
         gene_names_fixture,
         taxonomy_tree_fixture,
         tmp_dir_fixture,
         n_clip):
    """
    Test method that converts marker genes into a serializable
    dict.
    """

    taxonomy_tree = TaxonomyTree(data=taxonomy_tree_fixture)
    behemoth_cutoff = 1000000

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='marker_cache_',
        suffix='.h5')

    rng = np.random.default_rng(141414)
    query_gene_names = copy.deepcopy(gene_names_fixture)
    rng.shuffle(query_gene_names)

    # maybe we do not have all the available genes
    if n_clip > 0:
        query_gene_names = query_gene_names[:n_clip]

    # these are the results that should be recorded
    expected = select_all_markers(
        marker_cache_path=marker_cache_fixture,
        query_gene_names=query_gene_names,
        taxonomy_tree=taxonomy_tree,
        n_per_utility=7,
        n_processors=3,
        behemoth_cutoff=behemoth_cutoff)

    create_marker_cache_from_reference_markers(
        output_cache_path=output_path,
        input_cache_path=marker_cache_fixture,
        query_gene_names=query_gene_names,
        taxonomy_tree=taxonomy_tree,
        n_per_utility=7,
        n_processors=3,
        behemoth_cutoff=behemoth_cutoff)

    actual = serialize_markers(
        marker_cache_path=output_path,
        taxonomy_tree=taxonomy_tree)

    for parent in taxonomy_tree.all_parents:
        if parent is None:
            parent_key = 'None'
        else:
            parent_key = f'{parent[0]}/{parent[1]}'
        assert parent_key in actual

    ct = 0
    ct_genes = 0
    with h5py.File(output_path, "r") as in_file:
        reference_names = json.loads(
            in_file["reference_gene_names"][()].decode("utf-8"))
        for k in actual:
            actual_markers = actual[k]
            expected_markers = [
                reference_names[ii] for ii in in_file[k]['reference'][()]]
            assert set(actual_markers) == set(expected_markers)
            assert len(actual_markers) == len(expected_markers)
            ct_genes += len(actual_markers)
            ct += 1
    assert ct > 0
    assert ct_genes > 0


def get_all_datasets(h5_handle, current_grp=None):
    """
    return a list of all datasets in an h5ad
    """
    result = []
    for k in h5_handle.keys():
        if isinstance(h5_handle[k], h5py.Dataset):
            if current_grp is not None:
                final_k = f'{current_grp}/{k}'
            else:
                final_k = k
            result.append(final_k)
        else:
            if current_grp is None:
                next_grp = k
            else:
                next_grp = f'{current_grp}/{k}'
            result += get_all_datasets(
                    h5_handle[k],
                    current_grp=next_grp)
    return result


@pytest.mark.parametrize(
    "n_clip, provoke_warning, use_log",
    [(0, True, False),
     (0, False, False),
     (7, True, False),
     (7, False, False),
     (0, True, True)])
def test_marker_serialization_roundtrip(
         marker_cache_fixture,
         gene_names_fixture,
         taxonomy_tree_fixture,
         tmp_dir_fixture,
         n_clip,
         provoke_warning,
         use_log):
    """
    Test that we get the same result back when writing a
    marker cache from specified markers.

    if provoke_warning, add genes that we know are not
    in the query set, so we can test that a warning
    is issued.
    """

    taxonomy_tree = TaxonomyTree(data=taxonomy_tree_fixture)
    behemoth_cutoff = 1000000

    baseline_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='marker_cache_',
        suffix='.h5')

    rng = np.random.default_rng(311723)
    query_gene_names = copy.deepcopy(gene_names_fixture)
    rng.shuffle(query_gene_names)

    # maybe we do not have all the available genes
    if n_clip > 0:
        query_gene_names = query_gene_names[:n_clip]

    # these are the results that should be recorded
    expected = select_all_markers(
        marker_cache_path=marker_cache_fixture,
        query_gene_names=query_gene_names,
        taxonomy_tree=taxonomy_tree,
        n_per_utility=7,
        n_processors=3,
        behemoth_cutoff=behemoth_cutoff)

    create_marker_cache_from_reference_markers(
        output_cache_path=baseline_path,
        input_cache_path=marker_cache_fixture,
        query_gene_names=query_gene_names,
        taxonomy_tree=taxonomy_tree,
        n_per_utility=7,
        n_processors=3,
        behemoth_cutoff=behemoth_cutoff)

    baseline_serialization = serialize_markers(
        marker_cache_path=baseline_path,
        taxonomy_tree=taxonomy_tree)

    as_str = json.dumps(baseline_serialization)
    src_serialization = json.loads(as_str)

    shuffled_serialization = dict()
    ct_markers = 0
    ct_nonsense = 0
    extra_genes = []
    for k in src_serialization:
        markers = src_serialization[k]
        if len(markers) == 0:
            shuffled_serialization[k] = markers
            continue
        if provoke_warning:
            for ii in range(3):
                new_name = f'so_much_garbage_{ct_nonsense}'
                ct_nonsense += 1
                markers.append(new_name)
                extra_genes.append(new_name)

        rng.shuffle(markers)
        shuffled_serialization[k] = markers
        ct_markers += len(markers)

    assert ct_markers > 0
    if provoke_warning:
        assert ct_nonsense > 0

    with h5py.File(marker_cache_fixture, 'r') as src:
        reference_gene_names = json.loads(
            src['full_gene_names'][()].decode('utf-8'))

    for n in extra_genes:
        reference_gene_names.append(n)

    round_trip_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.h5')

    if use_log:
        log = CommandLog()
    else:
        log = None

    if provoke_warning:
        with pytest.warns(UserWarning,
                          match="not present in the query dataset"):

            create_marker_cache_from_specified_markers(
                marker_lookup=shuffled_serialization,
                query_gene_names=query_gene_names,
                reference_gene_names=reference_gene_names,
                output_cache_path=round_trip_path,
                log=log)
        if use_log:
            found_warning = False
            for l in log._log:
                if "not present in the query dataset" in l:
                    found_warning = True
            assert found_warning
    else:
        create_marker_cache_from_specified_markers(
            marker_lookup=shuffled_serialization,
            query_gene_names=query_gene_names,
            reference_gene_names=reference_gene_names,
            output_cache_path=round_trip_path)

    with h5py.File(baseline_path, 'r') as src:
        baseline_dataset_list = get_all_datasets(src)
    with h5py.File(round_trip_path, 'r') as src:
        round_trip_dataset_list = get_all_datasets(src)
    assert len(baseline_dataset_list) > 0
    assert len(baseline_dataset_list) == len(round_trip_dataset_list)
    assert set(baseline_dataset_list) == set(round_trip_dataset_list)

    with h5py.File(baseline_path, 'r') as baseline:
       with h5py.File(round_trip_path, 'r') as round_trip:
           for dataset in baseline_dataset_list:
               if provoke_warning and dataset == 'reference_gene_names':
                   continue
               expected = baseline[dataset][()]
               actual = round_trip[dataset][()]
               np.testing.assert_array_equal(expected, actual)


def test_specified_marker_failure():
    """
    Test that if the reference set is missing marker genes,
    marker cache creation fails.
    """

    marker_lookup = dict()
    marker_lookup['None'] = ['a', 'b', 'c']
    marker_lookup['class/ClassA'] = ['d', 'e' ,'f']

    query_gene_names = [n for n in 'abcdefghij']
    reference_gene_names = [n for n in 'abcdfg']

    with pytest.raises(RuntimeError, match='not in the reference'):
        create_marker_cache_from_specified_markers(
            marker_lookup=marker_lookup,
            query_gene_names=query_gene_names,
            reference_gene_names=reference_gene_names,
            output_cache_path='garbage')


@pytest.mark.parametrize('use_log', [True, False])
def test_specified_marker_empty_parent(
        use_log,
        tmp_dir_fixture):
    """
    Test that, when all of the markers for a given parent are missing
    from the query set, a warning is emitted and all of the query
    genes are used.
    """
    tmp_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.h5')
    reference_gene_names = ['a', 'b', 'c', 'd', 'e']
    query_gene_names = ['e', 'a', 'b']
    marker_lookup = {
        'pa': ['a', 'b'],
        'pb': ['b', 'c', 'e'],
        'pc': ['c', 'd']}

    if use_log:
        log = CommandLog()
    else:
        log = None

    with pytest.warns(UserWarning, match='No markers at parent node'):
        create_marker_cache_from_specified_markers(
            marker_lookup=marker_lookup,
            reference_gene_names=reference_gene_names,
            query_gene_names=query_gene_names,
            output_cache_path=tmp_path,
            log=log)
        if log is not None:
            found_warning = False
            for l in log._log:
                if 'No markers at parent node' in l:
                    found_warning = True
            assert found_warning

            # check for the final warning summarizing all the
            # missing marker genes
            found_warning = False
            for l in log._log:
                if 'marker genes were not present' in l:
                    found_warning = True
            assert found_warning

    with h5py.File(tmp_path, 'r') as src:
        np.testing.assert_array_equal(
            src['pa/reference'][()],
            np.array([0, 1]))

        np.testing.assert_array_equal(
            src['pa/query'][()],
            np.array([1, 2]))

        np.testing.assert_array_equal(
            src['pb/reference'][()],
            np.array([1, 4]))

        np.testing.assert_array_equal(
            src['pb/query'][()],
            np.array([2, 0]))

        # all the query genes, since the markers
        # specified for this parent are not
        # in the query set
        np.testing.assert_array_equal(
            src['pc/reference'][()],
            np.array([0, 1, 4]))

        np.testing.assert_array_equal(
            src['pc/query'][()],
            np.array([1, 2, 0]))
