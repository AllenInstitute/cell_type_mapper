import pytest

import h5py
from itertools import product
import json
import numpy as np
import pathlib
import scipy.sparse as scipy_sparse
import shutil

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.marker_selection.marker_array import (
    MarkerGeneArray)

from cell_type_mapper.diff_exp.markers import (
    add_sparse_by_gene_markers_to_file)

from cell_type_mapper.marker_selection.utils import (
    create_utility_array)


@pytest.fixture(scope='module')
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('marker_array'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def n_rows():
    return 551


@pytest.fixture
def n_cols():
    return 245

@pytest.fixture
def mask_array_fixture(
        n_rows,
        n_cols):
    rng =np.random.default_rng(554422)
    data = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    return data

@pytest.fixture
def up_regulated_fixture(
        n_rows,
        n_cols):
    rng =np.random.default_rng(118823)
    data = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    return data


@pytest.fixture
def marker_with_sparse_fixture(
        mask_array_fixture,
        up_regulated_fixture,
        tmp_dir_fixture,
        n_rows,
        n_cols):

    h5_path = pathlib.Path(
        mkstemp_clean(dir=tmp_dir_fixture,
                      suffix='.h5'))

    up_regulated = np.logical_and(
        mask_array_fixture,
        up_regulated_fixture)
    down_regulated = np.logical_and(
        mask_array_fixture,
        np.logical_not(up_regulated_fixture))

    csc_up = scipy_sparse.csc_array(up_regulated)
    csc_down = scipy_sparse.csc_array(down_regulated)

    with h5py.File(h5_path, "w") as out_file:
        out_file.create_dataset('n_pairs', data=n_cols)
        out_file.create_dataset(
            'gene_names',
            data=json.dumps(
                [f"g_{ii}" for ii in range(n_rows)]).encode('utf-8'))
        out_file.create_dataset(
            'pair_to_idx',
            data=json.dumps({'a': 1, 'b':2}).encode('utf-8'))

        grp = out_file.create_group('sparse_by_pair')
        grp.create_dataset(
            'up_gene_idx',
            data=csc_up.indices)
        grp.create_dataset(
            'up_pair_idx',
            data=csc_up.indptr)
        grp.create_dataset(
            'down_gene_idx',
            data=csc_down.indices)
        grp.create_dataset(
            'down_pair_idx',
            data=csc_down.indptr)

    add_sparse_by_gene_markers_to_file(
        h5_path=h5_path,
        n_genes=n_rows,
        max_gb=3,
        tmp_dir=tmp_dir_fixture)

    with h5py.File(h5_path, 'r') as src:
        assert 'sparse_by_pair' in src

    return h5_path


@pytest.mark.parametrize(
        "gb_size, taxonomy_mask",
        product([1, 1.0e-7],
                [None, np.array([13, 22, 81, 37])]))
def test_create_utility_array(
        mask_array_fixture,
        up_regulated_fixture,
        marker_with_sparse_fixture,
        n_rows,
        n_cols,
        gb_size,
        taxonomy_mask):

    arr = MarkerGeneArray.from_cache_path(
        cache_path=marker_with_sparse_fixture)

    (actual_util,
     actual_census) = create_utility_array(
        marker_gene_array=arr,
        gb_size=gb_size,
        taxonomy_mask=taxonomy_mask)

    if taxonomy_mask is None:
        expected_util = np.sum(mask_array_fixture, axis=1)
        expected_census = np.zeros((mask_array_fixture.shape[1], 2), dtype=int)
        for i_row in range(n_rows):
            for i_col in range(n_cols):
                if mask_array_fixture[i_row, i_col]:
                    if up_regulated_fixture[i_row, i_col]:
                        expected_census[i_col, 1] += 1
                    else:
                        expected_census[i_col, 0] += 1
    else:
        expected_util = np.zeros(n_rows, dtype=int)
        expected_census = np.zeros((len(taxonomy_mask), 2), dtype=int)
        for i_row in range(n_rows):
            for i_taxon, i_col in enumerate(taxonomy_mask):
                if mask_array_fixture[i_row, i_col]:
                    expected_util[i_row] += 1
                    if up_regulated_fixture[i_row, i_col]:
                        expected_census[i_taxon, 1] += 1
                    else:
                        expected_census[i_taxon, 0] += 1

    np.testing.assert_array_equal(expected_census, actual_census)
    assert expected_census.sum() > 0

    assert expected_util.shape == (n_rows, )
    np.testing.assert_array_equal(expected_util,
                                  actual_util)
    assert expected_util.sum() > 0
