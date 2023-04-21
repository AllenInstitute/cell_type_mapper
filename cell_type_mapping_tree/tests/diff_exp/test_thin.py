import pytest

import h5py
import json
import numpy as np
import pathlib

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.diff_exp.thin import (
    thin_marker_file)


@pytest.fixture
def tmp_dir_fixture(tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('thin'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def n_genes():
    return 2345


@pytest.fixture
def nonzero_gene_fixture(n_genes):
    rng = np.random.default_rng(766123)
    chosen = rng.choice(np.arange(n_genes, dtype=int),
                        1000, replace=False)
    chosen = np.sort(chosen)
    return chosen

@pytest.fixture
def baseline_marker_fixture(
        tmp_dir_fixture,
        n_genes,
        nonzero_gene_fixture):

    rng = np.random.default_rng(7653320)

    tmp_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5')

    n_pairs = 999

    marker_data = np.zeros(
        (n_genes, n_pairs),
        dtype=np.uint8)

    up_reg_data = np.zeros(
        (n_genes, n_pairs),
        dtype=np.uint8)

    for i_row in nonzero_gene_fixture:
        marker_data[i_row, :] = rng.integers(0, 255, n_pairs, dtype=np.uint8)
        up_reg_data[i_row, :] = rng.integers(0, 255, n_pairs, dtype=np.uint8)

    with h5py.File(tmp_path, 'w') as out_file:
        out_file.create_dataset(
            'gene_names',
            data=json.dumps([f'g_{ii}'
                             for ii in range(n_genes)]).encode('utf-8'))

        out_file.create_dataset(
            'n_pairs', data=n_pairs)
        out_file.create_dataset(
            'pair_to_idx', data=json.dumps('abcdefg').encode('utf-8'))
        out_file.create_dataset(
            'markers/data', data=marker_data)
        out_file.create_dataset(
            'up_regulated/data', data=up_reg_data)

    return tmp_path


def test_thin_marker_file(
        nonzero_gene_fixture,
        baseline_marker_fixture,
        tmp_dir_fixture):

    thin_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5')

    thin_marker_file(
        marker_file_path=baseline_marker_fixture,
        thinned_marker_file_path=thin_path,
        n_processors=3,
        max_bytes=5000)

    with h5py.File(baseline_marker_fixture, 'r') as expected:
        with h5py.File(thin_path, 'r') as actual:
            assert actual['pair_to_idx'][()] == expected['pair_to_idx'][()]
            assert actual['n_pairs'][()] == expected['n_pairs'][()]
            actual_genes = json.loads(
                actual['gene_names'][()].decode('utf-8'))
            expected_genes = json.loads(
                expected['gene_names'][()].decode('utf-8'))
            expected_genes = [
                expected_genes[ii] for ii in nonzero_gene_fixture]
            assert actual_genes == expected_genes

            expected_marker = expected['markers/data'][()]
            expected_marker = expected_marker[nonzero_gene_fixture, :]
            np.testing.assert_array_equal(
                actual['markers/data'][()],
                expected_marker)
            expected_up = expected['up_regulated/data'][()]
            expected_up = expected_up[nonzero_gene_fixture, :]
            np.testing.assert_array_equal(
                actual['up_regulated/data'][()],
                expected_up)
