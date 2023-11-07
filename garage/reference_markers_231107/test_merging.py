import pytest

import h5py
import numpy as np
import scipy.sparse as scipy_sparse


from p_value_mask import _merge_masks

from cell_type_mapper.utils.utils import (
    _clean_up,
    mkstemp_clean)


@pytest.fixture(scope='session')
def tmp_dir_fixture(
        tmp_path_factory):
    result = tmp_path_factory.mktemp('cell_type_mapper_')
    yield result
    _clean_up(result)


@pytest.fixture
def n_genes_fixture():
    return 35


@pytest.fixture
def n_pairs_fixture():
    return 112


@pytest.fixture
def true_mask_fixture(
        n_genes_fixture,
        n_pairs_fixture):

    rng = np.random.default_rng(221131)
    parent = rng.integers(0, 2,
                          (n_pairs_fixture, n_genes_fixture),
                          dtype=np.uint8)
    parent[77:84, :] = 0
    child0 = np.copy(parent[0:28, :])
    child1 = np.copy(parent[28:77, :])
    child2 = np.copy(parent[77:83, :])
    child3 = np.copy(parent[83:, :])
    return {
        'parent': parent,
        'children': [(0, child0), (28, child1),
                     (77, child2), (83, child3)]
    }


@pytest.fixture
def src_path_list_fixture(
        true_mask_fixture,
        tmp_dir_fixture,
        n_genes_fixture):

    src_path_list = []
    for child in true_mask_fixture['children']:
        path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.h5')
        src_path_list.append(path)
        sparse = scipy_sparse.csr_matrix(child[1])
        with h5py.File(path, 'w') as dst:
            dst.create_dataset('n_genes', data=n_genes_fixture)
            dst.create_dataset('n_pairs', data=child[1].shape[1])
            dst.create_dataset('indices', data=sparse.indices)
            dst.create_dataset('indptr', data=sparse.indptr)
            dst.create_dataset('min_row', data=child[0])

    return src_path_list


def test_p_value_merge(
            src_path_list_fixture,
            n_genes_fixture,
            n_pairs_fixture,
            tmp_dir_fixture,
            true_mask_fixture):

    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5')

    _merge_masks(
        src_path_list=src_path_list_fixture,
        dst_path=dst_path)

    expected_sparse = scipy_sparse.csr_array(true_mask_fixture['parent'])
    with h5py.File(dst_path, 'r') as src:
        indices = src['indices'][()]
        indptr = src['indptr'][()]
    np.testing.assert_array_equal(
        indices, expected_sparse.indices)
    np.testing.assert_array_equal(
        indptr, expected_sparse.indptr)
