import pytest

import h5py
import json
import numpy as np
import pathlib
import scipy.sparse as scipy_sparse
import shutil

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.binary_array.binary_array import (
    BinarizedBooleanArray)

from cell_type_mapper.diff_exp.markers import (
    add_sparse_markers_to_file)

from cell_type_mapper.marker_selection.marker_array import (
    MarkerGeneArray)


@pytest.fixture(scope='module')
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('adding_sparsity_'))
    yield tmp_dir
    _clean_up(tmp_dir)

@pytest.fixture(scope='module')
def n_genes():
    return 54

@pytest.fixture(scope='module')
def n_nodes():
    return 37

@pytest.fixture(scope='module')
def n_pairs(n_nodes):
    return n_nodes*(n_nodes-1)//2

@pytest.fixture(scope='module')
def marker_array_fixture(
        n_genes,
        n_pairs):

    rng = np.random.default_rng(221312)
    n_cols = np.ceil(n_pairs/8).astype(int)
    data = rng.integers(0, 256, (n_genes, n_cols)).astype(np.uint8)
    markers = BinarizedBooleanArray.from_data_array(
        data_array=data,
        n_cols=n_pairs)

    data = rng.integers(0, 256, (n_genes, n_cols)).astype(np.uint8)
    up_regulated = BinarizedBooleanArray.from_data_array(
        data_array=data,
        n_cols=n_pairs)

    return {'markers': markers, 'up_regulated': up_regulated}


@pytest.fixture(scope='module')
def dense_marker_file_fixture(
        marker_array_fixture,
        tmp_dir_fixture,
        n_genes,
        n_nodes,
        n_pairs):

    h5_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='dense_markers_',
        suffix='.h5')

    marker_array_fixture['markers'].write_to_h5(
        h5_path=h5_path,
        h5_group='markers')

    marker_array_fixture['up_regulated'].write_to_h5(
        h5_path=h5_path,
        h5_group='up_regulated')

    gene_names = [f'gene_{ii}' for ii in range(n_genes)]
    pair_to_idx = dict()
    i_row = 0
    for n0 in range(n_nodes):
        pair_to_idx[f'node_{n0}'] = dict()
        for n1 in range(n0+1, n_nodes, 1):
            pair_to_idx[f'node_{n0}'][f'node_{n1}'] = i_row
            i_row += 1

    with h5py.File(h5_path, 'a') as dst:
        dst.create_dataset(
            'n_pairs',
            data=n_pairs)

        dst.create_dataset(
            'gene_names',
            data=json.dumps(gene_names).encode('utf-8'))

        dst.create_dataset(
            'pair_to_idx',
            data=json.dumps(pair_to_idx).encode('utf-8'))

    return h5_path


def test_adding_general_sparse_markers(
        dense_marker_file_fixture,
        n_genes,
        n_pairs,
        tmp_dir_fixture):

    dense_markers = MarkerGeneArray.from_cache_path(
        dense_marker_file_fixture)

    new_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='copy_of_markers_',
        suffix='.h5')

    shutil.copy(
        src=dense_marker_file_fixture,
        dst=new_path)

    add_sparse_markers_to_file(
        h5_path=new_path,
        n_genes=n_genes,
        max_gb=0.6,
        tmp_dir=tmp_dir_fixture)

    sparse_markers = MarkerGeneArray.from_cache_path(
        new_path)

    assert not dense_markers.has_sparse
    assert sparse_markers.has_sparse

    # test that sparse and dense marker arrays give the same
    # result (this only exercises one set of sparse data)
    for i_pair in range(n_pairs):
        t = sparse_markers._up_mask_from_pair_idx_use_sparse(i_pair)
        b = dense_markers.up_mask_from_pair_idx(i_pair)
        np.testing.assert_array_equal(t, b)

        t = sparse_markers._down_mask_from_pair_idx_use_sparse(i_pair)
        b = dense_markers.down_mask_from_pair_idx(i_pair)
        np.testing.assert_array_equal(t, b)

    # test that the two sparse arrays are transposes of each other
    with h5py.File(new_path, 'r') as src:
        assert 'markers' in src
        assert 'up_regulated' in src
        for direction in ('up', 'down'):
            indptr_0 = src[f'sparse_by_pair/{direction}_pair_idx'][()]
            indices_0 = src[f'sparse_by_pair/{direction}_gene_idx'][()]
            data = np.ones(len(indices_0))
            csr = scipy_sparse.csr_matrix(
                (data,
                 indices_0,
                 indptr_0),
                shape=(n_pairs, n_genes))
            csc = scipy_sparse.csc_matrix(csr)
            indptr_1 = src[f'sparse_by_gene/{direction}_gene_idx'][()]
            indices_1 = src[f'sparse_by_gene/{direction}_pair_idx'][()]
            np.testing.assert_array_equal(
                csc.indices,
                indices_1)
            np.testing.assert_array_equal(
                csc.indptr,
                indptr_1)


def test_adding_general_sparse_markers_and_deleting_dense(
        dense_marker_file_fixture,
        n_genes,
        n_pairs,
        tmp_dir_fixture):
    """
    Test that, add_sparse_makers_to_file deletes the dense
    representation of markers when told to.
    """

    new_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='copy_of_markers_',
        suffix='.h5')

    shutil.copy(
        src=dense_marker_file_fixture,
        dst=new_path)

    add_sparse_markers_to_file(
        h5_path=new_path,
        n_genes=n_genes,
        max_gb=0.6,
        tmp_dir=tmp_dir_fixture,
        delete_dense=True)

    # test that the two sparse arrays are transposes of each other
    with h5py.File(new_path, 'r') as src:
        assert 'markers' not in src
        assert 'up_regulated' not in src
        for direction in ('up', 'down'):
            indptr_0 = src[f'sparse_by_pair/{direction}_pair_idx'][()]
            indices_0 = src[f'sparse_by_pair/{direction}_gene_idx'][()]
            data = np.ones(len(indices_0))
            csr = scipy_sparse.csr_matrix(
                (data,
                 indices_0,
                 indptr_0),
                shape=(n_pairs, n_genes))
            csc = scipy_sparse.csc_matrix(csr)
            indptr_1 = src[f'sparse_by_gene/{direction}_gene_idx'][()]
            indices_1 = src[f'sparse_by_gene/{direction}_pair_idx'][()]
            np.testing.assert_array_equal(
                csc.indices,
                indices_1)
            np.testing.assert_array_equal(
                csc.indptr,
                indptr_1)
