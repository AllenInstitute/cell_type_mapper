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

from cell_type_mapper.diff_exp.sparse_markers_by_pair import (
    SparseMarkersByPair)

from cell_type_mapper.diff_exp.sparse_markers_by_gene import (
    SparseMarkersByGene)

from cell_type_mapper.marker_selection.marker_array import (
    MarkerGeneArray)

from cell_type_mapper.marker_selection.marker_array_purely_sparse import (
    MarkerGeneArrayPureSparse)


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
    markers_raw = rng.integers(0, 2, (n_genes, n_pairs))
    up_raw = rng.integers(0, 2, (n_genes, n_pairs))
    up_raw = np.logical_and(markers_raw, up_raw)

    n_down = np.logical_and(
        markers_raw,
        np.logical_not(up_raw)).sum()
    assert n_down > 0

    markers = BinarizedBooleanArray(n_rows=n_genes, n_cols=n_pairs)
    up_regulated = BinarizedBooleanArray(n_rows=n_genes, n_cols=n_pairs)
    for i_row in range(n_genes):
        markers.set_row(i_row, markers_raw[i_row, :])
        up_regulated.set_row(i_row, up_raw[i_row, :])

    return {'markers': markers, 'up_regulated': up_regulated}


@pytest.fixture(scope='module')
def pair_to_idx_fixture(
    n_nodes):

    pair_to_idx = dict()
    i_row = 0
    for n0 in range(n_nodes):
        pair_to_idx[f'node_{n0}'] = dict()
        for n1 in range(n0+1, n_nodes, 1):
            pair_to_idx[f'node_{n0}'][f'node_{n1}'] = i_row
            i_row += 1

    pair_to_idx = {'cluster': pair_to_idx}
    return pair_to_idx

@pytest.fixture(scope='module')
def dense_marker_file_fixture(
        marker_array_fixture,
        tmp_dir_fixture,
        pair_to_idx_fixture,
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

    with h5py.File(h5_path, 'a') as dst:
        dst.create_dataset(
            'n_pairs',
            data=n_pairs)

        dst.create_dataset(
            'gene_names',
            data=json.dumps(gene_names).encode('utf-8'))

        dst.create_dataset(
            'pair_to_idx',
            data=json.dumps(pair_to_idx_fixture).encode('utf-8'))

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


@pytest.mark.parametrize(
   "downsample",['genes', 'pairs', 'pair_gene', 'gene_pair', None])
def test_adding_general_sparse_markers_specific_classes(
        dense_marker_file_fixture,
        n_genes,
        n_pairs,
        tmp_dir_fixture,
        downsample):
    """
    Test that the classes used to access the two 'flavors' of sparsity
    give consistent results
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

    rng = np.random.default_rng(22310)

    # test that the two sparse arrays are transposes of each other
    for direction in ('up', 'down'):
        with h5py.File(new_path, 'r') as src:
            by_gene = SparseMarkersByGene(
                pair_idx=src[f'sparse_by_gene/{direction}_pair_idx'][()],
                gene_idx=src[f'sparse_by_gene/{direction}_gene_idx'][()])

            by_pair = SparseMarkersByPair(
                pair_idx=src[f'sparse_by_pair/{direction}_pair_idx'][()],
                gene_idx=src[f'sparse_by_pair/{direction}_gene_idx'][()])

        this_n_pairs = n_pairs
        this_n_genes = n_genes
        if downsample == 'genes':
            gene_idx = rng.choice(np.arange(n_genes), n_genes//3, replace=False)
            by_pair.keep_only_genes(gene_idx)
            by_gene.keep_only_genes(gene_idx)
            this_n_genes = len(gene_idx)
        elif downsample == 'pairs':
            pair_idx = rng.choice(np.arange(n_pairs), n_pairs//3, replace=False)
            by_pair.keep_only_pairs(pair_idx)
            by_gene.keep_only_pairs(pair_idx)
            this_n_pairs = len(pair_idx)
        elif downsample == 'pair_gene' or downsample=='gene_pair':
            gene_idx = rng.choice(np.arange(n_genes), n_genes//3, replace=False)
            pair_idx = rng.choice(np.arange(n_pairs), n_pairs//3, replace=False)

            # make sure order of downsampling does not matter
            if downsample == 'pair_gene':
                by_pair.keep_only_pairs(pair_idx)
                by_pair.keep_only_genes(gene_idx)

                by_gene.keep_only_genes(gene_idx)
                by_gene.keep_only_pairs(pair_idx)
            else:
                by_pair.keep_only_genes(gene_idx)
                by_pair.keep_only_pairs(pair_idx)

                by_gene.keep_only_genes(gene_idx)
                by_gene.keep_only_pairs(pair_idx)

            this_n_pairs = len(pair_idx)
            this_n_genes = len(gene_idx)
        elif downsample is not None:
            raise RuntimeError(
                f"cannot parse downsample={downsample}")

        dense_by_gene = np.zeros((this_n_pairs, this_n_genes), dtype=bool)

        for i_gene in range(this_n_genes):
            dense_by_gene[by_gene.get_pairs_for_gene(i_gene), i_gene] = True
        dense_by_pair = np.zeros((this_n_pairs, this_n_genes), dtype=bool)

        for i_pair in range(this_n_pairs):
            dense_by_pair[i_pair, by_pair.get_genes_for_pair(i_pair)] = True
        np.testing.assert_array_equal(dense_by_gene, dense_by_pair)
        assert dense_by_gene.sum() > 20


def test_sparse_marker_access_class(
        dense_marker_file_fixture,
        n_genes,
        n_pairs,
        n_nodes,
        tmp_dir_fixture,
        pair_to_idx_fixture):

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
        tmp_dir=tmp_dir_fixture,
        delete_dense=True)

    sparse_markers = MarkerGeneArrayPureSparse.from_cache_path(
        new_path)

    # test consistency of idx_of_pair
    for n0 in range(n_nodes):
        for n1 in range(n0+1, n_nodes, 1):
            expected = dense_markers.idx_of_pair(
                level='cluster',
                node1=f'node_{n0}',
                node2=f'node_{n1}')
            actual = sparse_markers.idx_of_pair(
                level='cluster',
                node1=f'node_{n0}',
                node2=f'node_{n1}')
            assert expected == actual

    assert dense_markers.n_pairs == sparse_markers.n_pairs
    assert dense_markers.n_genes == sparse_markers.n_genes
    assert sparse_markers.has_sparse
    assert not dense_markers.has_sparse

    # test consistence of marker_mask_from_pair_idx
    for i_pair in range(dense_markers.n_pairs):
        expected = dense_markers.marker_mask_from_pair_idx(i_pair)
        actual = sparse_markers.marker_mask_from_pair_idx(i_pair)
        np.testing.assert_array_equal(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])

    # test consistence of marker_mask_from_gene_idx
    for i_gene in range(dense_markers.n_genes):
        expected = dense_markers.marker_mask_from_gene_idx(i_gene)
        actual = sparse_markers.marker_mask_from_gene_idx(i_gene)
        np.testing.assert_array_equal(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])

    # test consistence of up_mask_from_pair_idx
    for i_pair in range(dense_markers.n_pairs):
        expected = dense_markers.up_mask_from_pair_idx(i_pair)
        actual = sparse_markers.up_mask_from_pair_idx(i_pair)
        np.testing.assert_array_equal(expected, actual)

    # test consistence of down_mask_from_pair_idx
    for i_pair in range(dense_markers.n_pairs):
        expected = dense_markers.down_mask_from_pair_idx(i_pair)
        actual = sparse_markers.down_mask_from_pair_idx(i_pair)
        np.testing.assert_array_equal(expected, actual)
