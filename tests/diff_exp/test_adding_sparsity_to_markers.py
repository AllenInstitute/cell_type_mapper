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
    add_sparse_by_gene_markers_to_file)

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
    down_raw = np.logical_and(markers_raw, np.logical_not(up_raw))
    return {'down': down_raw, 'up': up_raw}


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
def marker_file_fixture(
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

    csc_down = scipy_sparse.csc_array(marker_array_fixture['down'])
    csc_up = scipy_sparse.csc_array(marker_array_fixture['up'])
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

        dst.create_dataset(
            'sparse_by_pair/up_pair_idx',
            data=csc_up.indptr)
        dst.create_dataset(
            'sparse_by_pair/up_gene_idx',
            data=csc_up.indices)
        dst.create_dataset(
            'sparse_by_pair/down_pair_idx',
            data=csc_down.indptr)
        dst.create_dataset(
           'sparse_by_pair/down_gene_idx',
           data=csc_down.indices)

    return h5_path


def test_adding_by_gene_sparse(
        marker_file_fixture,
        n_genes,
        n_pairs,
        tmp_dir_fixture):

    new_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='copy_of_markers_',
        suffix='.h5')

    shutil.copy(
        src=marker_file_fixture,
        dst=new_path)

    add_sparse_by_gene_markers_to_file(
        h5_path=new_path,
        n_genes=n_genes,
        max_gb=0.6,
        tmp_dir=tmp_dir_fixture)

    sparse_markers = MarkerGeneArrayPureSparse.from_cache_path(
        new_path)

    assert sparse_markers.has_sparse


    # test that the two sparse arrays are transposes of each other
    with h5py.File(new_path, 'r') as src:
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
def test_sparse_markers_specific_classes(
        marker_file_fixture,
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
        src=marker_file_fixture,
        dst=new_path)

    add_sparse_by_gene_markers_to_file(
        h5_path=new_path,
        n_genes=n_genes,
        max_gb=0.6,
        tmp_dir=tmp_dir_fixture)

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


@pytest.mark.parametrize(
    "downsampling",
    [None, ('genes',), ('pairs',), ('genes', 'pairs'), ('pairs', 'genes')])
def test_sparse_marker_access_class(
        marker_file_fixture,
        marker_array_fixture,
        n_genes,
        n_pairs,
        n_nodes,
        tmp_dir_fixture,
        pair_to_idx_fixture,
        downsampling):

    up_array = np.copy(marker_array_fixture['up'])
    down_array = np.copy(marker_array_fixture['down'])

    new_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='copy_of_markers_',
        suffix='.h5')

    shutil.copy(
        src=marker_file_fixture,
        dst=new_path)

    add_sparse_by_gene_markers_to_file(
        h5_path=new_path,
        n_genes=n_genes,
        max_gb=0.6,
        tmp_dir=tmp_dir_fixture)

    sparse_markers = MarkerGeneArrayPureSparse.from_cache_path(
        new_path)

    rng = np.random.default_rng(66513)
    all_pairs = [
       ('cluster', f'node_{i0}', f'node_{i1}')
       for i0 in range(n_nodes) for i1 in range(i0+1, n_nodes, 1)]

    chosen_pairs = None
    chosen_genes = None
    if downsampling is not None:
        if 'pairs' in downsampling:
            chosen_pairs = rng.choice(all_pairs, 56, replace=False)
            chosen_pair_idx = np.array(
                [pair_to_idx_fixture['cluster'][n[1]][n[2]]
                 for n in chosen_pairs])
        if 'genes' in downsampling:
            chosen_genes = rng.choice(
                np.arange(n_genes),
                n_genes//3,
                replace=False)
    if chosen_pairs is None:
        chosen_pairs = all_pairs
        chosen_pair_idx = np.arange(n_pairs)
    if chosen_genes is None:
        chosen_genes = np.arange(n_genes)

    if downsampling is not None:
        if 'genes' in downsampling:
            up_array = up_array[chosen_genes, :]
            down_array = down_array[chosen_genes, :]

        if downsampling[0] == 'genes':
            sparse_markers.downsample_genes(chosen_genes)
            expected_n_genes = len(chosen_genes)
        elif downsampling[0] == 'pairs':
            sparse_markers = sparse_markers.downsample_pairs_to_other(chosen_pairs)
            expected_n_pairs = len(chosen_pairs)
        else:
            raise RuntimeError(f"invalid downsampling {downsampling}")
        if len(downsampling) == 2:
            if downsampling[1] == 'genes':
                sparse_markers.downsample_genes(chosen_genes)
                expected_n_genes = len(chosen_genes)
            elif downsampling[1] == 'pairs':
                sparse_markers = sparse_markers.downsample_pairs_to_other(chosen_pairs)
                expected_n_pairs = len(chosen_pairs)
            else:
                raise RuntimeError(f"invalid downsampling {downsampling}")

    for expected, pair in enumerate(chosen_pairs):
        actual = sparse_markers.idx_of_pair(
            level=pair[0],
            node1=pair[1],
            node2=pair[2])
        assert expected == actual

    if downsampling is None or 'genes' not in downsampling:
        expected_n_genes = n_genes
    else:
        expected_n_genes = len(chosen_genes)

    if downsampling is None or 'pairs' not in downsampling:
        expected_n_pairs = n_pairs
    else:
        n_pairs = len(chosen_pairs)

    assert sparse_markers.n_pairs == expected_n_pairs
    assert sparse_markers.n_genes == expected_n_genes
    assert sparse_markers.has_sparse

    # test consistence of marker_mask_from_pair_idx
    for i_pair, pair in enumerate(chosen_pairs):
        pair_idx = pair_to_idx_fixture['cluster'][pair[1]][pair[2]]
        actual = sparse_markers.marker_mask_from_pair_idx(i_pair)

        expected_up = up_array[:, pair_idx]
        expected_down = down_array[:, pair_idx]
        expected_marker = np.logical_or(expected_up, expected_down)

        np.testing.assert_array_equal(expected_marker, actual[0])
        np.testing.assert_array_equal(expected_up, actual[1])

    # test consistence of marker_mask_from_gene_idx
    for i_gene in range(len(chosen_genes)):
        actual = sparse_markers.marker_mask_from_gene_idx(i_gene)

        expected_up = up_array[i_gene, :]
        expected_up = expected_up[chosen_pair_idx]
        expected_down = down_array[i_gene, :]
        expected_down = expected_down[chosen_pair_idx]
        expected_marker = np.logical_or(expected_up, expected_down)

        np.testing.assert_array_equal(expected_marker, actual[0])
        np.testing.assert_array_equal(expected_up, actual[1])

    # test consistence of up_mask_from_pair_idx
    for i_pair, pair in enumerate(chosen_pairs):
        pair_idx = pair_to_idx_fixture['cluster'][pair[1]][pair[2]]
        actual = sparse_markers.up_mask_from_pair_idx(i_pair)
        expected = up_array[:, pair_idx]
        np.testing.assert_array_equal(expected, actual)

    # test consistence of down_mask_from_pair_idx
    for i_pair, pair in enumerate(chosen_pairs):
        pair_idx = pair_to_idx_fixture['cluster'][pair[1]][pair[2]]
        expected = down_array[:, pair_idx]
        actual = sparse_markers.down_mask_from_pair_idx(i_pair)
        np.testing.assert_array_equal(expected, actual)
