import pytest

import h5py
import itertools
import numpy as np

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.diff_exp.p_value_markers import (
    _get_validity_mask)


@pytest.fixture(scope='module')
def tmp_dir(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp('p_mask_markers_')
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture(scope='module')
def gene_names():
    """
    List of gene names from this dataset.
    """
    return [f'g_{ii}' for ii in range(100)]


@pytest.fixture(scope='module')
def _pair_to_idx():
    """
    The pair_to_idx dict that needs to be in the p_mask
    and reference marker HDF5 files.
    """
    clusters = [f'cl_{ii}' for ii in range(5)]
    ct = 0
    result = dict()
    result['cluster'] = dict()
    pair_list = []
    for pair in itertools.combinations(clusters, 2):
        pair = list(pair)
        pair.sort()
        pair_list.append(pair)
        if pair[0] not in result['cluster']:
            result['cluster'][pair[0]] = dict()
        result['cluster'][pair[0]][pair[1]] = ct
        ct += 1
    return result, ct, pair_list


@pytest.fixture(scope='module')
def pair_to_idx(_pair_to_idx):
    """
    The pair_to_idx dict that needs to be in the p_mask
    and reference marker HDF5 files.
    """
    return _pair_to_idx[0]


@pytest.fixture(scope='module')
def n_pairs(_pair_to_idx):
    return _pair_to_idx[1]


@pytest.fixture(scope='module')
def pair_list(_pair_to_idx):
    return _pair_to_idx[2]


@pytest.fixture(scope='module')
def pair_to_radius(
        pair_list,
        pair_to_idx,
        gene_names):
    """
    A dict mapping a cluster_pair to a lookup table
    mapping gene_name to penetrance radius.
    """
    rng = np.random.default_rng(221312)
    for pair in pair_list:
        pair_idx = pair_to_idx['cluster'][pair[0]][pair[1]]
        radii = np.zeros(len(gene_names))

        # genes that are 'absolutely valid'
        i0 = 3*(pair_idx)
        i1 = 3*(pair_idx+1)
        radii[i0:i1] = -1.0



@pytest.mark.parametrize(
    "n_valid, valid_gene_idx, expected_markers",
    [(3, None, (6, 9, 11)),  # genes that absolutely pass penetrance test
     (5, None, (5, 6, 7, 9, 11)),  # grab smallest 2 distances after absolute passing genes
     (5, np.array([2, 3, 5, 6, 11, 12, 13]), (5, 6, 11, 12, 13)),  # 2, 3 do not pass p-value test
     (5, np.array([2, 3, 5, 6, 7, 8, 10, 11]), (5, 6, 7, 8, 11)),  # do not need 10, 11 (distance too large)
     (5, np.array([2, 3, 5, 6, 8, 10, 11, 12, 13]), (5, 6, 8, 10, 11, 12)),  # degeneracy in distance between 10 and 12
    ]
)
def test_get_validity_mask(
        n_valid,
        valid_gene_idx,
        expected_markers):
    rng = np.random.default_rng(2131)

    n_genes = 20
    gene_indices = np.arange(5, 16, dtype=int)
    raw_distances = np.arange(1, len(gene_indices)+1, dtype=float)

    raw_distances[7] = raw_distances[5]  # gene 12 == gene 10

    raw_distances[1] = -1.0  # gene 6
    raw_distances[4] = -1.0  # gene 9
    raw_distances[6] = -1.0  # gene 11

    actual = _get_validity_mask(
        n_valid=n_valid,
        n_genes=n_genes,
        gene_indices=gene_indices,
        raw_distances=raw_distances,
        valid_gene_idx=valid_gene_idx)

    expected = np.zeros(n_genes, dtype=bool)
    for ii in expected_markers:
        expected[ii] = True

    np.testing.assert_array_equal(expected, actual)



def test_dummy(
        pair_list,
        n_pairs,
        pair_to_idx):
    pass
