import pytest

import anndata
import json
import numpy as np
import pandas as pd
import scipy.sparse as scipy_sparse
import pathlib
import zarr

from hierarchical_mapping.utils.utils import (
    _clean_up)

from hierarchical_mapping.zarr_creation.zarr_from_h5ad import (
    contiguous_zarr_from_h5ad)

from hierarchical_mapping.utils.sparse_utils import (
    load_csr)

@pytest.fixture
def ncols():
    return 71


@pytest.fixture
def l1_to_l2_fixture():
    """
    Fixture modeling which level 2 objects belong
    to level 1
    """
    forward = {"l1a": set(["l2a", "l2d", "l2e"]),
               "l1b": set(["l2b", "l2f"]),
               "l1c": set(["l2c"])}

    backward = dict()
    for k in forward:
        for i in forward[k]:
            backward[i] = k
    return forward, backward


@pytest.fixture
def l2_to_class_fixture():
    """
    Fixture modeling which class objects belong
    to which level 2 objects
    """
    forward = {"l2a": set(["c4", "c5"]),
               "l2b": set(["c1", "c6"]),
               "l2c": set(["c3"]),
               "l2d": set(["c2", "c7", "c8"]),
               "l2e": set(["c9"]),
               "l2f": set(["c10", "c11"])}

    backward = dict()
    for k in forward:
        for i in forward[k]:
            backward[i] = k
    return forward, backward

@pytest.fixture
def class_to_cluster_fixture(l2_to_class_fixture):
    """
    Fixture modeling which cluster objects belong
    to which class objects
    """
    list_of_classes = list(l2_to_class_fixture[1].keys())

    forward = dict()
    backward = dict()
    ct = 0
    for c in list_of_classes:
        forward[c] = set()
        for ii in range(4):
            this = f"clu_{ct}"
            ct += 1
            backward[this] = c
            forward[c].add(this)

    return forward, backward


@pytest.fixture
def records_fixture(
         class_to_cluster_fixture,
         l2_to_class_fixture,
         l1_to_l2_fixture):
    rng = np.random.default_rng(871234)
    cluster_list = list(class_to_cluster_fixture[1].keys())
    records = []
    for ii in range(7):
        for clu in cluster_list:
            cl = class_to_cluster_fixture[1][clu]
            l2 = l2_to_class_fixture[1][cl]
            l1 = l1_to_l2_fixture[1][l2]
            this = {"cluster": clu,
                    "class": cl,
                    "level2": l2,
                    "level1": l1,
                    "garbage": rng.integers(8, 1000)}
            records.append(this)

    rng.shuffle(records)
    return records

@pytest.fixture
def obs_fixture(records_fixture):
    return pd.DataFrame(records_fixture)


@pytest.fixture
def nrows(records_fixture):
    return len(records_fixture)

@pytest.fixture
def x_fixture(records_fixture, ncols, nrows):
    rng = np.random.default_rng(66213)

    data = np.zeros(nrows*ncols, dtype=np.float64)
    chosen_dex = rng.choice(np.arange(nrows*ncols, dtype=int),
                            nrows*ncols//7,
                            replace=False)
    data[chosen_dex] = rng.random(len(chosen_dex))
    data = data.reshape((nrows, ncols))
    return data


@pytest.fixture
def h5ad_path_fixture(
        obs_fixture,
        x_fixture,
        tmp_path_factory):
    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('anndata'))
    a_data = anndata.AnnData(X=scipy_sparse.csr_matrix(x_fixture),
                             obs=obs_fixture)
    h5ad_path = tmp_dir / 'h5ad_file.h5ad'
    a_data.write_h5ad(h5ad_path, force_dense=False)
    import h5py
    with h5py.File(h5ad_path, 'r', swmr=True) as in_file:
        d = in_file['X']['data']
    yield h5ad_path
    _clean_up(tmp_dir)


def test_contiguous_zarr(
        x_fixture,
        h5ad_path_fixture,
        tmp_path_factory):
    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('contiguous'))
    h5ad_dir = tmp_dir / 'h5ad'
    h5ad_dir.mkdir()
    zarr_path = tmp_dir / 'as_zarr.zarr'

    contiguous_zarr_from_h5ad(
        h5ad_path=h5ad_path_fixture,
        zarr_path=zarr_path,
        taxonomy_hierarchy=["level1", "level2", "class", "cluster"],
        zarr_chunks=100,
        write_buffer_size=50,
        read_buffer_size=100,
        n_processors=3,
        tmp_dir=h5ad_dir)

    # make sure temporary files cleaned up after themselves
    h5ad_contents = [n for n in h5ad_dir.iterdir()]
    assert len(h5ad_contents) == 0

    metadata = json.load(open(zarr_path/"metadata.json", "rb"))
    new_data = np.zeros(x_fixture.shape, dtype=x_fixture.dtype)
    for ii, r in enumerate(metadata["mapped_row_order"]):
        new_data[ii, :] = x_fixture[r, :]

    with zarr.open(zarr_path, 'r') as in_file:
        actual = load_csr(
                    row_spec=(0, metadata['shape'][0]),
                    n_cols=metadata['shape'][1],
                    data=in_file['data'],
                    indices=in_file['indices'],
                    indptr=in_file['indptr'])

    np.testing.assert_allclose(actual, new_data)

    _clean_up(zarr_path)
