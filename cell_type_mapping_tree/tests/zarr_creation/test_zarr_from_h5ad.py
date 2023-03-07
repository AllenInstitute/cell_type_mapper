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

from hierarchical_mapping.utils.taxonomy_utils import (
    compute_row_order)

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
    cell_name = 0
    for ii in range(7):
        for clu in cluster_list:
            cl = class_to_cluster_fixture[1][clu]
            l2 = l2_to_class_fixture[1][cl]
            l1 = l1_to_l2_fixture[1][l2]
            this = {"cell": f"cell_{cell_name}",
                    "cluster": clu,
                    "class": cl,
                    "level2": l2,
                    "level1": l1,
                    "garbage": rng.integers(8, 1000)}
            records.append(this)
            cell_name += 1

    rng.shuffle(records)
    return records

@pytest.fixture
def obs_fixture(records_fixture):
    obs = pd.DataFrame(records_fixture)
    obs = obs.set_index("cell")
    return obs


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
def var_names(request, ncols):
    if not request.param:
        return None
    var_names = [f"gene_{i}" for i in range(ncols)]
    return var_names

@pytest.fixture
def h5ad_path_fixture(
        obs_fixture,
        x_fixture,
        var_names,
        tmp_path_factory):
    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('anndata'))

    if var_names is None:
        var = None
    else:
        var_data = [{'gene_name': v} for v in var_names]
        var = pd.DataFrame(var_data)
        var = var.set_index('gene_name')

    a_data = anndata.AnnData(X=scipy_sparse.csr_matrix(x_fixture),
                             obs=obs_fixture,
                             var=var,
                             dtype=x_fixture.dtype)
    h5ad_path = tmp_dir / 'h5ad_file.h5ad'
    a_data.write_h5ad(h5ad_path)
    yield h5ad_path
    _clean_up(tmp_dir)


@pytest.fixture
def baseline_tree_fixture(records_fixture):
    return compute_row_order(
        obs_records=records_fixture,
        column_hierarchy=["level1", "level2", "class", "cluster"])


@pytest.mark.parametrize(
        "var_names", [True, False], indirect=["var_names"])
def test_contiguous_zarr(
        x_fixture,
        h5ad_path_fixture,
        baseline_tree_fixture,
        var_names,
        ncols,
        records_fixture,
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

    if var_names is None:
        assert metadata["col_names"] == [str(i) for i in range(ncols)]
    else:
        assert metadata["col_names"] == [f"gene_{i}" for i in range(ncols)]

    row_names = metadata["row_names"]

    new_data = np.zeros(x_fixture.shape, dtype=x_fixture.dtype)
    for ii, r in enumerate(metadata["mapped_row_order"]):
        new_data[ii, :] = x_fixture[r, :]
        assert row_names[ii] == records_fixture[r]['cell']

    with zarr.open(zarr_path, 'r') as in_file:
        actual = load_csr(
                    row_spec=(0, metadata['shape'][0]),
                    n_cols=metadata['shape'][1],
                    data=in_file['data'],
                    indices=in_file['indices'],
                    indptr=in_file['indptr'])

    np.testing.assert_allclose(actual, new_data)

    assert metadata['h5ad_path'] == str(h5ad_path_fixture.resolve().absolute())

    for k in baseline_tree_fixture["tree"]:
        if k == "hierarchy":
            expected = baseline_tree_fixture["tree"][k]
            actual = metadata["taxonomy_tree"][k]
            assert expected == actual
        else:
            for node in baseline_tree_fixture["tree"][k]:
                expected = set(baseline_tree_fixture["tree"][k][node])
                actual = set(metadata["taxonomy_tree"][k][node])
                assert expected == actual

    _clean_up(zarr_path)
