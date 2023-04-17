import pytest

import copy
import numpy as np

from hierarchical_mapping.cell_by_gene.utils import (
    convert_to_cpm)

from hierarchical_mapping.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)


@pytest.fixture
def gene_id_fixture():
    n_genes = 132
    result = []
    for ii in range(n_genes):
        result.append(f"gene_{ii}")
    return result


@pytest.fixture
def raw_fixture(gene_id_fixture):
    rng = np.random.default_rng(117231)
    n_cells = 72
    n_genes = len(gene_id_fixture)
    data = rng.random((n_cells, n_genes))
    data[14,: ] = 0.0
    return data


@pytest.fixture
def log2cpm_fixture(raw_fixture):
    cpm = convert_to_cpm(raw_fixture)
    return np.log(cpm+1.0)/np.log(2.0)


def test_cell_by_gene_init_errors():
    """
    Test that basic validation fails when it should
    """
    n_cells = 13
    n_genes = 7
    data = np.zeros((n_cells, n_genes))
    good_id = [f"gene_{ii}" for ii in range(n_genes)]

    # bad normalization
    with pytest.raises(RuntimeError, match="how to handle normalization"):
        CellByGeneMatrix(
            data=data,
            gene_identifiers=good_id,
            normalization="garbage")

    # bad number of gene identifiers
    with pytest.raises(RuntimeError, match="You gave 8 gene_identifiers"):
        CellByGeneMatrix(
            data=data,
            gene_identifiers = [f"{ii}" for ii in range(8)],
            normalization="raw")

    # not unique gene_identifiers
    bad_id = copy.deepcopy(good_id)
    bad_id [1] = "gene_0"
    with pytest.raises(RuntimeError, match="appear more than once"):
        CellByGeneMatrix(
            data=data,
            gene_identifiers=bad_id,
            normalization="raw")


def test_cell_by_gene_init(
       raw_fixture,
       gene_id_fixture):
    """
    Test that mapping between gene identifier and column is correct
    """
    actual = CellByGeneMatrix(
        data=raw_fixture,
        gene_identifiers=gene_id_fixture,
        normalization="raw")

    assert actual.gene_identifiers is not gene_id_fixture
    assert actual.gene_identifiers == gene_id_fixture

    assert actual.n_genes == raw_fixture.shape[1]
    assert actual.n_cells == raw_fixture.shape[0]

    for ii in range(actual.n_genes):
        assert actual.gene_to_col[f"gene_{ii}"] == ii

def test_conversion_to_log2cpm(
        raw_fixture,
        log2cpm_fixture,
        gene_id_fixture):

    raw = CellByGeneMatrix(
        data=raw_fixture,
        gene_identifiers=gene_id_fixture,
        normalization="raw")

    # as other CellByGeneMatrix
    other = raw.to_log2CPM()
    assert other is not raw
    assert not np.allclose(raw.data, other.data, atol=0.0, rtol=1.0e-6)
    np.testing.assert_allclose(
        other.data,
        log2cpm_fixture,
        atol=0.0,
        rtol=1.0e-6)
    assert other.gene_identifiers is not raw.gene_identifiers
    assert other.gene_identifiers == raw.gene_identifiers
    assert other.gene_to_col is not raw.gene_to_col
    assert other.gene_to_col == raw.gene_to_col

    # in place
    raw.to_log2CPM_in_place()
    np.testing.assert_allclose(
        raw.data,
        log2cpm_fixture,
        atol=0.0,
        rtol=1.0e-6)
    assert other.gene_identifiers is not raw.gene_identifiers
    assert other.gene_identifiers == raw.gene_identifiers
    assert other.gene_to_col is not raw.gene_to_col
    assert other.gene_to_col == raw.gene_to_col


def test_conversion_to_log2cpm_error(
        log2cpm_fixture,
        gene_id_fixture):
    """
    Make sure we cannot convert a matrix that is already in
    log2CPM to log2CPM
    """

    base = CellByGeneMatrix(
        data=log2cpm_fixture,
        gene_identifiers=gene_id_fixture,
        normalization="log2CPM")

    with pytest.raises(RuntimeError,  match="not raw"):
        base.to_log2CPM()

    with pytest.raises(RuntimeError,  match="not raw"):
        base.to_log2CPM_in_place()


def test_downsampling(
        raw_fixture,
        gene_id_fixture):

    selected_genes = ["gene_32", "gene_17", "gene_43"]
    expected_data = raw_fixture[:, [32, 17, 43]]
    raw = CellByGeneMatrix(
        data=raw_fixture,
        gene_identifiers=gene_id_fixture,
        normalization="raw")

    # as other CellByGeneMatrix
    other = raw.downsample_genes(selected_genes)
    assert other is not raw
    assert other.gene_identifiers is not selected_genes
    assert other.gene_identifiers == selected_genes
    assert other.gene_to_col == {
        "gene_32": 0,
        "gene_17": 1,
        "gene_43": 2}

    np.testing.assert_array_equal(
        other.data,
        expected_data)

    assert other.n_genes == 3
    assert other.n_cells == raw.n_cells

    # in place
    base_n_cells = raw.n_cells

    raw.downsample_genes_in_place(selected_genes)
    assert raw is not other
    assert raw.gene_identifiers is not selected_genes
    assert raw.gene_identifiers == selected_genes
    assert raw.gene_to_col == {
        "gene_32": 0,
        "gene_17": 1,
        "gene_43": 2}

    np.testing.assert_array_equal(
        raw.data,
        expected_data)

    assert raw.n_genes == 3
    assert raw.n_cells == base_n_cells


def test_downsampling_error(
        raw_fixture,
        gene_id_fixture):

    selected_genes = ["gene_32", "gene_17", "gene_43", "gene_17"]
    raw = CellByGeneMatrix(
        data=raw_fixture,
        gene_identifiers=gene_id_fixture,
        normalization="raw")

    # as other CellByGeneMatrix
    with pytest.raises(RuntimeError, match="gene_17 occurs more than once"):
        raw.downsample_genes(selected_genes)

    # in place
    with pytest.raises(RuntimeError, match="gene_17 occurs more than once"):
        raw.downsample_genes_in_place(selected_genes)
