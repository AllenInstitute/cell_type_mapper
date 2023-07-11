import pytest

import copy
import numpy as np

from cell_type_mapper.cell_by_gene.utils import (
    convert_to_cpm)

from cell_type_mapper.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)


@pytest.fixture
def gene_id_fixture():
    n_genes = 132
    result = []
    for ii in range(n_genes):
        result.append(f"gene_{ii}")
    return result

@pytest.fixture
def cell_id_fixture():
     n_cells = 23
     result = []
     for ii in range(n_cells):
         result.append(f"cell_{ii}")
     return result


@pytest.fixture
def raw_fixture(gene_id_fixture, cell_id_fixture):
    rng = np.random.default_rng(117231)
    n_cells = len(cell_id_fixture)
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


@pytest.mark.parametrize(
    "use_cell_id", [True, False])
def test_cell_by_gene_init(
       raw_fixture,
       gene_id_fixture,
       cell_id_fixture,
       use_cell_id):
    """
    Test that mapping between gene identifier and column is correct
    """
    if use_cell_id:
        cell_id = cell_id_fixture
    else:
        cell_id = None

    actual = CellByGeneMatrix(
        data=raw_fixture,
        gene_identifiers=gene_id_fixture,
        normalization="raw",
        cell_identifiers=cell_id)

    assert actual.gene_identifiers is not gene_id_fixture
    assert actual.gene_identifiers == gene_id_fixture

    assert actual.n_genes == raw_fixture.shape[1]
    assert actual.n_cells == raw_fixture.shape[0]

    for ii in range(actual.n_genes):
        assert actual.gene_to_col[f"gene_{ii}"] == ii

    if use_cell_id:
        assert actual.cell_identifiers is not cell_id_fixture
        assert actual.cell_identifiers == cell_id_fixture
        for ii in range(actual.n_cells):
            assert actual.cell_to_row[f"cell_{ii}"] == ii
    else:
        assert actual.cell_identifiers is None
        assert actual.cell_to_row is None

@pytest.mark.parametrize(
    "use_cell_id", [True, False])
def test_conversion_to_log2cpm(
        raw_fixture,
        log2cpm_fixture,
        gene_id_fixture,
        cell_id_fixture,
        use_cell_id):

    if use_cell_id:
        cell_id = cell_id_fixture
    else:
        cell_id = None

    raw = CellByGeneMatrix(
        data=raw_fixture,
        gene_identifiers=gene_id_fixture,
        normalization="raw",
        cell_identifiers=cell_id)

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
    assert other.n_cells == raw.n_cells
    assert other.n_genes == raw.n_genes

    if use_cell_id:
        assert other.cell_identifiers is not cell_id_fixture
        assert other.cell_identifiers == cell_id_fixture
        for ii in range(other.n_cells):
            assert other.cell_to_row[f"cell_{ii}"] == ii
    else:
        assert other.cell_identifiers is None
        assert other.cell_to_row is None

    # in place
    raw.to_log2CPM_in_place()
    np.testing.assert_allclose(
        raw.data,
        log2cpm_fixture,
        atol=0.0,
        rtol=1.0e-6)
    assert raw.normalization == "log2CPM"
    assert other.gene_identifiers is not raw.gene_identifiers
    assert other.gene_identifiers == raw.gene_identifiers
    assert other.gene_to_col is not raw.gene_to_col
    assert other.gene_to_col == raw.gene_to_col

    if use_cell_id:
        assert raw.cell_identifiers is not cell_id_fixture
        assert raw.cell_identifiers == cell_id_fixture
        for ii in range(raw.n_cells):
            assert raw.cell_to_row[f"cell_{ii}"] == ii
    else:
        assert raw.cell_identifiers is None
        assert raw.cell_to_row is None


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


@pytest.mark.parametrize(
    "use_cell_id", [True, False])
def test_downsampling(
        raw_fixture,
        gene_id_fixture,
        cell_id_fixture,
        use_cell_id):

    if use_cell_id:
        cell_id = cell_id_fixture
    else:
        cell_id = None

    selected_genes = ["gene_32", "gene_17", "gene_43"]
    expected_data = raw_fixture[:, [32, 17, 43]]
    raw = CellByGeneMatrix(
        data=raw_fixture,
        gene_identifiers=gene_id_fixture,
        normalization="raw",
        cell_identifiers=cell_id)

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

    if use_cell_id:
        assert other.cell_identifiers is not cell_id_fixture
        assert other.cell_identifiers == cell_id_fixture
        for ii in range(other.n_cells):
            assert other.cell_to_row[f"cell_{ii}"] == ii
    else:
        assert other.cell_identifiers is None
        assert other.cell_to_row is None

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

    if use_cell_id:
        assert raw.cell_identifiers is not cell_id_fixture
        assert raw.cell_identifiers == cell_id_fixture
        for ii in range(raw.n_cells):
            assert raw.cell_to_row[f"cell_{ii}"] == ii
    else:
        assert raw.cell_identifiers is None
        assert raw.cell_to_row is None


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


def test_downsample_by_cell_idx(
        raw_fixture,
        gene_id_fixture,
        cell_id_fixture):

    base = CellByGeneMatrix(
        data=raw_fixture,
        gene_identifiers=gene_id_fixture,
        normalization="raw")

    cell_idx = [1, 9, 7]
    expected_data = raw_fixture[[1, 9, 7], :]
    other = base.downsample_cells(cell_idx)
    np.testing.assert_allclose(
        other.data,
        expected_data,
        atol=0.0,
        rtol=1.0e-6)

    assert other.n_cells == 3
    assert other.n_genes == base.n_genes
    assert other.gene_identifiers is not base.gene_identifiers
    assert other.gene_identifiers == base.gene_identifiers
    assert other.gene_to_col is not base.gene_to_col
    assert other.gene_to_col == base.gene_to_col

    # if there are cell identifiers, KeyError should be raised
    base = CellByGeneMatrix(
        data=raw_fixture,
        gene_identifiers=gene_id_fixture,
        normalization="raw",
        cell_identifiers=cell_id_fixture)

    with pytest.raises(KeyError):
        base.downsample_cells(cell_idx)

def test_downsample_by_cell_id(
        raw_fixture,
        gene_id_fixture,
        cell_id_fixture):

    base = CellByGeneMatrix(
        data=raw_fixture,
        gene_identifiers=gene_id_fixture,
        normalization="raw",
        cell_identifiers=cell_id_fixture)

    selected_cells = ["cell_13", "cell_5", "cell_9"]
    expected_data = raw_fixture[[13, 5, 9], :]

    other = base.downsample_cells(selected_cells)
    assert other.n_cells == 3
    assert other.n_genes == base.n_genes
    np.testing.assert_allclose(
        other.data,
        expected_data,
        atol=0.0,
        rtol=1.0e-6)
    assert other.cell_identifiers is not selected_cells
    assert other.cell_identifiers == selected_cells
    assert other.gene_identifiers is not base.gene_identifiers
    assert other.gene_identifiers == base.gene_identifiers
    assert other.gene_to_col is not base.gene_to_col
    assert other.gene_to_col == base.gene_to_col
    assert other.cell_to_row == {
        "cell_13": 0,
        "cell_5": 1,
        "cell_9": 2}

    # if cell_identifier is None, must select cells by idx
    base = CellByGeneMatrix(
        data=raw_fixture,
        gene_identifiers=gene_id_fixture,
        normalization="raw",
        cell_identifiers=None)

    with pytest.raises(IndexError):
        base.downsample_cells(selected_cells)



def test_cpm_after_gene_ds(
        raw_fixture,
        gene_id_fixture):
    """
    Make sure that an exception is raised after you try to
    convert a CellByGeneMatrix that has been downsampled in
    gene space from 'raw' to 'log2CPM'
    """

    raw = CellByGeneMatrix(
        data=raw_fixture,
        gene_identifiers=gene_id_fixture,
        normalization="raw")

    ds1 = raw.downsample_genes(
        selected_genes = [gene_id_fixture[1], gene_id_fixture[3]])
    assert len(raw.gene_identifiers) == len(gene_id_fixture)
    assert len(ds1.gene_identifiers) < len(gene_id_fixture)
    with pytest.raises(RuntimeError, match="downsampled by genes"):
        ds1.to_log2CPM_in_place()
    with pytest.raises(RuntimeError, match="downsampled by genes"):
        ds1.to_log2CPM()

    raw.downsample_genes_in_place(
        selected_genes = [gene_id_fixture[1], gene_id_fixture[3]])
    assert len(raw.gene_identifiers) < len(gene_id_fixture)
    with pytest.raises(RuntimeError, match="downsampled by genes"):
        raw.to_log2CPM_in_place()
    with pytest.raises(RuntimeError, match="downsampled by genes"):
        raw.to_log2CPM()
