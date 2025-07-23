import pytest

import gzip
import numpy as np

from cell_type_mapper.utils.utils import (
    mkstemp_clean
)

from cell_type_mapper.test_utils.gene_mapping.mouse_gene_id_lookup import (
    mouse_gene_id_lookup
)


@pytest.fixture()
def suffix_fixture(request):
    return request.param


@pytest.fixture()
def label_heading_fixture(request):
    return request.param


@pytest.fixture()
def label_type_fixture(request):
    return request.param


@pytest.fixture()
def gene_identifier_type_fixture(request):
    if not hasattr(request, 'param'):
        return 'ensembl'
    return request.param


@pytest.fixture()
def x_dtype_fixture(request):
    if not hasattr(request, 'param'):
        return 'integer'
    return request.param


@pytest.fixture()
def transposition_fixture(request):
    if not hasattr(request, 'param'):
        return False
    assert isinstance(request.param, bool)
    return request.param


@pytest.fixture()
def csv_anndata_fixture(
        suffix_fixture,
        label_heading_fixture,
        label_type_fixture,
        gene_identifier_type_fixture,
        x_dtype_fixture,
        transposition_fixture,
        tmp_dir_fixture):
    """
    Returns
        path to CSV file
        list of cell_labels
        list of gene_labels
        x matrix as an array
    """

    suffix = suffix_fixture
    label_heading = label_heading_fixture
    label_type = label_type_fixture

    assert suffix in ('.csv', '.csv.gz')
    assert label_heading in (True, False)
    assert label_type in (
        'string',
        'sequential',
        'big',
        'random',
        'sequential_float',
        'big_float',
        'hybrid',
        'degenerate')

    rng = np.random.default_rng(221111)
    n_cells = 4
    n_genes = 7
    if x_dtype_fixture == 'integer':
        x = rng.integers(10, 100, (n_cells, n_genes))
    elif x_dtype_fixture == 'float':
        x = 10.0*rng.random((n_cells, n_genes))
    else:
        raise RuntimeError(
            "unclear what type of X you want -- "
            f"{x_dtype_fixture}")

    if gene_identifier_type_fixture == 'ensembl':
        gene_labels = sorted(set(mouse_gene_id_lookup.values()))[:n_genes]
    elif gene_identifier_type_fixture == 'symbol':
        gene_labels = sorted(set(mouse_gene_id_lookup.keys()))[:n_genes]
    else:
        raise RuntimeError("unclear what gene identifiers you want")

    if label_type == 'sequential':
        cell_labels = [ii for ii in range(n_cells)]
    elif label_type == 'string':
        cell_labels = [f'c_{ii}' for ii in range(n_cells)]
    elif label_type == 'big':
        cell_labels = [ii for ii in range(1000000, 1000000+80*n_cells, 80)]
    elif label_type == 'random':
        cell_labels = [
           11, 7, 3, 2
        ]
    elif label_type == 'sequential_float':
        cell_labels = [
            ii+0.25 for ii in range(n_cells)
        ]
    elif label_type == 'big_float':
        cell_labels = list(
            1000000*(1.0+rng.random(n_cells))
        )
    elif label_type == 'hybrid':
        cell_labels = [
            11, 'aa', 3, 2
        ]
    elif label_type == 'degenerate':
        cell_labels = [
            'cellA', 'cellA', 'cellB', 'cellB'
        ]

    csv_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix=suffix
    )

    if suffix == '.csv':
        open_fn = open
        is_gzip = False
    else:
        open_fn = gzip.open
        is_gzip = True

    if transposition_fixture:
        row_labels = gene_labels
        col_labels = cell_labels
        output_x = x.transpose()
    else:
        row_labels = cell_labels
        col_labels = gene_labels
        output_x = x

    with open_fn(csv_path, 'w') as dst:
        data = ''
        if label_heading:
            data += 'cell_label'
        for label in col_labels:
            data += f',{label}'
        data += '\n'
        for i_row in range(len(row_labels)):
            data += f'{row_labels[i_row]}'
            for i_col in range(len(col_labels)):
                if x_dtype_fixture == 'integer':
                    data += f',{output_x[i_row, i_col]}'
                else:
                    data += f',{output_x[i_row, i_col]:.4e}'
            data += '\n'
        if is_gzip:
            data = data.encode('utf-8')
        dst.write(data)
    return csv_path, cell_labels, gene_labels, x
