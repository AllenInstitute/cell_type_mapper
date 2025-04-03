import pytest

import gzip
import numpy as np

from cell_type_mapper.utils.utils import (
    mkstemp_clean
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
def csv_anndata_fixture(
        suffix_fixture,
        label_heading_fixture,
        label_type_fixture,
        tmp_dir_fixture):

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
        'hybrid')

    rng = np.random.default_rng(221111)
    n_cells = 4
    n_genes = 7
    x = rng.integers(10, 100, (n_cells, n_genes))
    gene_labels = [f'g_{ii}' for ii in range(n_genes)]
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

    with open_fn(csv_path, 'w') as dst:
        data = ''
        if label_heading:
            data += 'cell_label'
        for label in gene_labels:
            data += f',{label}'
        data += '\n'
        for i_row in range(n_cells):
            data += f'{cell_labels[i_row]}'
            for i_col in range(n_genes):
                data += f',{x[i_row, i_col]}'
            data += '\n'
        if is_gzip:
            data = data.encode('utf-8')
        dst.write(data)
    return csv_path, cell_labels, gene_labels, x
