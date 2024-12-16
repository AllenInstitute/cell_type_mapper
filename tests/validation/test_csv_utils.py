"""
This is where we will test the utility functions associated with accepting
CSV files as inputs to the MapMyCells validator
"""

import pytest

import anndata
import gzip
import itertools
import numpy as np

from cell_type_mapper.utils.utils import (
    mkstemp_clean
)

from cell_type_mapper.validation.csv_utils import (
    is_first_column_sequential,
    is_first_column_large,
    convert_csv_to_h5ad
)


def test_is_first_column_sequential():
    """
    Test utility to detect if first column in a numpy array
    is sequential when sorted
    """
    xx = np.array(
        [[12, 4, 6],
         [14, 11, 13],
         [13, 55, 66]]
    )
    assert is_first_column_sequential(xx)

    xx = np.array(
        [[12.0, 4.3, 6.01],
         [14.0, 11.1, 13.98],
         [13.0, 55.2, 66.5]]
    )
    assert is_first_column_sequential(xx)

    xx = np.array(
        [[2, 4, 6],
         [5, 11, 13],
         [3, 55, 66]]
    )
    assert not is_first_column_sequential(xx)

    xx = np.array(
        [[2, 4, 6],
         [2, 11, 13],
         [3, 55, 66]]
    )
    assert not is_first_column_sequential(xx)


def test_is_first_column_large():
    """
    Test utility to detect if first column in array
    is abnormally large
    """
    xx = np.array(
        [[10, 2, 3, 0, 4],
         [20, 1, 1, 0, 1],
         [30, 4, 1, 2, 0]]
    )
    assert is_first_column_large(xx)

    xx = np.array(
        [[10, 2, 3, 0, 4],
         [20, 1, 1, 0, 1],
         [0, 4, 1, 2, 0]]
    )
    assert not is_first_column_large(xx)

    xx = np.array(
        [[4, 2, 3, 0, 4],
         [1, 1, 1, 0, 1],
         [3, 4, 1, 2, 0]]
    )
    assert not is_first_column_large(xx)


@pytest.mark.parametrize(
    "label_heading,label_type,suffix",
    itertools.product(
        [True, False],
        ['string', 'sequential', 'big', 'random'],
        ['.csv', '.csv.gz']
    )
)
def test_convert_csv(
        tmp_dir_fixture,
        label_heading,
        label_type,
        suffix):

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

    h5ad_path, flag = convert_csv_to_h5ad(
        src_path=csv_path,
        log=None)
    assert flag

    adata = anndata.read_h5ad(h5ad_path, backed='r')

    if not label_heading or label_type != 'random':
        # identify first column as cell labels
        expected_cell_labels = np.array(cell_labels)
        expected_gene_labels = np.array(gene_labels)
        expected_x = x
    else:
        # has to treat first column as genes
        expected_cell_labels = np.arange(n_cells)
        expected_gene_labels = np.array(
            ['cell_label'] + gene_labels
        )
        expected_x = np.hstack(
            [np.array(cell_labels).reshape(n_cells, 1),
             x]
        )

    np.testing.assert_array_equal(
        adata.obs.index.values.astype(expected_cell_labels.dtype),
        expected_cell_labels
    )

    np.testing.assert_array_equal(
        adata.var.index.values,
        expected_gene_labels
    )

    np.testing.assert_allclose(
        expected_x,
        adata.X,
        atol=0.0,
        rtol=1.0e-6
    )
