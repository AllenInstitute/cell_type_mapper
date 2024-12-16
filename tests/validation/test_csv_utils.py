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

from cell_type_mapper.validation.validate_h5ad import (
    _convert_csv_to_h5ad)


@pytest.mark.parametrize(
    "label_heading,label_is_numerical,suffix",
    itertools.product(
        [True, False],
        [True, False],
        ['.csv', '.csv.gz']
    )
)
def test_convert_csv(
        tmp_dir_fixture,
        label_heading,
        label_is_numerical,
        suffix):

    rng = np.random.default_rng(221111)
    n_cells = 4
    n_genes = 7
    x = rng.integers(10, 100, (n_cells, n_genes))
    gene_labels = [f'g_{ii}' for ii in range(n_genes)]
    if label_is_numerical:
        cell_labels = [ii for ii in range(n_cells)]
    else:
        cell_labels = [f'c_{ii}' for ii in range(n_cells)]

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

    h5ad_path, flag = _convert_csv_to_h5ad(
        src_path=csv_path,
        log=None)
    assert flag

    adata = anndata.read_h5ad(h5ad_path, backed='r')

    if not label_heading or not label_is_numerical:
        expected_cell_labels = np.array(cell_labels)
        expected_gene_labels = np.array(gene_labels)
        expected_x = x
    else:
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
