"""
This is where we will test the utility functions associated with accepting
CSV files as inputs to the MapMyCells validator
"""

import pytest

import anndata
import gzip
import itertools
import numpy as np
import warnings

from cell_type_mapper.utils.utils import (
    mkstemp_clean
)

from cell_type_mapper.validation.csv_utils import (
    is_first_header_column_blank,
    is_first_column_str,
    is_first_column_sequential,
    is_first_column_floats,
    is_first_column_large,
    is_first_column_label,
    convert_csv_to_h5ad
)


@pytest.mark.parametrize(
    "first_col, expected",
    [('', True),
     ('""', True),
     ("''", True),
     (" ", True),
     ("'", True),
     ('"', True),
     ('" "', True),
     ("' '", True),
     (" '' ", True),
     (' "" ', True),
     ('a', False),
     (1, False)
     ]
)
def test_is_first_header_column_blank(first_col, expected):
    row = str(first_col) + ',a,b,c'
    if expected:
        assert is_first_header_column_blank(row)
    else:
        assert not is_first_header_column_blank(row)


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


def test_is_first_column_floats():
    """
    Test utility to detect if the first column in a numpy
    array is floats
    """
    xx = np.array(
        [[1.0, 2.1, 3.2],
         [4.0, 5.2, 4.6],
         [17.0, 1.1, 2.2]]
    )
    assert not is_first_column_floats(xx)

    xx = np.array(
        [[1.0, 2.1, 3.2],
         [4.05, 5.2, 4.6],
         [17.0, 1.1, 2.2]]
    )
    assert is_first_column_floats(xx)

    xx = np.array(
        [[1, 2, 3],
         [4, 5, 4],
         [17, 1, 2]],
        dtype=np.int32
    )
    assert not is_first_column_floats(xx)


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
    "label_heading_fixture,label_type_fixture,suffix_fixture",
    itertools.product(
        [True, False],
        ['string',
         'sequential',
         'big',
         'random',
         'sequential_float',
         'big_float',
         'hybrid'],
        ['.csv', '.csv.gz']
    ),
    indirect=['label_heading_fixture',
              'label_type_fixture',
              'suffix_fixture']
)
def test_detection_of_cell_label_column(
        label_heading_fixture,
        label_type_fixture,
        suffix_fixture,
        csv_anndata_fixture):
    """
    Test function that detects whether or not the first
    column of a CSV is cell_labels, or a gene
    """
    label_heading = label_heading_fixture
    label_type = label_type_fixture

    (csv_path,
     _,
     _,
     _) = csv_anndata_fixture

    expected = False

    if not label_heading:
        expected = True

    if label_type in ('string', 'sequential', 'big', 'big_float', 'hybrid'):
        expected = True

    if expected:
        assert is_first_column_label(csv_path)
    else:
        assert not is_first_column_label(csv_path)


@pytest.mark.parametrize(
    "label_heading_fixture,label_type_fixture,suffix_fixture",
    itertools.product(
        [True, False],
        ['string',
         'sequential',
         'big',
         'random',
         'sequential_float',
         'big_float',
         'hybrid'],
        ['.csv', '.csv.gz']
    ),
    indirect=['label_heading_fixture',
              'label_type_fixture',
              'suffix_fixture']
)
def test_convert_csv(
        label_heading_fixture,
        label_type_fixture,
        suffix_fixture,
        csv_anndata_fixture):

    label_heading = label_heading_fixture
    label_type = label_type_fixture

    (csv_path,
     cell_labels,
     gene_labels,
     x) = csv_anndata_fixture

    n_cells = len(cell_labels)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        h5ad_path, flag = convert_csv_to_h5ad(
            src_path=csv_path,
            log=None)

    assert flag

    adata = anndata.read_h5ad(h5ad_path, backed='r')

    if not label_heading or label_type not in (
                                     'random',
                                     'sequential_float'):
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


def test_csv_conversion_with_string_in_expression(
        tmp_dir_fixture):
    """
    Test that CSV conversion fails as expected if there is a string
    in one of the gene expression value slots
    """
    csv_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='csv_with_string_gene_expression_',
        suffix='.csv'
    )
    with open(csv_path, 'w') as dst:
        dst.write(',g0,g1,g2\n')
        dst.write('c0,0.1,silly,0.2\n')
        dst.write('c1,0.5,0.3,1.2\n')
    msg = "could not convert string to float: 'silly'"

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(RuntimeError, match=msg):
            convert_csv_to_h5ad(
                src_path=csv_path,
                log=None)


@pytest.mark.parametrize('use_gzip', [True, False])
def test_csv_conversion_with_bad_txt_file(
        tmp_dir_fixture,
        use_gzip):
    """
    Test that CSV conversion fails as expected if the CSV file
    does not contain a table as expected
    """
    if use_gzip:
        suffix = '.gz'
        open_fn = gzip.open
    else:
        suffix = '.csv'
        open_fn = open

    csv_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='csv_with_string_gene_expression_',
        suffix=suffix
    )
    with open_fn(csv_path, 'w') as dst:
        if use_gzip:
            dst.write(b'just some text')
        else:
            dst.write('just some text')

    msg = "An error occurred when reading your CSV with anndata"

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(RuntimeError, match=msg):
            convert_csv_to_h5ad(
                src_path=csv_path,
                log=None)


@pytest.mark.parametrize(
    'compression,expected',
    [(True, True),
     (False, True),
     (True, False),
     (False, False)
     ]
)
def test_is_first_column_str(
        tmp_dir_fixture,
        compression,
        expected):
    if compression:
        csv_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.csv.gz'
        )
        open_fn = gzip.open
    else:
        csv_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.csv'
        )
        open_fn = open

    lines = [','.join(('a', 'b', 'c')) + '\n']
    if expected:
        for ii in range(3):
            lines.append('label,' + ','.join(('1', '2')) + '\n')
    else:
        for ii in range(3):
            lines.append(','.join(('1', '2', '3')) + '\n')

    with open_fn(csv_path, 'w') as dst:
        for this in lines:
            if compression:
                this = this.encode('utf-8')
            dst.write(this)

    if expected:
        assert is_first_column_str(csv_path)
    else:
        assert not is_first_column_str(csv_path)
