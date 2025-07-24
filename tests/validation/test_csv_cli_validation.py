"""
Unit tests for passing a CSV file through the validator
"""
import pytest

import anndata
import hashlib
import itertools
import json
import numpy as np
import warnings

from cell_type_mapper.utils.utils import (
    mkstemp_clean
)


from cell_type_mapper.cli.validate_h5ad import (
    ValidateH5adRunner
)


@pytest.mark.parametrize(
    "label_heading_fixture,label_type_fixture,suffix_fixture,"
    "gene_identifier_type_fixture,x_dtype_fixture,transposition_fixture",
    itertools.product(
        [True, False],
        ["big", "degenerate", "sequential", "big", "random", "string"],
        [".csv", ".csv.gz"],
        ["ensembl", "symbol"],
        ["integer", "float"],
        [True, False]
    ),
    indirect=[
        "label_heading_fixture",
        "label_type_fixture",
        "suffix_fixture",
        "gene_identifier_type_fixture",
        "x_dtype_fixture",
        "transposition_fixture"
    ]
)
def test_cli_validation_for_csv(
        label_heading_fixture,
        label_type_fixture,
        suffix_fixture,
        gene_identifier_type_fixture,
        x_dtype_fixture,
        transposition_fixture,
        csv_anndata_fixture,
        tmp_dir_fixture,
        legacy_gene_mapper_db_path_fixture):

    if label_heading_fixture:
        if label_type_fixture == "random":
            return

    if transposition_fixture:
        if label_type_fixture not in ("string", "degenerate"):
            return

    (csv_path,
     cell_labels,
     gene_labels,
     x_array) = csv_anndata_fixture

    md50 = hashlib.md5()
    with open(csv_path, 'rb') as src:
        md50.update(src.read())

    json_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.json'
    )

    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad'
    )

    config = {
        'input_path': csv_path,
        'output_json': json_path,
        'valid_h5ad_path': dst_path,
        'gene_mapping': {
            'db_path': legacy_gene_mapper_db_path_fixture
        }
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        runner = ValidateH5adRunner(args=[], input_data=config)
        runner.run()

    md51 = hashlib.md5()
    with open(csv_path, 'rb') as src:
        md51.update(src.read())
    assert md50.hexdigest() == md51.hexdigest()

    result = anndata.read_h5ad(dst_path, backed='r')
    if x_dtype_fixture == 'integer':
        np.testing.assert_array_equal(
            result.X[()],
            x_array
        )
    else:
        np.testing.assert_array_equal(
            result.X[()],
            np.round(x_array).astype(int)
        )

    if label_type_fixture != 'degenerate':
        expected_cell_labels = np.array(cell_labels).astype(str)
    else:
        expected_cell_labels = np.array(
            [json.dumps({"original_index": label, "row": ii})
             for ii, label in enumerate(cell_labels)]
        )

    np.testing.assert_array_equal(
        result.obs.index.values,
        expected_cell_labels
    )

    # 2025-07-17: var index should no longer be changed
    # by validation. We are moving gene mapping into the
    # actual cell type mapper.
    np.testing.assert_array_equal(
        result.var.index.values,
        np.array(gene_labels)
    )
