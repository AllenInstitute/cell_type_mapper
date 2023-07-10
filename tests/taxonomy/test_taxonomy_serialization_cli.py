import pytest

import anndata
import copy
import json
import numpy as np
import pandas as pd
import pathlib

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up,
    json_clean_dict)

from cell_type_mapper.cli.serialize_taxonomy_tree import (
    TaxonomySerializationRunner)


@pytest.fixture(scope='module')
def tmp_dir_fixture(tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('taxonomy_serialization_'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def anndata_fixture(records_fixture, tmp_dir_fixture):
    obs = pd.DataFrame(records_fixture)
    x = np.random.random_sample((len(records_fixture), 5))
    h5ad_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.h5ad')
    a_data = anndata.AnnData(X=x, obs=obs)
    a_data.write_h5ad(h5ad_path)
    return h5ad_path

def test_serialization_cli(
        anndata_fixture,
        column_hierarchy,
        taxonomy_tree_fixture,
        tmp_dir_fixture):
    out_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.json')
    config = dict()
    config['h5ad_path'] = anndata_fixture
    config['column_hierarchy'] = column_hierarchy
    config['output_path'] = out_path

    runner = TaxonomySerializationRunner(
        args=[],
        input_data=config)
    runner.run()

    test_tree = json.load(open(out_path, 'rb'))
    expected = copy.deepcopy(taxonomy_tree_fixture._data)
    for k in ('metadata', 'alias_mapping'):
        if k in test_tree:
            test_tree.pop(k)
    assert test_tree == json_clean_dict(expected)
