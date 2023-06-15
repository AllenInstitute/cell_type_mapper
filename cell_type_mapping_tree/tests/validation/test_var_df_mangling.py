import pytest

import anndata
import numpy as np
import pandas as pd
import pathlib
from unittest.mock import patch

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.gene_id.gene_id_mapper import (
    GeneIdMapper)

from hierarchical_mapping.validation.utils import (
    map_gene_ids_in_var)


@pytest.fixture(scope='module')
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('var_mangling_'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def map_data_fixture():

    data = {
        "gene_0": {
            "name": "alice",
            "nickname": "allie"
        },
        "gene_1": {
            "name": "robert"
        },
        "gene_2": {
            "nickname": "hammer"
        },
        "gene_3": {
            "name": "charlie",
            "nickname": "chuck"
        }
    }

    return data


def test_var_mapping(
        map_data_fixture):
    records = [
        {'gene_identifier': 'allie', 'garbage': 'x'},
        {'gene_identifier': 'chuck', 'garbage': 'y'},
        {'gene_identifier': 'zack', 'garbage': 'z'}
    ]

    var_orig = pd.DataFrame(records).set_index('gene_identifier')

    def new_timestamp():
        return 'xxx'

    with patch('hierarchical_mapping.utils.utils.get_timestamp', new=new_timestamp):
        gene_id_mapper = GeneIdMapper(data=map_data_fixture)
        new_var = map_gene_ids_in_var(
            var_df=var_orig,
            gene_id_mapper=gene_id_mapper)

    # make sure input did not change
    pd.testing.assert_frame_equal(
        var_orig,
        pd.DataFrame(records).set_index('gene_identifier'))

    expected_records = [
        {'EnsemblID_VALIDATED': 'gene_0',
          'gene_identifier': 'allie',
          'garbage': 'x'},
        {'EnsemblID_VALIDATED': 'gene_3',
         'gene_identifier': 'chuck',
         'garbage': 'y'},
        {'EnsemblID_VALIDATED': 'nonsense_0_xxx',
         'gene_identifier': 'zack',
         'garbage': 'z'}
    ]
    expected = pd.DataFrame(expected_records).set_index('EnsemblID_VALIDATED')
    pd.testing.assert_frame_equal(new_var, expected)


def test_var_mapping_column_name_taken(
        map_data_fixture):
    """
    Test case when the desired index column is already
    taken
    """

    records = [
        {'EnsemblID_VALIDATED': 'allie', 'garbage': 'x'},
        {'EnsemblID_VALIDATED': 'chuck', 'garbage': 'y'},
        {'EnsemblID_VALIDATED': 'zack', 'garbage': 'z'}
    ]

    var_orig = pd.DataFrame(records).set_index('EnsemblID_VALIDATED')

    def new_timestamp():
        return 'xxx'

    with patch('hierarchical_mapping.utils.utils.get_timestamp', new=new_timestamp):
        gene_id_mapper = GeneIdMapper(data=map_data_fixture)
        new_var = map_gene_ids_in_var(
            var_df=var_orig,
            gene_id_mapper=gene_id_mapper)

    # make sure input did not change
    pd.testing.assert_frame_equal(
        var_orig,
        pd.DataFrame(records).set_index('EnsemblID_VALIDATED'))

    expected_records = [
        {'EnsemblID_VALIDATED_0': 'gene_0',
          'EnsemblID_VALIDATED': 'allie',
          'garbage': 'x'},
        {'EnsemblID_VALIDATED_0': 'gene_3',
         'EnsemblID_VALIDATED': 'chuck',
         'garbage': 'y'},
        {'EnsemblID_VALIDATED_0': 'nonsense_0_xxx',
         'EnsemblID_VALIDATED': 'zack',
         'garbage': 'z'}
    ]
    expected = pd.DataFrame(expected_records).set_index('EnsemblID_VALIDATED_0')
    pd.testing.assert_frame_equal(new_var, expected)


def test_var_mapping_column_no_op(
        map_data_fixture):
    """
    Test case when nothing needs to be done
    """

    records = [
        {'gene_identifier': 'gene_0', 'garbage': 'x'},
        {'gene_identifier': 'gene_3', 'garbage': 'y'},
        {'gene_identifier': 'gene_1', 'garbage': 'z'}
    ]

    var_orig = pd.DataFrame(records).set_index('gene_identifier')

    gene_id_mapper = GeneIdMapper(data=map_data_fixture)
    new_var = map_gene_ids_in_var(
        var_df=var_orig,
        gene_id_mapper=gene_id_mapper)

    assert new_var is None
