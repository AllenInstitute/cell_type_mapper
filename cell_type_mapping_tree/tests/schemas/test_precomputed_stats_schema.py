import pytest

import argschema
from marshmallow import ValidationError
import pathlib

from hierarchical_mapping.utils.utils import (
    _clean_up,
    mkstemp_clean)

from hierarchical_mapping.cli.schemas import (
    PrecomputedStatsSchema)


@pytest.fixture(scope='module')
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('stats_schema'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def ref_path_fixture(tmp_dir_fixture):
    pth = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')
    with open(pth, 'wb') as out_file:
        out_file.write(b'123')
    return pth


@pytest.fixture
def stats_path_fixture(tmp_dir_fixture):
    pth = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5')
    with open(pth, 'wb') as out_file:
        out_file.write(b'456')
    return pth


@pytest.fixture
def taxonomy_path_fixture(tmp_dir_fixture):
    pth = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.json')
    with open(pth, 'wb') as out_file:
        out_file.write(b'789')
    return pth


@pytest.mark.parametrize('norm', ['raw', 'log2CPM'])
def test_correct_stats_schema(
        ref_path_fixture,
        taxonomy_path_fixture,
        norm,
        tmp_dir_fixture):

    out_path = tmp_dir_fixture / 'garbage.txt'
    assert not out_path.exists()

    config = {'path': str(out_path.resolve().absolute()),
              'reference_path': ref_path_fixture,
              'taxonomy_tree': taxonomy_path_fixture,
              'normalization': norm}

    argschema.ArgSchemaParser(
        input_data=config,
        schema_type=PrecomputedStatsSchema,
        args=[])


def test_cannot_write_stats(
        ref_path_fixture,
        taxonomy_path_fixture,
        tmp_dir_fixture):

    out_path = tmp_dir_fixture / 'non_existent'/ 'garbage.txt'
    assert not out_path.exists()
    assert not out_path.parent.is_dir()

    config = {'path': str(out_path.resolve().absolute()),
              'reference_path': ref_path_fixture,
              'taxonomy_tree': taxonomy_path_fixture,
              'normalization': 'raw'}

    with pytest.raises(ValidationError, match="will not be able to write"):
        argschema.ArgSchemaParser(
            input_data=config,
            schema_type=PrecomputedStatsSchema,
            args=[])


def test_correct_stats_schema_preexisting(
        stats_path_fixture):

    config = {'path': stats_path_fixture}
    argschema.ArgSchemaParser(
        input_data=config,
        schema_type=PrecomputedStatsSchema,
        args=[])


@pytest.mark.parametrize(
    'proper_ref, proper_taxonomy, proper_norm',
    [(False, True, True),
     (False, False, True),
     (False, False, False),
     (False, True, False),
     (True, False, True),
     (True, False, False),
     (True, True, False)
    ])
def test_stats_schema_errors(
        tmp_dir_fixture,
        ref_path_fixture,
        taxonomy_path_fixture,
        proper_ref,
        proper_taxonomy,
        proper_norm):

    out_path = tmp_dir_fixture / 'garbage.txt'
    assert not out_path.exists()

    config = dict()
    config['path'] = out_path
    if proper_ref:
        config['reference_path'] = ref_path_fixture
    else:
        config['reference_path'] = 'nope.h5ad'

    if proper_taxonomy:
        config['taxonomy_tree'] = taxonomy_path_fixture
    else:
        config['taxonomy_tree'] = 'zilch.json'

    if proper_norm:
        config['normalization'] = 'raw'
    else:
        config['normalization'] = 'uh uh'

    with pytest.raises(ValidationError):
        argschema.ArgSchemaParser(
            input_data=config,
            schema_type=PrecomputedStatsSchema,
            args=[])


@pytest.mark.parametrize(
    'proper_ref, proper_taxonomy, proper_norm',
    [(False, True, True),
     (False, False, True),
     (False, False, False),
     (False, True, False),
     (True, False, True),
     (True, False, False),
     (True, True, False)
    ])
def test_stats_schema_errors_from_none(
        tmp_dir_fixture,
        ref_path_fixture,
        taxonomy_path_fixture,
        proper_ref,
        proper_taxonomy,
        proper_norm):

    out_path = tmp_dir_fixture / 'garbage.txt'
    assert not out_path.exists()

    config = dict()
    config['path'] = out_path
    if proper_ref:
        config['reference_path'] = ref_path_fixture
    else:
        config['reference_path'] = None

    if proper_taxonomy:
        config['taxonomy_tree'] = taxonomy_path_fixture
    else:
        config['taxonomy_tree'] = None

    if proper_norm:
        config['normalization'] = None
    else:
        config['normalization'] = 'uh uh'

    with pytest.raises(ValidationError):
        argschema.ArgSchemaParser(
            input_data=config,
            schema_type=PrecomputedStatsSchema,
            args=[])
