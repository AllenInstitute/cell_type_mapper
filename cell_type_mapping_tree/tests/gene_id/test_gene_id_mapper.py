import pytest

from hierarchical_mapping.gene_id.gene_id_mapper import (
    GeneIdMapper)

from hierarchical_mapping.cli.cli_log import CommandLog

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

def test_gene_id_mapper(map_data_fixture):
    """
    Test that gene_id_mapper maps genes as expected
    """
    mapper = GeneIdMapper(data=map_data_fixture)

    good = ["gene_1", "gene_0", "gene_3", "gene_1"]
    actual = mapper.map_gene_identifiers(good)
    assert actual == good

    names = ["charlie", "alice", "zachary", "mark", "robert"]
    actual = mapper.map_gene_identifiers(names)
    assert len(actual) == 5
    assert actual[0] == 'gene_3'
    assert actual[1] == 'gene_0'
    assert 'nonsense_0' in actual[2]
    assert 'nonsense_1' in actual[3]
    assert actual[4] == 'gene_1'

    # will choose 'nicknames', since they are more common than names here
    nicknames = ["alice", "hammer", "allie", "robert", "chuck", "hammer"]
    actual = mapper.map_gene_identifiers(nicknames)
    assert len(actual) == 6
    assert 'nonsense_2' in actual[0]
    assert actual[1] == 'gene_2'
    assert actual[2] == 'gene_0'
    assert 'nonsense_3' in actual[3]
    assert actual[4] == 'gene_3'
    assert actual[5] == 'gene_2'


def test_gene_id_mapper_strict(map_data_fixture):
    """
    Test that an error is raised if strict == True and
    you cannot map all gene identifiers
    """
    mapper = GeneIdMapper(data=map_data_fixture)

    good = ["gene_1", "gene_0", "gene_3", "gene_1"]
    actual = mapper.map_gene_identifiers(good, strict=True)
    assert actual == good

    names = ["charlie", "alice", "zachary", "mark", "robert"]
    with pytest.raises(RuntimeError, match="genes had no mapping"):
        mapper.map_gene_identifiers(names, strict=True)

    mapper = GeneIdMapper(data=map_data_fixture, log=CommandLog())
    names = ["charlie", "alice", "zachary", "mark", "robert"]
    with pytest.raises(RuntimeError, match="genes had no mapping"):
        mapper.map_gene_identifiers(names, strict=True)


def test_bad_gene_mapping(map_data_fixture):
    """
    Test error when cannot map genes
    """
    mapper = GeneIdMapper(data=map_data_fixture)
    names = ["zack", "tyler", "miguel"]
    with pytest.raises(RuntimeError, match="did not match any known schema"):
        mapper.map_gene_identifiers(names)

def test_from_default():
    """
    Just a smoke test to make sure that default gene mapping can load
    """
    mapper = GeneIdMapper.from_default()
    assert isinstance(mapper, GeneIdMapper)
