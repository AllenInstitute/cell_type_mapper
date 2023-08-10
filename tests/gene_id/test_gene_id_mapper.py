import pytest

from cell_type_mapper.gene_id.utils import (
    is_ensembl)

from cell_type_mapper.gene_id.gene_id_mapper import (
    GeneIdMapper)

from cell_type_mapper.cli.cli_log import CommandLog

@pytest.fixture
def map_data_fixture():
    data = {
        "alice": "ENSG0",
        "allie": "ENSG0",
        "robert": "ENSG1",
        "hammer": "ENSG2",
        "charlie": "ENSG3",
        "chuck": "ENSG3"
    }

    return data

def test_gene_id_mapper_class(map_data_fixture):
    """
    Test that gene_id_mapper maps genes as expected
    """
    mapper = GeneIdMapper(data=map_data_fixture)

    good = ["ENSG1", "ENSG0", "ENSG3", "ENSG1"]
    actual = mapper.map_gene_identifiers(good)
    assert actual == good

    names = ["charlie", "alice", "zachary", "mark", "robert"]
    actual = mapper.map_gene_identifiers(names)
    assert len(actual) == 5
    assert actual[0] == 'ENSG3'
    assert actual[1] == 'ENSG0'
    assert 'unmapped_0' in actual[2]
    assert 'unmapped_1' in actual[3]
    assert actual[4] == 'ENSG1'

    nicknames = ["alice", "hammer", "allie", "kyle", "chuck", "hammer"]
    actual = mapper.map_gene_identifiers(nicknames)
    assert len(actual) == 6
    assert actual[0] == 'ENSG0'
    assert actual[1] == 'ENSG2'
    assert actual[2] == 'ENSG0'
    assert 'unmapped_2' in actual[3]
    assert actual[4] == 'ENSG3'
    assert actual[5] == 'ENSG2'


def test_gene_id_mapper_strict(map_data_fixture):
    """
    Test that an error is raised if strict == True and
    you cannot map all gene identifiers
    """
    mapper = GeneIdMapper(data=map_data_fixture)

    good = ["ENSG1", "ENSG0", "ENSG3", "ENSG1"]
    actual = mapper.map_gene_identifiers(good, strict=True)
    assert actual == good

    names = ["charlie", "alice", "zachary", "mark", "robert"]
    with pytest.raises(RuntimeError, match="could not be mapped"):
        mapper.map_gene_identifiers(names, strict=True)

    mapper = GeneIdMapper(data=map_data_fixture, log=CommandLog())
    names = ["charlie", "alice", "zachary", "mark", "robert"]
    with pytest.raises(RuntimeError, match="could not be mapped"):
        mapper.map_gene_identifiers(names, strict=True)

def test_from_default():
    """
    Just a smoke test to make sure that default gene mapping can load
    """
    mapper = GeneIdMapper.from_default()
    assert isinstance(mapper, GeneIdMapper)


def test_is_ens():
    """
    Test that mapper can correctly identify if an ID is an
    EnsemblID
    """
    assert is_ensembl('ENSF6')
    assert is_ensembl('ENSFBBD883346')
    assert not is_ensembl('ENS7')
    assert not is_ensembl('ENSGabc8899')
    assert not is_ensembl('XYENSG8812')
    assert not is_ensembl('ENS781abcd')
    assert not is_ensembl('ENSG')
