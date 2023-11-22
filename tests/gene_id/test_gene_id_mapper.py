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
    assert actual['mapped_genes'] == good
    assert actual['n_unmapped'] == 0

    names = ["charlie", "alice", "zachary", "mark", "robert"]
    actual = mapper.map_gene_identifiers(names)
    actual_genes = actual['mapped_genes']
    assert len(actual_genes) == 5
    assert actual_genes[0] == 'ENSG3'
    assert actual_genes[1] == 'ENSG0'
    assert 'unmapped_0' in actual_genes[2]
    assert 'unmapped_1' in actual_genes[3]
    assert actual_genes[4] == 'ENSG1'
    assert actual['n_unmapped'] == 2

    nicknames = ["alice", "hammer", "allie", "kyle", "chuck", "hammer"]
    actual = mapper.map_gene_identifiers(nicknames)
    actual_genes = actual['mapped_genes']
    assert len(actual_genes) == 6
    assert actual_genes[0] == 'ENSG0'
    assert actual_genes[1] == 'ENSG2'
    assert actual_genes[2] == 'ENSG0'
    assert 'unmapped_2' in actual_genes[3]
    assert actual_genes[4] == 'ENSG3'
    assert actual_genes[5] == 'ENSG2'
    assert actual['n_unmapped'] == 1


def test_gene_id_mapper_strict(map_data_fixture):
    """
    Test that an error is raised if strict == True and
    you cannot map all gene identifiers
    """
    mapper = GeneIdMapper(data=map_data_fixture)

    good = ["ENSG1", "ENSG0", "ENSG3", "ENSG1"]
    actual = mapper.map_gene_identifiers(good, strict=True)
    assert actual['mapped_genes'] == good
    assert actual['n_unmapped'] == 0

    names = ["charlie", "alice", "zachary", "mark", "robert"]
    with pytest.raises(RuntimeError, match="could not be mapped"):
        mapper.map_gene_identifiers(names, strict=True)

    mapper = GeneIdMapper(data=map_data_fixture, log=CommandLog())
    names = ["charlie", "alice", "zachary", "mark", "robert"]
    with pytest.raises(RuntimeError, match="could not be mapped"):
        mapper.map_gene_identifiers(names, strict=True)

def test_class_methods():
    """
    Just a smoke test to make sure that default gene mapping can load
    """
    mapper = GeneIdMapper.from_mouse()
    assert isinstance(mapper, GeneIdMapper)
    mapper = GeneIdMapper.from_human()
    assert isinstance(mapper, GeneIdMapper)


def test_is_ens():
    """
    Test that mapper can correctly identify if an ID is an
    EnsemblID
    """
    assert is_ensembl('ENSF6')
    assert is_ensembl('ENSFBBD883346')
    assert is_ensembl('ENSR00182311.9')
    assert is_ensembl('ENSG00812312.22732')
    assert not is_ensembl('ENS7')
    assert not is_ensembl('ENSGabc8899')
    assert not is_ensembl('XYENSG8812')
    assert not is_ensembl('ENS781abcd')
    assert not is_ensembl('ENSG')
    assert not is_ensembl('ENSG889123.2262a')


def test_suffix_clipping():
    data = {
        "alice": "ENSG0",
        "allie": "ENSG0",
        "robert": "ENSG1",
        "hammer": "ENSG2",
        "charlie": "ENSG3",
        "chuck": "ENSG3"
    }
    mapper = GeneIdMapper(data=data)
    input_arr = ['hammer', 'chuck', 'allie', 'ENSG555.7', 'robert']
    expected = ['ENSG2', 'ENSG3', 'ENSG0', 'ENSG555', 'ENSG1']
    actual = mapper.map_gene_identifiers(input_arr)
    assert actual['mapped_genes'] == expected
    assert actual['n_unmapped'] == 0


@pytest.mark.parametrize('strict', [True, False])
def test_when_all_bad(strict):
    """
    Make sure an error is raised when no genes can be mapped
    """
    mapper = GeneIdMapper.from_mouse()
    gene_id_list = ['garbage_a', 'garbage_b', 'garbage_c']
    msg = (
        "Could not map any of your genes to EnsemblID\n"
        "First five gene identifiers are:\n"
        "\['garbage_a', 'garbage_b', 'garbage_c'\]"
    )
    with pytest.raises(RuntimeError, match=msg):
        mapper.map_gene_identifiers(gene_id_list, strict=strict)
