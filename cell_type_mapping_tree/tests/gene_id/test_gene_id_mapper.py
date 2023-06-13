import pytest

from hierarchical_mapping.gene_id.gene_id_mapper import (
    GeneIdMapper)


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
    expected = ["gene_3", "gene_0", "nonsense_0", "nonsense_1", "gene_1"]
    assert actual == expected

    # will choose 'nicknames', since they are more common than names here
    nicknames = ["alice", "hammer", "allie", "robert", "chuck", "hammer"]
    actual = mapper.map_gene_identifiers(nicknames)
    expected = ["nonsense_2", "gene_2", "gene_0", "nonsense_3", "gene_3",
                "gene_2"]
    assert actual == expected
