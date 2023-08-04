import pytest

from cell_type_mapper.marker_lookup.marker_lookup import (
    map_aibs_gene_names)



def test_aibs_gene_name_mapper_fn():

    gene_symbols = [
        "Gm38336",
        "Fam168b",
        "Arhgef4 ENSMUSG00000118272"]
    mapping = map_aibs_gene_names(gene_symbols)
    assert len(mapping) == len(gene_symbols)
    assert mapping["Gm38336"] == "ENSMUSG00000104002"
    assert mapping["Fam168b"] == "ENSMUSG00000037503"
    assert mapping["Arhgef4 ENSMUSG00000118272"] == "ENSMUSG00000118272"

    gene_symbols = [
        "Gm38336",
        "Arhgef4",
        "Fam168b",
        "Arhgef4 ENSMUSG00000118272"]
    mapping = map_aibs_gene_names(gene_symbols)
    assert len(mapping) == len(gene_symbols)
    assert mapping["Gm38336"] == "ENSMUSG00000104002"
    assert mapping["Fam168b"] == "ENSMUSG00000037503"
    assert mapping["Arhgef4 ENSMUSG00000118272"] == "ENSMUSG00000118272"
    assert mapping["Arhgef4"] == "ENSMUSG00000037509"


    gene_symbols = [
        "Gm38336",
        "Arhgef4",
        "Fam168b",
        "Arhgef4 ENSMUSG00000037509"]
    mapping = map_aibs_gene_names(gene_symbols)
    assert len(mapping) == len(gene_symbols)
    assert mapping["Gm38336"] == "ENSMUSG00000104002"
    assert mapping["Fam168b"] == "ENSMUSG00000037503"
    assert mapping["Arhgef4"] == "ENSMUSG00000118272"
    assert mapping["Arhgef4 ENSMUSG00000037509"] == "ENSMUSG00000037509"

    gene_symbols = [
        "Gm38336",
        "Ccl21c",
        "Fam168b"]
    # will fail because Ccl21c is not in gen_id_lookup
    with pytest.raises(RuntimeError, match="Too many possible Ensembl"):
        map_aibs_gene_names(gene_symbols)

    gene_symbols = [
        "Gm38336",
        "Ccl21c",
        "Fam168b",
        "Ccl21c ENSMUSG00000096271"]
    mapping = map_aibs_gene_names(gene_symbols)
    assert len(mapping) == len(gene_symbols)
    assert mapping["Gm38336"] == "ENSMUSG00000104002"
    assert mapping["Fam168b"] == "ENSMUSG00000037503"
    assert mapping["Ccl21c ENSMUSG00000096271"] == "ENSMUSG00000096271"
    assert mapping["Ccl21c"] == "ENSMUSG00000096873"

    gene_symbols = [
        "Gm38336",
        "Ccl21c",
        "Fam168b",
        "Ccl21c ENSMUSG00000096271",
        "blah"]
    with pytest.raises(RuntimeError, match="Could not find Ensembl"):
        map_aibs_gene_names(gene_symbols)

    # Ccl27a has 3 possible IDs
    gene_symbols = [
        "Gm38336",
        "Ccl27a",
        "Ccl27a ENSMUSG00000095247"]
    with pytest.raises(RuntimeError, match="Too many possible Ensembl"):
        map_aibs_gene_names(gene_symbols)

    gene_symbols = [
        "Gm38336",
        "Ccl27a",
        "Ccl27a ENSMUSG00000095247",
        "Ccl27a ENSMUSG00000073888"]
    mapping = map_aibs_gene_names(gene_symbols)
    assert len(mapping) == len(gene_symbols)
    assert mapping["Gm38336"] == "ENSMUSG00000104002"
    assert mapping["Ccl27a"] == "ENSMUSG00000093828"
    assert mapping["Ccl27a ENSMUSG00000095247"] == "ENSMUSG00000095247"
    assert mapping["Ccl27a ENSMUSG00000073888"] == "ENSMUSG00000073888"

    gene_symbols = [
        "Gm38336",
        "Ccl27a ENSMUSG00000093828",
        "Ccl27a ENSMUSG00000095247",
        "Ccl27a ENSMUSG00000073888",
        "Ccl127a"]
    with pytest.raises(RuntimeError, match="Could not find Ensembl"):
        map_aibs_gene_names(gene_symbols)
