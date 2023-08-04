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
    # will fail because Arhgef4 is specified in the main gene_id_lookup data
    with pytest.raises(RuntimeError, match="more than one gene symbol"):
        map_aibs_gene_names(gene_symbols)

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
