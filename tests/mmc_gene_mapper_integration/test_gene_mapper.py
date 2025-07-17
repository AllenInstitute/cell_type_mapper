import pytest


@pytest.mark.parametrize(
    "gene_list, species, authority, expected",
    [(['ENSM12', 'ENSM5', 'ENSM88', 'ENSM4567899'],
      'mouse',
      'NCBI',
      ['NCBIGene:12', 'NCBIGene:5', 'NCBIGene:88', None]),
     (['ENSM1', 'ENSM7', 'ENSM18', 'ENSM23', 'ENSM26'],
      'human',
      'ENSEMBL',
      [None, None, 'ENSG118', None, 'ENSG126']),
     (['e_mouse_symbol_1',
       'e_mouse_symbol_5',
       'e_mouse_symbol_18',
       'e_mouse_name_23',
       'e_mouse_name_26'],
      'mouse',
      'ENSEMBL',
      ['ENSM1', 'ENSM5', 'ENSM18',
       'ENSM23', 'ENSM26']),
     (['mouse_symbol_1',
       'mouse_symbol_5',
       'mouse_symbol_18',
       'mouse_symbol_23',
       'mouse_symbol_26'],
      'mouse',
      'NCBI',
      ['NCBIGene:1', 'NCBIGene:5', 'NCBIGene:18',
       'NCBIGene:23', 'NCBIGene:26'])
     ]
)
def test_mmc_gene_mapper_interface(
        mapper_fixture,
        gene_list,
        species,
        authority,
        expected):
    """
    Just test that the mmc_gene_mapper is behaving as expected
    (this might be redundant with unit tests that should be in the
    mmc_gene_mapper repo itself)
    """

    result = mapper_fixture.map_genes(
        gene_list=gene_list,
        dst_species=species,
        dst_authority=authority)

    assert len(result['gene_list']) == len(gene_list)
    assert len(result['gene_list']) == len(expected)
    for r_gene, e_gene in zip(result['gene_list'], expected):
        is_okay = True
        if e_gene is None:
            if 'UNMAPPABLE' not in r_gene:
                is_okay = False
        else:
            if r_gene != e_gene:
                is_okay = False
        if not is_okay:
            raise RuntimeError(
                "Gene mapping error\n"
                f"expected: {expected}\n"
                f"actual: {result['gene_list']}"
            )
