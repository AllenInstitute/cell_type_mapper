import re

from cell_type_mapper.data.mouse_gene_id_lookup import (
    mouse_gene_id_lookup)

from cell_type_mapper.data.human_gene_id_lookup import (
    human_gene_id_lookup)


def is_ensembl(gene_id):
    """
    Return a boolean indicating if the specified
    gene_id is an Ensembl identifier
    """
    if not hasattr(is_ensembl, 'pattern'):
        is_ensembl.pattern = re.compile(r'ENS[A-Z]+[0-9]+(\.[0-9]+)?')   # noqa W605
    match = is_ensembl.pattern.fullmatch(gene_id)
    if match is None:
        return False
    return True


def detect_species(gene_id_list):
    """
    Take a list of arbitrary gene identifiers (strings). Return a string
    indicating if the genes are from the 'mouse' or 'human' genome.

    Decision is as follows:

    - If any known mouse EnsemblIDs are present, return 'mouse'
    - If any known human EnsemblIDs are present, return 'human'
    - If EnsemblIDs from human and mouse are present, raise error
    - If there are more mouse than human gene symbols, return 'mouse'
    - If there are more human than mouse gene symbols, return 'human'
    - If there are no known gene symbols present, return None
    """

    # break on dots in Ensembl IDs
    clean_gene_id_list = [
        g if not g.startswith('ENS') else g.split('.')[0]
        for g in gene_id_list
    ]

    gene_set = set(clean_gene_id_list)
    census = dict()
    for species, lookup in [('mouse', mouse_gene_id_lookup),
                            ('human', human_gene_id_lookup)]:

        ens = set(lookup.values())
        symb = set(lookup.keys())
        n_ens = len(ens.intersection(gene_set))
        n_symb = len(symb.intersection(gene_set))
        census[species] = {'ens': n_ens, 'symb': n_symb}

    # check presence of EnsemblIDs
    chosen_species = None
    error_msg = ""
    for species in census:
        if census[species]['ens'] == 0:
            continue
        if chosen_species is not None:
            error_msg += (
                f"There are EnsemblIDs from {chosen_species} and {species}\n")
        chosen_species = species
    if len(error_msg) > 0:
        raise RuntimeError(
            f"{error_msg}Unclear how to choose species.")
    if chosen_species is not None:
        return chosen_species

    n_max = 0
    chosen_species = []
    for species in census:
        if census[species]['symb'] > n_max:
            chosen_species = [species]
            n_max = census[species]['symb']
        elif census[species]['symb'] == n_max and n_max > 0:
            chosen_species.append(species)

    if len(chosen_species) > 1:
        error_msg = (
            f"These species\n{chosen_species}\nall have {n_max} gene symbols "
            "present in data. Unclear how to choose species."
        )
        raise RuntimeError(error_msg)
    if len(chosen_species) == 0:
        return None
    return chosen_species[0]
