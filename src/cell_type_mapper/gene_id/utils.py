import re


def is_ensembl(gene_id):
    """
    Return a boolean indicating if the specified
    gene_id is an Ensembl identifier
    """
    if not hasattr(is_ensembl, 'pattern'):
        is_ensembl.pattern = re.compile('ENS[A-Z]+[0-9]+')
    match = is_ensembl.pattern.fullmatch(gene_id)
    if match is None:
        return False
    return True
