import scipy.sparse as scipy_sparse


def summary_stats_for_chunk(
        cell_x_gene: scipy_sparse.csc_array) -> dict:
    """
    Take a cell (rows) by gene (columns) expression
    matrix and return summary statistics needed to aggregate
    chunks for differential gene search.

    Parameters
    ----------
    cell_x_gene: scipy_sparse.csc_array
        rows are cells; columns are genes

    Returns
    -------
    A dict of summary stats
        'sum' is a (n_genes,) array summing the gene expression
        'sumsq' is a (n_genes,) array summy the square of gene expression
        'gt0' is a (n_genes,) mask indicating how many cells at expression > 0
        'gt1' is a (n_genes,) mask indicating how many cells had expression > 1
    """
    result = dict()
    result['sum'] = cell_x_gene.sum(axis=0)
    result['sumsq'] = (cell_x_gene**2).sum(axis=0)
    result['gt0'] = (cell_x_gene > 0).sum(axis=0)
    result['gt1'] = (cell_x_gene > 1).sum(axis=0)
    return result
