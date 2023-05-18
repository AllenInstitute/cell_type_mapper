import numpy as np


def create_utility_array(
        marker_gene_array,
        gb_size=10,
        taxonomy_mask=None):
    """
    Create an (n_genes,) array of how useful each gene is as a marker.
    Utility is just a count of how many (+/-, taxonomy_pair) combinations
    the gene is a marker for (in this case +/- indicates which node in the
    taxonomy pair the gene is up-regulated for).

    Parameters
    ----------
    marker_gene_array:
        A MarkerGeneArray
    gb_size:
        Number of gigabytes to load at a time (approximately)
    taxonomy_mask:
        if not None, a list of integers denoting which columns to
        sum utility for.

    Returns
    -------
    utility_arry:
        A numpy array of floats indicating the utility of each gene.

    marker_census:
        A numpy of ints indicating how many markers there are for
        each (taxonomy pair, sign) combination.

    Notes
    -----
    As implemented, it is assumed that the rows of the arrays in cache_path
    are genes and the columns are taxonomy pairs
    """

    is_marker = marker_gene_array.is_marker
    up_regulated = marker_gene_array.up_regulated
    n_cols = marker_gene_array.n_pairs
    n_rows = marker_gene_array.n_genes

    if taxonomy_mask is None:
        n_taxon = n_cols
    else:
        n_taxon = len(taxonomy_mask)
    marker_census = np.zeros((n_taxon, 2), dtype=int)
    utility_sum = np.zeros(is_marker.n_rows, dtype=int)

    byte_size = gb_size*1024**3
    batch_size = max(1, np.round(byte_size/(3*n_cols)).astype(int))

    for row0 in range(0, n_rows, batch_size):
        row1 = min(n_rows, row0+batch_size)
        up_reg_batch = up_regulated.get_row_batch(row0, row1)
        marker_batch = is_marker.get_row_batch(row0, row1)

        if taxonomy_mask is not None:
            marker_batch = marker_batch[:, taxonomy_mask]
            up_reg_batch = up_reg_batch[:, taxonomy_mask]

        utility_sum[row0:row1] = marker_batch.sum(axis=1)
        marker_census[:, 0] += (np.logical_not(up_reg_batch)
                                * marker_batch).sum(axis=0)
        marker_census[:, 1] += (up_reg_batch*marker_batch).sum(axis=0)

    return utility_sum, marker_census
