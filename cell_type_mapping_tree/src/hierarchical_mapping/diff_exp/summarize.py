"""
Utility for summarizing which genes are up- and down-
regulated for which taxon pairs in a sparse matrix manner
"""
import numpy as np
import warnings


def summarize_from_arrays(
        marker_array,
        up_array,
        gb_cutoff=15):
    """
    Return a sparse representation of marker genes.

    Parameters
    ----------
    marker_array:
        BinarizedBooleanArray representing
        whether or not genes are markers
        (n_genes, n_pairs)
    up_array:
        BinarizedBooleanArray representing
        if the markers are up-regulated in
        the pair
        (n_genes, n_pairs)
    gb_cutoff:
        Size in gigabytes at which the summary
        becomes too large to be worth calculating

    Returns
    -------
    A dict
        {
        'up_values': values of gene_idx for up-regulated markers
        'up_idx': indexes in up_values at which each column begins
        'down_values': ditto for down-regulated markers
        'down_idx':....
        }
    """
    n_rows = marker_array.n_rows
    n_cols = marker_array.n_cols

    if up_array.n_rows != n_rows or up_array.n_cols != n_cols:
        raise RuntimeError(
            "Shape mismatch\n"
            f"marker: ({marker_array.n_rows}, {marker_array.n_cols})\n"
            f"up: ({up_array.n_rows}, {up_array.n_cols})\n")

    summary_specs = can_we_summarize(
        marker_array=marker_array,
        up_array=up_array,
        gb_cutoff=gb_cutoff)

    if summary_specs is None:
        return None

    gene_dtype = summary_specs['gene_dtype']
    up_col_sum = summary_specs['up_col_sum']
    down_col_sum = summary_specs['down_col_sum']

    up_start_idx = np.hstack([[0], np.cumsum(up_col_sum)])
    down_start_idx = np.hstack([[0], np.cumsum(down_col_sum)])
    up_values = np.zeros(up_col_sum.sum(), dtype=gene_dtype)
    down_values = np.zeros(down_col_sum.sum(), dtype=gene_dtype)

    up_next = np.copy(up_start_idx[:n_cols])
    down_next = np.copy(down_start_idx[:n_cols])

    for i_row in range(n_rows):
        marker_row = marker_array.get_row(i_row)
        up_row = up_array.get_row(i_row)

        up_mask = np.where(
            np.logical_and(
                marker_row,
                up_row))[0]

        down_mask = np.where(
            np.logical_and(
                marker_row,
                np.logical_not(up_row)))[0]

        to_set = up_next[up_mask]
        up_values[to_set] = i_row
        up_next[up_mask] += 1

        to_set = down_next[down_mask]
        down_values[to_set] = i_row
        down_next[down_mask] += 1

    return {
        'up_values': up_values,
        'up_idx': up_start_idx,
        'down_values': down_values,
        'down_idx': down_start_idx}


def can_we_summarize(
        marker_array,
        up_array,
        gb_cutoff=15):
    """
    Return dict with gene dtype and number of markers per column.
    Return None if estimated memory footprint is too large
    """

    n_rows = marker_array.n_rows
    n_cols = marker_array.n_cols
    raw_gene_bits = np.ceil(np.log2(n_rows)).astype(int)
    gene_bits = None
    for ii in (8, 16, 32):
        if raw_gene_bits < ii:
            gene_bits = ii
            break
    if gene_bits is None:
        gene_bits = 64
    gene_dtype = {
        8: np.uint8,
        16: np.uint16,
        32: np.uint32,
        64: np.uint64}[gene_bits]

    up_col_sum = np.zeros(n_cols, dtype=int)
    down_col_sum = np.zeros(n_cols, dtype=int)

    for i_row in range(marker_array.n_rows):
        marker_row = marker_array.get_row(i_row)
        up_row = up_array.get_row(i_row)
        up_mask = np.logical_and(marker_row, up_row)
        up_col_sum[up_mask] += 1
        down_mask = np.logical_and(marker_row, np.logical_not(up_row))
        down_col_sum[down_mask] += 1

    n_markers = up_col_sum.sum()+down_col_sum.sum()

    estimated_gb = (gene_bits*n_markers)/(8*1024**3)

    if estimated_gb > gb_cutoff:
        warnings.warn(
            f"Estimated to need {estimated_gb} gigabytes for "
            f"marker summary; cutoff = {gb_cutoff}; skipping")
        return None

    return {
        'gene_dtype': gene_dtype,
        'up_col_sum': up_col_sum,
        'down_col_sum': down_col_sum}
