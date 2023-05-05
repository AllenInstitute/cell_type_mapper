"""
Utility for summarizing which genes are up- and down-
regulated for which taxon pairs in a sparse matrix manner
"""
import h5py
import numpy as np
import warnings
import time

from hierarchical_mapping.binary_array.binary_array import (
    BinarizedBooleanArray)


class MarkerSummary(object):
    """"
    Class to contain the sparse summary of the marker array

    Parameters
    ----------
    gene_idx:
        List of integers denoting marker genes
    pair_idx:
        Integers denoting where in gene_idx each
        taxon pair begins, i.e.
        gene_idx[pair_idx[1]:pair_idx[2]]
        are the marker genes for taxon_pair[1]
    """
    def __init__(
           self,
           gene_idx,
           pair_idx):
        self.gene_idx = np.array(gene_idx)
        self.pair_idx = np.array(pair_idx)

    def keep_only_pairs(self, pairs_to_keep):
        new_gene_idx = []
        new_pair_idx = []
        ct = 0
        for i_pair in pairs_to_keep:
            new_pair_idx.append(ct)

            chunk = self.gene_idx[
                self.pair_idx[i_pair]:self.pair_idx[i_pair+1]]

            ct += len(chunk)

            new_gene_idx.append(chunk)

        self.gene_idx = np.hstack(new_gene_idx)
        self.pair_idx = np.hstack([new_pair_idx, [len(self.gene_idx)]])

    def keep_only_genes(self, genes_to_keep):

        gene_mask = np.zeros(len(self.gene_idx), dtype=bool)

        for gene_value in genes_to_keep:
            valid = (self.gene_idx == gene_value)
            gene_mask[valid] = True

        self.gene_idx = self.gene_idx[gene_mask]
        new_gene_idx = np.copy(self.gene_idx)
        for new_val, old_val in enumerate(genes_to_keep):
            valid = (self.gene_idx == old_val)
            new_gene_idx[valid] = new_val
        self.gene_idx = new_gene_idx
        new_pair_idx = np.zeros(len(self.pair_idx), dtype=int)

        ct = 0
        for i_pair in range(len(self.pair_idx)-1):
            new_pair_idx[i_pair] = ct
            chunk = gene_mask[self.pair_idx[i_pair]:self.pair_idx[i_pair+1]]
            ct += chunk.sum()
        new_pair_idx[-1] = len(self.gene_idx)
        self.pair_idx = new_pair_idx

        for ii, i0 in enumerate(self.pair_idx[:-1]):
            i1 = self.pair_idx[ii+1]
            chunk = self.gene_idx[i0:i1]
            self.gene_idx[i0:i1] = np.sort(chunk)

    def get_genes_for_pair(self, pair_idx):
        if pair_idx >= len(self.pair_idx)-1:
            raise RuntimeError(
                f"{pair_idx} is an invalid pair_idx; "
                f"len(self.pair_idx) = {len(self.pair_idx)}")
        return np.copy(self.gene_idx[
                        self.pair_idx[pair_idx]:self.pair_idx[pair_idx+1]])


def add_summary_to_h5(
        marker_h5_path):
    """
    If possible, add the marker summary to the HDF5 file at the specified path
    """
    t0 = time.time()
    with h5py.File(marker_h5_path, 'r') as in_file:
        n_cols = in_file['n_pairs'][()]
        marker_array = BinarizedBooleanArray.from_data_array(
            n_cols=n_cols,
            data_array=in_file['markers/data'][()])
        up_array = BinarizedBooleanArray.from_data_array(
            n_cols=n_cols,
            data_array=in_file['up_regulated/data'][()])

    summary = summarize_from_arrays(
        marker_array=marker_array,
        up_array=up_array,
        gb_cutoff=20)

    if summary is None:
        return

    with h5py.File(marker_h5_path, 'a') as out_file:
        grp = out_file.create_group('summary')
        grp.create_dataset(
            'up_gene_idx', data=summary['up_values'])
        grp.create_dataset(
            'up_pair_idx', data=summary['up_idx'])
        grp.create_dataset(
            'down_gene_idx', data=summary['down_values'])
        grp.create_dataset(
            'down_pair_idx', data=summary['down_idx'])
    duration = time.time()-t0
    print(f"adding summary to {marker_h5_path} took "
          f"{duration:.2e} seconds")


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
