"""
In this module, we provide a class MarkerArray that will serve as the
accessor for the HDF5 file of marker genes produced by

diff_exp.markers.find_markers_for_all_taxonomy_pairs

The point of this abstraction is that it will allow us to change
the backend storage of marker gene data (should we decide another model
is more efficient) without having to change the utility function that
selects marker genes for a given query dataset.
"""

import copy
import h5py
import json
import numpy as np
import pathlib

from hierarchical_mapping.binary_array.binary_array import (
    BinarizedBooleanArray)

from hierarchical_mapping.diff_exp.summarize import (
    MarkerSummary)


class MarkerGeneArray(object):
    """
    A class providing access to the marker genes for a given reference
    dataset as computed and stored by
    markers.find_markers_for_all_taxonomy_pairs

    should be instantiated with from_cache_path (usually)

    Parameters
    ----------
    cache_path:
        path to the file created by
        diff_exp.markers.find_markers_for_all_taxonomy_pairs

    only_keep_pairs:
        If not None, a list of (level, node1, node2) pairs that
        will be the only taxonomy pairs kept in this MarkerGeneArray

    is_dummy:
        if True, will skip reading data from cache_path

    __init__ Parameters
    -------------------
    gene_names:
        list of gene_names
    taxonomy_pair_to_idx:
        dict mapping [level][node1][node2] to column idx
    n_pairs:
        number of taxonomy pairs (columns) in arrays
    is_marker:
        BinarizedBooleanArray indicating marker genes
    up_regulated:
        BinarizedBooleanArray indicating up-regulation of
        markers.
    up_marker_summary:
        a diff_exp.summarize.MarkerSummary representing
        up-regulated marker genes (can be None)
    down_marker_summary:
        a diff_exp.summarize.MakerSummary representing
        down-regulated marker genes (can be None)
    """
    def __init__(
            self,
            gene_names,
            taxonomy_pair_to_idx,
            n_pairs,
            is_marker,
            up_regulated,
            up_marker_summary=None,
            down_marker_summary=None):

        valid_summary = True
        if up_marker_summary is None:
            if down_marker_summary is not None:
                valid_summary = False
        else:
            if down_marker_summary is None:
                valid_summary = False
        if not valid_summary:
            raise RuntimeError(
                "up_regulated_summary and down_regulated_summary "
                "must both be None or both be not None\n"
                f"up is None: {up_marker_summary is None}\n"
                f"down is None: {down_marker_summary is None}\n")

        self._gene_names = copy.deepcopy(gene_names)
        self.taxonomy_pair_to_idx = copy.deepcopy(
            taxonomy_pair_to_idx)
        self.n_pairs = n_pairs
        self.is_marker = is_marker
        self.up_regulated = up_regulated

        self._up_marker_summary = up_marker_summary
        self._down_marker_summary = down_marker_summary

    @classmethod
    def from_cache_path(
            cls,
            cache_path,
            only_keep_pairs=None):

        cache_path = pathlib.Path(cache_path)
        if not cache_path.is_file():
            raise RuntimeError(
                f"{cache_path} is not a file")

        with h5py.File(cache_path, "r", swmr=True) as src:
            gene_names = json.loads(
                src['gene_names'][()].decode('utf-8'))
            taxonomy_pair_to_idx = json.loads(
                src['pair_to_idx'][()].decode('utf-8'))

            if only_keep_pairs is not None:
                col_idx = np.array(
                    [_idx_of_pair(
                        taxonomy_pair_to_idx,
                        pair[0],
                        pair[1],
                        pair[2])
                     for pair in only_keep_pairs])

            n_pairs = src['n_pairs'][()]

            is_marker = BinarizedBooleanArray.from_data_array(
                data_array=src['markers/data'][()],
                n_cols=n_pairs)

            if only_keep_pairs is not None:
                is_marker.downsample_columns(col_idx)

            up_regulated = BinarizedBooleanArray.from_data_array(
                data_array=src['up_regulated/data'][()],
                n_cols=n_pairs)

            if 'summary' in src:
                up_marker_summary = MarkerSummary(
                    gene_idx=src['summary/up_gene_idx'][()],
                    pair_idx=src['summary/up_pair_idx'][()])
                if only_keep_pairs is not None:
                    up_marker_summary.keep_only_pairs(col_idx)

                down_marker_summary = MarkerSummary(
                    gene_idx=src['summary/down_gene_idx'][()],
                    pair_idx=src['summary/down_pair_idx'][()])
                if only_keep_pairs is not None:
                    down_marker_summary.keep_only_pairs(col_idx)
            else:
                up_marker_summary = None
                down_marker_summary = None

            if only_keep_pairs is not None:
                up_regulated.downsample_columns(col_idx)
                n_pairs = len(col_idx)
                taxonomy_pair_to_idx = _create_new_pair_lookup(
                    only_keep_pairs)

        return cls(
            gene_names=gene_names,
            taxonomy_pair_to_idx=taxonomy_pair_to_idx,
            n_pairs=n_pairs,
            is_marker=is_marker,
            up_regulated=up_regulated,
            up_marker_summary=up_marker_summary,
            down_marker_summary=down_marker_summary)

    def downsample_pairs_to_other(self, only_keep_pairs):
        """
        Create and return a new MarkerGeneArray, only keeping
        the specified taxonomy pairs.
        """
        col_idx = np.array(
            [_idx_of_pair(
                self.taxonomy_pair_to_idx,
                pair[0],
                pair[1],
                pair[2])
             for pair in only_keep_pairs])

        new_taxonomy_lookup = _create_new_pair_lookup(
                    only_keep_pairs)

        new_n_pairs = len(only_keep_pairs)
        new_is_marker = self.is_marker.downsample_columns_to_other(
            col_idx_array=col_idx)
        new_up = self.up_regulated.downsample_columns_to_other(
            col_idx_array=col_idx)

        if self._up_marker_summary is not None:
            new_up_s = copy.deepcopy(self._up_marker_summary)
            new_up_s.keep_only_pairs(col_idx)
            new_down_s = copy.deepcopy(self._down_marker_summary)
            new_down_s.keep_only_pairs(col_idx)
        else:
            new_up_s = None
            new_down_s = None

        return MarkerGeneArray(
            gene_names=self.gene_names,
            taxonomy_pair_to_idx=new_taxonomy_lookup,
            n_pairs=new_n_pairs,
            is_marker=new_is_marker,
            up_regulated=new_up,
            up_marker_summary=new_up_s,
            down_marker_summary=new_down_s)

    def downsample_genes(self, gene_idx_array):
        """
        Downselect to just the specified genes
        """
        self.is_marker.downsample_rows(gene_idx_array)
        self.up_regulated.downsample_rows(gene_idx_array)
        self._gene_names = [
            self._gene_names[ii]
            for ii in gene_idx_array]

        if self._up_marker_summary is not None:
            self._up_marker_summary.keep_only_genes(gene_idx_array)
            self._down_marker_summary.keep_only_genes(gene_idx_array)

    @property
    def gene_names(self):
        return self._gene_names

    @property
    def n_genes(self):
        return len(self.gene_names)

    def idx_of_pair(
            self,
            level,
            node1,
            node2):
        if node1 not in self.taxonomy_pair_to_idx[level]:
            raise RuntimeError(
                f"{node1} not under taxonomy level {level}")
        if node2 not in self.taxonomy_pair_to_idx[level][node1]:
            raise RuntimeError(
                f"({level},  {node1}, {node2})\n"
                "not a valid taxonomy pair specification; try reversing "
                "node1 and node2")

        pair_idx = self.taxonomy_pair_to_idx[level][node1][node2]
        return pair_idx

    def marker_mask_from_gene_idx(
            self,
            gene_idx):
        marker_mask = self.is_marker.get_row(i_row=gene_idx)
        up_mask = self.up_regulated.get_row(i_row=gene_idx)
        return marker_mask, up_mask

    def marker_mask_from_pair_idx(
            self,
            pair_idx):
        """
        pair_idx is the index of the taxonomy pair.

        sign is (+1, -1), indicating in which node the gene is
        upsampled.

        Returns (marker_mask, up_mask):
        (n_genes,) array of booleans indicating
                - is the gene a marker for this pair
                - for which node in the pair is the gene up-regulated
        """
        marker_mask = self.is_marker.get_col(i_col=pair_idx)
        up_mask = self.up_regulated.get_col(i_col=pair_idx)
        return (marker_mask, up_mask)

    def _up_mask_from_pair_idx_use_full(
            self,
            pair_idx):
        """
        Return (n_genes,) boolean array indicating all genes that
        are up_regulated markers for the pair

        Use full arrays (rather than summaries)
        """
        (marker_mask,
         up_mask) = self.marker_mask_from_pair_idx(
                         pair_idx=pair_idx)
        return np.logical_and(marker_mask, up_mask)

    def _up_mask_from_pair_idx_use_summary(
            self,
            pair_idx):
        idx = self._up_marker_summary.get_genes_for_pair(pair_idx)
        mask = np.zeros(self.n_genes, dtype=bool)
        mask[idx] = True
        return mask

    def up_mask_from_pair_idx(
            self,
            pair_idx):
        """
        Return (n_genes,) boolean array indicating all genes that
        are up_regulated markers for the pair
        """
        if self._up_marker_summary is not None:
            return self._up_mask_from_pair_idx_use_summary(pair_idx)
        return self._up_mask_from_pair_idx_use_full(pair_idx)

    def _down_mask_from_pair_idx_use_full(
            self,
            pair_idx):
        """
        Return (n_genes,) boolean array indicating all genes that
        are down_regulated markers for the pair

        Use full arrays (rather than summaries)
        """
        (marker_mask,
         up_mask) = self.marker_mask_from_pair_idx(
                         pair_idx=pair_idx)
        return np.logical_and(marker_mask, np.logical_not(up_mask))

    def _down_mask_from_pair_idx_use_summary(
            self,
            pair_idx):
        idx = self._down_marker_summary.get_genes_for_pair(pair_idx)
        mask = np.zeros(self.n_genes, dtype=bool)
        mask[idx] = True
        return mask

    def down_mask_from_pair_idx(
            self,
            pair_idx):
        """
        Return (n_genes,) boolean array indicating all genes that
        are wown_regulated markers for the pair
        """
        if self._down_marker_summary is not None:
            return self._down_mask_from_pair_idx_use_summary(pair_idx)
        return self._down_mask_from_pair_idx_use_full(pair_idx)


def _create_new_pair_lookup(only_keep_pairs):
    """
    Create new pair-to-idx lookup for case where we
    are only keeping the specified pairs
    """
    new_lookup = dict()
    for ii, pair in enumerate(only_keep_pairs):
        level = pair[0]
        node1 = pair[1]
        node2 = pair[2]
        if level not in new_lookup:
            new_lookup[level] = dict()
        if node1 not in new_lookup[level]:
            new_lookup[level][node1] = dict()
        new_lookup[level][node1][node2] = ii
    return new_lookup


def _idx_of_pair(
        taxonomy_pair_to_idx,
        level,
        node1,
        node2):
    if node1 not in taxonomy_pair_to_idx[level]:
        raise RuntimeError(
            f"{node1} not under taxonomy level {level}")
    if node2 not in taxonomy_pair_to_idx[level][node1]:
        raise RuntimeError(
            f"({level},  {node1}, {node2})\n"
            "not a valid taxonomy pair specification; try reversing "
            "node1 and node2")

    pair_idx = taxonomy_pair_to_idx[level][node1][node2]
    return pair_idx
