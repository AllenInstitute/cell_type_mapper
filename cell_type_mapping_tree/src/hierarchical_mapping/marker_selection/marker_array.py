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
    """
    def __init__(
            self,
            gene_names,
            taxonomy_pair_to_idx,
            n_pairs,
            is_marker,
            up_regulated):
        self._gene_names = copy.deepcopy(gene_names)
        self.taxonomy_pair_to_idx = copy.deepcopy(
            taxonomy_pair_to_idx)
        self.n_pairs = n_pairs
        self.is_marker = is_marker
        self.up_regulated = up_regulated

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
            up_regulated=up_regulated)

    def downsample_pairs_to_other(self, only_keep_pairs):
        """
        Create and return a new MarkerGeneArray, only keeping
        the specified taxonomy pairs.
        """

    def downsample_genes(self, gene_idx_array):
        """
        Downselect to just the specified genes
        """
        self.is_marker.downsample_rows(gene_idx_array)
        self.up_regulated.downsample_rows(gene_idx_array)
        self._gene_names = [
            self._gene_names[ii]
            for ii in gene_idx_array]

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
