"""
In this module, we provide a class MarkerArray that will serve as the
accessor for the HDF5 file of marker genes produced by

diff_exp.markers.find_markers_for_all_taxonomy_pairs

The point of this abstraction is that it will allow us to change
the backend storage of marker gene data (should we decide another model
is more efficient) without having to change the utility function that
selects marker genes for a given query dataset.
"""

import h5py
import json
import pathlib

from hierarchical_mapping.binary_array.backed_binary_array import (
    BackedBinarizedBooleanArray)

from hierarchical_mapping.binary_array.binary_array import (
    BinarizedBooleanArray)


class MarkerGeneArray(object):
    """
    A class providing access to the marker genes for a given reference
    dataset as computed and stored by
    markers.find_markers_for_all_taxonomy_pairs

    Parameters
    ----------
    cache_path:
        path to the file created by
        diff_exp.markers.find_markers_for_all_taxonomy_pairs
    """
    def __init__(self, cache_path):
        self.cache_path = pathlib.Path(cache_path)
        if not self.cache_path.is_file():
            raise RuntimeError(
                f"{self.cache_path} is not a file")

        with h5py.File(self.cache_path, "r", swmr=True) as src:
            self._gene_names = json.loads(
                src['gene_names'][()].decode('utf-8'))
            self.taxonomy_pair_to_idx = json.loads(
                src['pair_to_idx'][()].decode('utf-8'))
            self.n_pairs = src['n_pairs'][()]
            self.is_marker = BinarizedBooleanArray.from_data_array(
                data_array=src['markers/data'][()],
                n_cols=self.n_pairs)
            self.up_regulated = BinarizedBooleanArray.from_data_array(
                data_array=src['up_regulated/data'][()],
                n_cols=self.n_pairs)

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
