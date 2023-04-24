"""
In this module, we provide a class MarkerArray that will serve as the
accessor for the HDF5 file of marker genes produced by

markers.find_markers_for_all_taxonomy_pairs

The point of this abstraction is that it will allow us to change
the backend storage of marker gene data (should we decide another model
is more efficient) without having to change the utility function that
selects marker genes for a given query dataset.
"""

import h5py
import json
import numpy as np
import pathlib

from hierarchical_mapping.binary_array.backed_binary_array import (
    BackedBinarizedBooleanArray)


class MarkerGeneArray(object):
    """
    A class providing access to the marker genes for a given reference
    dataset as computed and stored by
    markers.find_markers_for_all_taxonomy_pairs

    Parameters
    ----------
    cache_path:
        path to the file created by markers.find_markers_for_all_taxonomy_pairs
    """
    def __init__(self, cache_path):
        self.cache_path = pathlib.Path(cache_path)
        if not self.cache_path.is_file():
            raise RuntimeError(
                f"{self.cache_path} is not a file")

        with h5py.File(self.cache_path, "r", swmr=True) as src:
            self.gene_names = json.loads(
                src['gene_names'][()].decode('utf-8'))
            self.taxonomy_pair_to_idx = json.loads(
                src['pair_to_idx'][()].decode('utf-8'))
            self.n_pairs = src['n_pairs'][()]

        self.is_marker = BackedBinarizedBooleanArray(
            h5_path=self.cache_path,
            h5_group='markers/data',
            n_rows=len(self.gene_names),
            n_cols=self.n_pairs,
            read_only=True)

        self.up_regulated = BackedBinarizedBooleanArray(
            h5_path=self.cache_path,
            h5_group='up_regulated/data',
            n_rows=len(self.gene_names),
            n_cols=self.n_pairs,
            read_only=True)

    def gene_idx_from_utility(
            self,
            level,
            node1,
            node2,
            sign):
        """
        Get an array of gene indexes that match the utility
        specification.

        Parameters
        ----------
        level:
            The taxonomy level of node1 and node 2
        node1, node2:
            The two taxonomy pairs we are looking for markers between
        sign:
            if +1, then return marker genes that are up-regulated in
            node1; if -1, then return marker genes that are up-regulated
            in node 2.

        Returns
        -------
        Numpy array of numerical indexes of genes that fit the utility
        criterion
        """
        if sign not in (1, -1):
            raise RuntimeError(
                f"Do not know how to interpret sign={sign}\n"
                "must be (+1 or -1)")

        if node1 not in self.taxonomy_pair_to_idx[level]:
            raise RuntimeError(
                f"{node1} not under taxonomy level {level}")
        if node2 not in self.taxnomy_pair_to_idx[level][node1]:
            raise RuntimeError(
                f"({level},  {node1}, {node2})\n"
                "not a valid taxonomy pair specifcation; try reversing "
                "node1 and node2")

        pair_idx = self.taxonomy_pair_to_idx[level][node1][node2]
        marker_mask = self.is_marker.get_col(i_col=pair_idx)
        up_mask = self.up_regulated.get_col(i_col=pair_idx)
        if sign < 0:
            up_mask = np.logical_not(up_mask)
        full_mask = np.logical_and(marker_mask, up_mask)
        return np.where(full_mask)[0]
