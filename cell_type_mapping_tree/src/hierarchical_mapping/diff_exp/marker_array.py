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

        self.gene_utility = create_usefulness_array(
            cache_path=self.cache_path,
            gb_size=10)


def create_usefulness_array(
        cache_path,
        gb_size=10):
    """
    Create an (n_genes,) array of how useful each gene is as a marker.
    Usefulness is just a count of how many (+/-, taxonomy_pair) combinations
    the gene is a marker for (in this case +/- indicates which node in the
    taxonomy pair the gene is up-regulated for).

    Parameters
    ----------
    cache_path:
        path to the file created by markers.find_markers_for_all_taxonomy_pairs
    gb_size:
        Number of gigabytes to load at a time (approximately)

    Returns
    -------
    A numpy array of ints indicating the usefulness of each gene.

    Notes
    -----
    As implemented, it is assumed that the rows of the arrays in cache_path
    are genes and the columns are taxonomy pairs
    """

    with h5py.File(cache_path, "r", swmr=True) as src:
        n_cols = src['n_pairs'][()]
        n_rows = len(json.loads(src['gene_names'][()].decode('utf-8')))

    is_marker = BackedBinarizedBooleanArray(
        h5_path=cache_path,
        h5_group='markers',
        n_rows=n_rows,
        n_cols=n_cols,
        read_only=True)

    usefulness_sum = np.zeros(is_marker.n_rows, dtype=int)

    byte_size = gb_size*1024**3
    batch_size = max(1, np.round(byte_size/n_cols).astype(int))

    for row0 in range(0, n_rows, batch_size):
        row1 = min(n_rows, row0+batch_size)
        row_batch = is_marker.get_row_batch(row0, row1)
        usefulness_sum[row0:row1] = row_batch.sum(axis=1)

    return usefulness_sum
