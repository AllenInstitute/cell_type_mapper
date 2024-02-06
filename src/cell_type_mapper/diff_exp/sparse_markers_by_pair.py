"""
Utility for summarizing which genes are up- and down-
regulated for which taxon pairs in a sparse matrix manner
"""
import numpy as np

from cell_type_mapper.diff_exp.sparse_markers import (
    SparseMarkersAbstract)


class SparseMarkersByPair(SparseMarkersAbstract):
    """"
    Class to contain the sparse summary of the marker array

    Arrays are stored to optimize access by pair index

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
        super().__init__(
            indices=gene_idx,
            indptr=pair_idx)

    @property
    def gene_idx(self):
        return self.indices

    @property
    def pair_idx(self):
        return self.indptr

    def keep_only_pairs(self, pairs_to_keep, in_place=True):
        """
        Downsample, keeping only the pairs denoted by the indexes
        in pairs_to_keep
        """
        if in_place:
            self.keep_only_indptr(indptr_to_keep=pairs_to_keep)
            return None
        else:
            other = SparseMarkersByPair(
                pair_idx=np.copy(self.pair_idx),
                gene_idx=np.copy(self.gene_idx))
            other.keep_only_indptr(indptr_to_keep=pairs_to_keep)
            return other

    def keep_only_genes(self, genes_to_keep, in_place=True):
        """
        This will work by creating a map between old gene idx and
        new gene idx. This is done because downsampling the sparse
        matrix is too expensive.
        """
        if in_place:
            self.keep_only_indices(
                indices_to_keep=genes_to_keep)
            return None
        else:
            other = SparseMarkersByPair(
                pair_idx=np.copy(self.pair_idx),
                gene_idx=np.copy(self.gene_idx))
            other.keep_only_indices(
                indices_to_keep=genes_to_keep)
            return other

    def get_genes_for_pair(self, pair_idx):
        return self.get_indices_for_indptr(indptr_idx=pair_idx)

    def get_sparse_genes_for_pair_array(self, pair_idx_array):
        """
        Take an array of pair indices and return the pair_idx, gene_idx
        (a la sparse matrices indptr, indices) for that group of taxon
        pairs.

        Returns new_pairs, new_genes
        """
        return self.get_sparse_arrays_for_indptr_array(
            indptr_idx_array=pair_idx_array)
