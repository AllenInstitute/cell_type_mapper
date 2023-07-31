import numpy as np

from cell_type_mapper.diff_exp.sparse_markers import (
    SparseMarkersAbstract)


class SparseMarkersByGene(SparseMarkersAbstract):
    """"
    Class to contain the sparse summary of the marker array

    Arrays are stored to optimize access by gene index

    Parameters
    ----------
    pair_idx:
        List of integers denoting marker genes
    gene_idx:
        Integers denoting where in gene_idx each
        taxon pair begins, i.e.
        pair_idx[gene_idx[1]:gene_idx[2]]
        are the pairs for which the gene[gene_idx[1]] is a
        marker
    """
    def __init__(
           self,
           gene_idx,
           pair_idx):
        super().__init__(
            indices=np.copy(pair_idx),
            indptr=np.copy(gene_idx))

    @property
    def gene_idx(self):
        return self.indptr

    @property
    def pair_idx(self):
        return self.indices

    def keep_only_genes(self, genes_to_keep, in_place=True):
        """
        Downsample, keeping only the genes denoted by the indexes
        in genes_to_keep
        """
        if in_place:
            self.keep_only_indptr(indptr_to_keep=genes_to_keep)
            return None
        else:
            other = SparseMarkersByGene(
                gene_idx=self.gene_idx,
                pair_idx=self.pair_idx)
            other.keep_only_indptr(indptr_to_keep=genes_to_keep)
            return other

    def keep_only_pairs(self, pairs_to_keep, in_place=True):
        """
        This will work by creating a map between old pair idx and
        new pair idx. This is done because downsampling the sparse
        matrix is too expensive.
        """
        if in_place:
            self.keep_only_indices(
                indices_to_keep=pairs_to_keep)
            return None
        else:
            other = SparseMarkersByGene(
                gene_idx=self.gene_idx,
                pair_idx=self.pair_idx)
            other.keep_only_indices(
                indices_to_keep=pairs_to_keep)
            return other

    def get_pairs_for_gene(self, gene_idx):
        return self.get_indices_for_indptr(indptr_idx=gene_idx)

    def get_sparse_pairs_for_gene_array(self, gene_idx_array):
        """
        Take an array of gene indices and return the gene_idx, pair_idx
        (a la sparse matrices indptr, indices) for that group of taxon
        pairs.

        Returns new_genes, new_pairs
        """
        return self.get_sparse_arrays_for_indptr_array(
            indptr_idx_array=gene_idx_array)
