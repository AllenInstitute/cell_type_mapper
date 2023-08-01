"""
Defines abstract sparse marker class ('abstract' in the sense
that it is agnostic to the mapping between indptr, indices and
pairs, genes)
"""
import numpy as np

from cell_type_mapper.utils.sparse_utils import (
    downsample_indptr,
    mask_indptr_by_indices)


class SparseMarkersAbstract(object):
    """"
    Class to contain the sparse summary of the marker array

    Agnostic to which dimension (genes or pairs) is represented
    by indptr and which is represented by indices
    """
    def __init__(
           self,
           indices,
           indptr):
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)
        self.dtype = self.indices.dtype

    def keep_only_indptr(self, indptr_to_keep):
        """
        Downsample, keeping only the pairs denoted by the indexes
        in pairs_to_keep
        """
        (self.indptr,
         self.indices) = downsample_indptr(
             indptr_old=self.indptr,
             indices_old=self.indices,
             indptr_to_keep=indptr_to_keep)

    def keep_only_indices(self, indices_to_keep):
        indices_map = {
            nn: ii for ii, nn in enumerate(indices_to_keep)}

        (self.indptr,
         self.indices) = mask_indptr_by_indices(
             indptr_old=self.indptr,
             indices_old=self.indices,
             indices_map=indices_map)
        self.indices = self.indices.astype(self.dtype)

    def get_indices_for_indptr(self, indptr_idx):
        if indptr_idx >= len(self.indptr)-1:
            raise RuntimeError(
                f"{indptr_idx} is an invalid pair_idx; "
                f"len(self.indptr) = {len(self.indptr)}")
        return np.copy(
            self.indices[
                   self.indptr[indptr_idx]:self.indptr[indptr_idx+1]])

    def get_sparse_arrays_for_indptr_array(self, indptr_idx_array):

        (new_indptr,
         new_indices) = downsample_indptr(
             indptr_old=self.indptr,
             indices_old=self.indices,
             indptr_to_keep=indptr_idx_array)

        return (new_indptr, new_indices)
