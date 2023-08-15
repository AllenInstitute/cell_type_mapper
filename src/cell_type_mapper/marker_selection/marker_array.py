import copy
import h5py
import json
import numpy as np
import pathlib

from cell_type_mapper.diff_exp.sparse_markers_by_pair import (
    SparseMarkersByPair)

from cell_type_mapper.diff_exp.sparse_markers_by_gene import (
    SparseMarkersByGene)

from cell_type_mapper.marker_selection.marker_array_utils import (
    _create_new_pair_lookup,
    _idx_of_pair)


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
    up_marker_sparse:
        a diff_exp.sparse_markers.SparseMarkersByPair representing
        up-regulated marker genes (can be None)
    down_marker_sparse:
        ditto for down-regulated markers
    """
    def __init__(
            self,
            gene_names,
            taxonomy_pair_to_idx,
            n_pairs,
            up_by_pair,
            down_by_pair,
            up_by_gene,
            down_by_gene):

        self._gene_names = copy.deepcopy(gene_names)
        self.taxonomy_pair_to_idx = copy.deepcopy(
            taxonomy_pair_to_idx)
        self.n_pairs = n_pairs

        self.up_by_pair = up_by_pair
        self.down_by_pair = down_by_pair
        self.up_by_gene = up_by_gene
        self.down_by_gene = down_by_gene

    @classmethod
    def from_cache_path(
            cls,
            cache_path):

        cache_path = pathlib.Path(cache_path)
        if not cache_path.is_file():
            raise RuntimeError(
                f"{cache_path} is not a file")

        with h5py.File(cache_path, "r", swmr=True) as src:
            gene_names = json.loads(
                src['gene_names'][()].decode('utf-8'))
            taxonomy_pair_to_idx = json.loads(
                src['pair_to_idx'][()].decode('utf-8'))

            n_pairs = src['n_pairs'][()]

            up_by_gene = SparseMarkersByGene(
                gene_idx=src['sparse_by_gene/up_gene_idx'][()],
                pair_idx=src['sparse_by_gene/up_pair_idx'][()])
            up_by_pair = SparseMarkersByPair(
                gene_idx=src['sparse_by_pair/up_gene_idx'][()],
                pair_idx=src['sparse_by_pair/up_pair_idx'][()])
            down_by_gene = SparseMarkersByGene(
                gene_idx=src['sparse_by_gene/down_gene_idx'][()],
                pair_idx=src['sparse_by_gene/down_pair_idx'][()])
            down_by_pair = SparseMarkersByPair(
                gene_idx=src['sparse_by_pair/down_gene_idx'][()],
                pair_idx=src['sparse_by_pair/down_pair_idx'][()])

        return cls(
            gene_names=gene_names,
            taxonomy_pair_to_idx=taxonomy_pair_to_idx,
            n_pairs=n_pairs,
            up_by_pair=up_by_pair,
            up_by_gene=up_by_gene,
            down_by_pair=down_by_pair,
            down_by_gene=down_by_gene)

    def downsample_pairs_to_other(
            self,
            only_keep_pairs):
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

        up_by_gene = self.up_by_gene.keep_only_pairs(
            col_idx,
            in_place=False)
        up_by_pair = self.up_by_pair.keep_only_pairs(
            col_idx,
            in_place=False)
        down_by_gene = self.down_by_gene.keep_only_pairs(
            col_idx,
            in_place=False)
        down_by_pair = self.down_by_pair.keep_only_pairs(
            col_idx,
            in_place=False)

        return MarkerGeneArray(
            gene_names=self.gene_names,
            taxonomy_pair_to_idx=new_taxonomy_lookup,
            n_pairs=new_n_pairs,
            up_by_gene=up_by_gene,
            down_by_gene=down_by_gene,
            up_by_pair=up_by_pair,
            down_by_pair=down_by_pair)

    def downsample_genes(self, gene_idx_array):
        """
        Downselect to just the specified genes
        """
        print("downsampling genes")
        self._gene_names = [
            self._gene_names[ii]
            for ii in gene_idx_array]

        self.up_by_gene.keep_only_genes(gene_idx_array, in_place=True)
        self.up_by_pair.keep_only_genes(gene_idx_array, in_place=True)
        self.down_by_gene.keep_only_genes(gene_idx_array, in_place=True)
        self.down_by_pair.keep_only_genes(gene_idx_array, in_place=True)

    def downsample_genes_to_other(self, gene_idx_array):
        """
        Downselect to just the specified genes
        """
        print("downsampling genes")
        new_gene_names = [
            self._gene_names[ii]
            for ii in gene_idx_array]

        new_up_by_gene = self.up_by_gene.keep_only_genes(
                gene_idx_array,
                in_place=False)
        new_up_by_pair = self.up_by_pair.keep_only_genes(
                gene_idx_array,
                in_place=False)
        new_down_by_gene = self.down_by_gene.keep_only_genes(
                gene_idx_array,
                in_place=False)
        new_down_by_pair = self.down_by_pair.keep_only_genes(
                gene_idx_array,
                in_place=False)

        return MarkerGeneArray(
            gene_names=new_gene_names,
            taxonomy_pair_to_idx=self.taxonomy_pair_to_idx,
            n_pairs=self.n_pairs,
            up_by_pair=new_up_by_pair,
            down_by_pair=new_down_by_pair,
            up_by_gene=new_up_by_gene,
            down_by_gene=new_down_by_gene)

    @property
    def has_sparse(self):
        return True

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

        marker_mask = np.zeros(self.n_pairs, dtype=bool)
        up_mask = np.zeros(self.n_pairs, dtype=bool)

        up = self.up_by_gene.get_pairs_for_gene(gene_idx)
        down = self.down_by_gene.get_pairs_for_gene(gene_idx)

        up_mask[up] = True
        marker_mask[up] = True
        marker_mask[down] = True

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
        marker_mask = np.zeros(self.n_genes, dtype=bool)
        up_mask = np.zeros(self.n_genes, dtype=bool)

        up = self.up_by_pair.get_genes_for_pair(pair_idx)
        down = self.down_by_pair.get_genes_for_pair(pair_idx)
        assert len(set(up).intersection(set(down))) == 0

        up_mask[up] = True
        marker_mask[up] = True
        marker_mask[down] = True

        return (marker_mask, up_mask)

    def up_mask_from_pair_idx(
            self,
            pair_idx):
        """
        Return (n_genes,) boolean array indicating all genes that
        are up_regulated markers for the pair
        """
        result = np.zeros(self.n_genes, dtype=bool)
        result[self.up_by_pair.get_genes_for_pair(pair_idx)] = True
        return result

    def down_mask_from_pair_idx(
            self,
            pair_idx):
        """
        Return (n_genes,) boolean array indicating all genes that
        are wown_regulated markers for the pair
        """
        result = np.zeros(self.n_genes, dtype=bool)
        result[self.down_by_pair.get_genes_for_pair(pair_idx)] = True
        return result

    def up_regulated_gene_batch(self, gene0, gene1):
        """
        Return up_regulated mask between gene0:gene1 as a
        np.ndarray
        """
        raise NotImplementedError(
            "up_regulated_gene_batch not implemented for purely sparse "
            "marker array")

    def is_marker_gene_batch(self, gene0, gene1):
        """
        Return is_marker mask between gene0:gene1 as a
        np.ndarray
        """
        raise NotImplementedError(
            "is_marker_gene_batch not implemented for purely sparse "
            "marker array")
