import copy
import numpy as np

from hierarchical_mapping.cell_by_gene.utils import (
    convert_to_cpm)


class CellByGeneMatrix(object):
    """
    A class to store a cell by gene matrix, keeping track
    of the genes stored in each column and the normalization
    of the data.

    Parameters
    ----------
    data:
        A numpy array. Each row is a cell; each column is a gen
    gene_identifiers:
        A list of the gene identifiers in the data
    normalization:
        Either "raw" or "log2CPM"; how is this data normalized
    cell_identifiers:
        Optional list of cell identifiers
    """
    def __init__(
            self,
            data,
            gene_identifiers,
            normalization,
            cell_identifiers=None):

        # has this file been downsampled by
        # genes (if it has, then conversion to
        # CPM will be impossible)
        self._genes_downsampled = False

        valid_norm = ("raw", "log2CPM")

        # check for valid normalization
        if normalization not in valid_norm:
            raise RuntimeError(
                f"Do not know how to handle normalization: {normalization}\n"
                "Valid values are {valid_norm}")

        # check that number of genes and number of gene identifiers
        # match
        if len(gene_identifiers) != data.shape[1]:
            raise RuntimeError(
                f"You gave {len(gene_identifiers)} gene_identifiers, "
                f"but data has {data.shape[1]} columns")

        # make sure gene identifiers are unique
        id_set = set()
        duplicates = []
        for gene_id in gene_identifiers:
            if gene_id in id_set:
                duplicates.append(gene_id)
            id_set.add(gene_id)
        if len(duplicates) > 0:
            raise RuntimeError(
                f"gene identifiers\n{duplicates}\nappear more than once "
                "in your list of gene_identifiers")

        self._normalization = normalization
        self._data = data
        self._gene_identifiers = copy.deepcopy(gene_identifiers)
        self._create_gene_to_col()
        self._process_cell_identifiers(cell_identifiers)

    def _create_gene_to_col(self):
        """
        Create the dict mapping gene_identifier to column index
        """
        self._gene_to_col = {n: ii
                             for ii, n in enumerate(self.gene_identifiers)}

    def _process_cell_identifiers(self, cell_identifiers):
        self._cell_identifiers = None
        self._cell_to_row = None
        if cell_identifiers is None:
            return
        duplicates = []
        id_set = set()
        for c in cell_identifiers:
            if c in id_set:
                duplicates.append(c)
            id_set.add(c)
        if len(duplicates) > 0:
            raise RuntimeError(
                f"cell identifiers\n{duplicates}\nrepeated")

        self._cell_identifiers = copy.deepcopy(cell_identifiers)
        self._cell_to_row = dict()
        for ii, n in enumerate(cell_identifiers):
            self._cell_to_row[n] = ii

    @property
    def normalization(self):
        return self._normalization

    @property
    def gene_identifiers(self):
        return self._gene_identifiers

    @property
    def gene_to_col(self):
        """
        A dict mapping gene_identifier to column index in data array
        """
        return self._gene_to_col

    @property
    def cell_identifiers(self):
        return self._cell_identifiers

    @property
    def cell_to_row(self):
        """
        A dict mapping cell identifier to row in the data array
        """
        return self._cell_to_row

    @property
    def data(self):
        return self._data

    @property
    def n_genes(self):
        return self._data.shape[1]

    @property
    def n_cells(self):
        return self._data.shape[0]

    def _downsample_genes(self, selected_genes):
        """
        Return the data array with only selected_genes included
        """
        id_set = set()
        for g in selected_genes:
            if g in id_set:
                raise RuntimeError(
                    f"gene {g} occurs more than once in selected_genes")
            id_set.add(g)

        idx_array = np.array([self.gene_to_col[n] for n in selected_genes],
                             dtype=int)

        return self.data[:, idx_array]

    def downsample_genes(self, selected_genes):
        """
        Return a new CellByGeneMatrix including only selected_genes
        """
        result = CellByGeneMatrix(
            data=self._downsample_genes(selected_genes),
            gene_identifiers=selected_genes,
            normalization=self.normalization,
            cell_identifiers=self.cell_identifiers)
        result._genes_downsampled = True
        return result

    def downsample_genes_in_place(self, selected_genes):
        """
        Alter this CellByGeneMatrix to contain only selected_genes
        """
        self._data = self._downsample_genes(selected_genes)
        self._gene_identifiers = copy.deepcopy(selected_genes)
        self._create_gene_to_col()
        self._genes_downsampled = True

    def downsample_cells(self, selected_cells):
        """
        Return another CellByGeneMatrix that only contains
        the cells specified by the selected_cells.

        Note: if self.cell_identifiers is None, selected_cells
        must be a list of integer indices.

        If self.cell_identifiers is not None, selected_cells
        must be a list of cell_identifiers.
        """

        if self.cell_identifiers is None:
            selected_cell_idx = selected_cells
            new_cell_id = None
        else:
            selected_cell_idx = [
                self.cell_to_row[c] for c in selected_cells]
            new_cell_id = selected_cells

        subset = self.data[selected_cell_idx, :]
        return CellByGeneMatrix(
            data=subset,
            gene_identifiers=self.gene_identifiers,
            normalization=self.normalization,
            cell_identifiers=new_cell_id)

    def to_log2CPM(self):
        """
        Return a new CellByGeneMatrix that is normalized to log2CPM
        """
        if self.normalization != "raw":
            raise RuntimeError(
                "You are calling to_log2CPM, but this CellByGeneMatrix "
                "already is not raw")

        if self._genes_downsampled:
            raise RuntimeError(
                "This CellByGeneMatrix has been downsampled by genes; "
                "converting to CPM will give a nonsense result")

        return CellByGeneMatrix(
            data=np.log2(1.0+convert_to_cpm(self.data)),
            gene_identifiers=self.gene_identifiers,
            normalization="log2CPM",
            cell_identifiers=self.cell_identifiers)

    def to_log2CPM_in_place(self):
        """
        Convert this CellByGeneMatrix to log2CPM normalization
        """
        if self.normalization != "raw":
            raise RuntimeError(
                "You are calling to_log2CPM_in_place, but this "
                "CellByGeneMatrix already is not raw")

        if self._genes_downsampled:
            raise RuntimeError(
                "This CellByGeneMatrix has been downsampled by genes; "
                "converting to CPM will give a nonsense result")

        self._data = np.log2(1.0+convert_to_cpm(self.data))
        self._normalization = "log2CPM"
