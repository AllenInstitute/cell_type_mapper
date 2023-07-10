# Running the cell type mapping pipeline

This document explains how to run the code in this library to map unlabeled
cell by gene expression data onto the Allen Institute cell type taxonomy.
There are several steps in the pipeline that, in theory, only need to be run
once per (reference data, cell type taxonomy) pair. Those steps will be
pointed out and explained.

## General pipeline

There are three steps to running this pipeline

- Computing the average gene expression profile per cell type cluster in
the cell type taxonomy. **This step only needs to be run once per reference
dataset and produces a file that can be reused multiple times.**
- Identifying marker genes corresponding to the parent nodes in the cell type
taxonomy. **This step only needs to be run once per reference dataset and
produces a file that can be used multiple times.**
- Mapping unlabeled data onto the reference taxonomy using the executable
provided [here.](../src/hierarchical_mapping/cli/from_specified_markers.py)

### Computing the average gene expression profile per cell type cluster

For each (reference dataset, cell type taxonomy) pair, we need to store
the average gene expression profile per cell type cluster. This data, along
with the actual cell type taxonomy, is stored in an HDF5 file which can
be passed to the actuall cell type mapping exectuable as a config parameter.
An example script for creating this HDF5 file using the data from the
summer 2023 ABC Atlas release is provided
(here.)[../examples/precompute_stats_from_abc_release_data.py]
The resulting HDF5 file contains the following datasets.

- `metadata`: the JSON serialization of metadata encoding how and when
the HDF5 file was created.
- `cluster_to_row`: the JSON serialization of a dict mapping cell type
cluster labels to the corresponding row indices in the numerical datasets.
- `col_names`: the JSON serialization of a list mapping column index
in the numerical datasets to Ensembl gene identifiers.
- `taxonomy_tree`: the JSON serialization of a dict encoding the cell type
taxonomy tree to which data will be mapped. The contents of this dict match that of the taxonomy tree encoded in the cell type mapping output file and documented
[here.](output.md#taxonomy_tree) It can be loaded into a Python
object with
```
from hierarchical_mapping.taxonomy.taxonomy_tree import TaxonomyTree
tree = TaxonomyTree(
            data=json.loads(hdf5_handle['taxonomy_tree'][()].decode('utf-8')))
```
**Note:** that this encoding includes a mapping from cell type cluster (the
leaf level of the taxonomy tree) to individual cells in the reference dataset).

### Encoding marker genes

