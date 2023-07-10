# Running the cell type mapping pipeline

This document explains how to run the code in this library to map unlabeled
cell by gene expression data onto the Allen Institute cell type taxonomy.
There are several steps in the pipeline that, in theory, only need to be run
once per (reference data, cell type taxonomy) pair. Those steps will be
pointed out and explained.

## General pipeline

There are four steps to running this pipeline

- Computing the average gene expression profile per cell type cluster in
the cell type taxonomy. **This step only needs to be run once per reference
dataset and produces a file that can be reused multiple times.**
- Identifying marker genes corresponding to the parent nodes in the cell type
taxonomy. **This step only needs to be run once per reference dataset and
produces a file that can be used multiple times.**
- "Validating" the unlabeled dataset. This step converts gene symbols to
Ensembl IDs and (optionally) validates that the cell by gene expression
matrix is a set of raw count integers (you can skip this last step and use
the native normalization of your unlabeld h5ad file; more on that below).
- Mapping unlabeled data onto the reference taxonomy using the executable
provided [here.](../src/hierarchical_mapping/cli/from_specified_markers.py)

### Computing the average gene expression profile per cell type cluster

For each (reference dataset, cell type taxonomy) pair, we need to store
the average gene expression profile per cell type cluster. This data, along
with the actual cell type taxonomy, is stored in an HDF5 file which can
be passed to the actuall cell type mapping exectuable as a config parameter.
An example script for creating this HDF5 file using the data from the
summer 2023 ABC Atlas release is provided
[here.](../examples/precompute_stats_from_abc_release_data.py)
The resulting HDF5 file contains the following datasets.

- `metadata`: the JSON serialization of metadata encoding how and when
the HDF5 file was created.
- `cluster_to_row`: the JSON serialization of a dict mapping cell type
cluster labels to the corresponding row indices in the numerical datasets.
- `col_names`: the JSON serialization of a list mapping column index
in the numerical datasets to Ensembl gene identifiers.
- `taxonomy_tree`: the JSON serialization of a dict encoding the cell type
taxonomy tree to which data will be mapped. The contents of this dict match that of the taxonomy tree encoded in the cell type mapping output file and documented
[here.](output.md#taxonomy_tree) **Note:** that this encoding includes a
mapping from cell type cluster (the leaf level of the taxonomy tree) to
individual cells in the reference dataset.

#### Numerical datasets

The following datasets represent numerical statistics about the cell type
clusters and are indexed according to `cluster_to_row` and `col_names`. Not
all of these statistics are used at present. They are calculated in anticipation
of a future release in which this library will be able to identify marker genes
in the reference dataset without relying on legacy R code.

- `n_cells`: The number of cells in each cluster. `n_cells[ii]` is the number
of cells in the cluster with name `cluster_to_row[name] == ii`.
- `sum`: An (`n_clusters`, `n_genes`) array indicating the sum of `log2(CPM+1)`
values for each gene in each cluster (i.e. raw counts are converted to
`log2(CPM+1)` and then summed). Rows are indexed according to `cluster_to_row`.
Columns are indexed according to `col_names`.
- `sumsq`: like `sum`, except it represents the sum of the squares of the
`log2(CPM+1)` expression values (for use in estimating the variance).
- `gt0`: A (`n_clusters`, `n_genes`) array indicating how many cells had raw
expression greater than zero in each gene for each cluster.
- `gt1`: A (`n_clusters`, `n_genes`) array indicating how many cells had raw
expression greater than unity in each gene for each cluster.
- `ge1`: A (`n_clusters`, `n_genes`) array indicating how many cells ahd
raw expression greater than or equql to unity in each geen for each cluster.

### Encoding marker genes

At present, this library does not contain functionality for identifying the
marker genes needed to map unlabeled data onto the cell type taxonomy.
Hopefully, that functionality will come in a (near) future release. Until then,
our workflow is to have Changkyu Lee at the Allen Institute use his R library
to identify the marker genes, and then ingest his result into the file format
that this library expects. The expected form is a JSON serialized dict whose
contents are documented [here.](output.md#marker_genes) Changkyu's R code
produces a directory containing a series of CSV files, each listing the marker
genes for a parent node in the cell type taxonomy. We provide a tool to
combine these files into a single dict (and transform the gene sybols output
by the R code into Ensembl identifiers)
[here.](../src/hierarchical_mapping/cli/marker_cache_from_csv_dir.py)
To see the call signature and config parameters for that tool, run
```
python -m hierarchical_mapping.cli.marker_cache_from_csv_dir --help
```
**Note:**Because we encode the cell type taxonomy in the precomputed stats
HDF5 file documented
[above](#computing-the-average-gene-expression-profile-per-cell-type-cluster),
the path to that file is a config parameter of this tool.
