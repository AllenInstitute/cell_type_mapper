# precomputed_stats.h5

For each (reference dataset, cell type taxonomy) pair, we need to store
the average gene expression profile per cell type cluster. This data, along
with the actual cell type taxonomy, is stored in an HDF5 file. This repository
contains several tools for generating this file from different input datafiles.
Those will be discussed at the bottom of this page. First, we will document
the structure and contents of the file so that users whose exact use case
is not served by the available tools can create the file on their own
using their preferred method for writing HDF5 files.

## Schema

The precomputed stats file contains several non-numerical datasets representing
metadata about the cell type taxonomy and several numerical datasets
representing the average gene expression profiles of the cell types in the
taxonomy.

### Non-numerical datasets

The resulting HDF5 file contains the following datasets.

**Note:** throughout the discussion below we will refer to various
JSON serializations of data. These are strings represented as bytes
objects, i.e. in python
```
metadata = {'version': '1.1.2', 'date': 2023-11-14}
serialization = json.dumps(metadata).encode('utf-8')
```
Any downstream code that uses this data ultimately deserializes it with
```
with h5py.File('path/to/precomputed_stats.h5', 'r') as src:
    metadata = json.loads(
        src['metadata'][()].decode('utf-8'))
```

#### `metadata`

This is the JSON serialization of unstructured metadata regarding how
the precomputed_stats file was generated. It can contain whatever the
user desires.

#### `cluster_to_row`

The heart of the numerical datasets is a representation of
`(n_clusters, n_genes)` data, where "clusters" refers to the leaf node
of the cell type taxonomy. The `cluster_to_row` dataset is the JSON
serialization of a dict that maps the unique identifiers of the cell
type clusters to the row index in the `(n_clusters, n_genes)` arrays
of numerical data.

#### `col_names`

As discussed above, the numerical data stored in `precomputed_stats.h5`
is generally stored in arrays in which each row is a cell type cluster
and each column is a gene. The `col_names` dataset is the JSON serialization
of a list representing the names of the genes represented by these columns.

#### `taxonomy_tree`

This is the JSON serialization of a dict encoding the cell type
taxonomy tree to which data will be mapped. The contents of this dict match
that of the taxonomy tree encoded in the cell type mapping output file and
documented
[here.](../output.md#taxonomy_tree) **Note:** that this encoding includes a
mapping from cell type cluster (the leaf level of the taxonomy tree) to
individual cells in the reference dataset.


### Numerical datasets

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
raw expression greater than or equal to unity in each geen for each cluster.
