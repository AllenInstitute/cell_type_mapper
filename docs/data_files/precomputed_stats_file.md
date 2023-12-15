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

The numerical datasets encode the cell-type-cluster-by-gene statistics
characterizing the cell type clusters in the taxonomy. With the exception
of `n_cells` they areall `(n_cluster, n_gene)` in shape. The columns must
be in the same order as the gene names encoded in `col_names` above.
The mapping from cell type cluster to row is encoded in the `cluster_to_row`
dataset above.

#### `n_cells`

This is a `(n_clusters,)` array of ints. Each element records the number
of cells in the reference dataset that were assigned to that particular
cell type cluter.

#### `sum` and `sumsq`

These are `(n_clusters, n_genes)` arrays of floats aggregating the sum of the
gene expression values and the sum of the squares of gene expression values
per cell type cluster per gene (we collect the sum of the squares for use
when computing variances during marker gene selection)

**Note:** If you are not using the `precomputed_stats.h5` file for marker
gene selection (i.e. if you are only using it as the source of the average
gene expression per cell type cluster in your reference dataset) it is safe
to leave out `sumsq`. Only `sum` is **required.**

#### `ge1`, `gt1`, `gt0`

These are `(n_clusters, n_genes)` arrays of integers recording, for each
`(cluster, gene)` pair, how many cells

- expressed that gene at greater than or equal to 1 CPM
- expressed that gene at greater than 1 CPM
- expressed that gene at greater 0 CPM

**Note:** These arrays are only used for marker gene selection. If you are just
creating a `precomputed_stats.h5` file to serve as the source of
"average gene expression per cell type", having selected your marker genes
using some other tool, it is save to leave `ge`, `gt1` and `gt0` out of your
`precomputed_stats.h5` file.

**Note:** As of this writing, `gt1` and `gt0` actually are not even used
in marker gene selection (they are placeholders against future developments
in marker gene selection). They can safely be left out of
`precomputed_stats.h5`.
