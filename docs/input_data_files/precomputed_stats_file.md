# precomputed_stats.h5

For each (reference dataset, cell type taxonomy) pair, we need to store
the average gene expression profile per cell type cluster. This data, along
with the actual cell type taxonomy, is stored in an HDF5 file. This repository
contains several tools for generating this file from different input datafiles.
Those tools will be discussed next. For users whose need are not served
by those tools and who feel comfortable writing out their own HDF5 file
"by hand" the schema of `precomputed_stats.h5` is documented
[at the end of this page.](#Schema)

## Tools to create `precomputed_stats.h5`

There are currently two command-line tools provided to create
`precomputed_stats.h5`. One that creates the file from a single H5AD file
which is expected to contain all of the cell-by-gene expression data as well
as the cell type taxonomy. Another tool creates the file from a series of
H5AD files containing the cell-by-gene expression data alongside a series of
CSV files encoding the cell type taxonomy parent-child relationships (this
mirrors the way data is being released for the Allen Institute ABC Atlas).

### Creating `precomputed_stats.h5` from a single H5AD file

The tool to create `precomputed_stats.h5` from a single H5AD file is invoked
via
```
python -m cell_type_mapper.cli.precompute_stats_scrattch --help
```
Running with `--help` will give the specific command line arguments that the
tool accepts/needs.

The tool expects the input H5AD file (specified with the `--h5ad_path` argument)
to contain the following data.

- cell-by-gene expression data stored in the `X` matrix.
- cell type assignments at each level in the taxonomy stored as separate columns
in the `obs` dataframe.

To specify the cell type taxonomy, you pass the list of levels in the taxonomy
from most gross to most fine in the `--hierarchy` command line argument. For
instance, if you ran
```
python -m cell_type_mapper.cli.precompute_stats_scrattch \
--h5ad_path path/to/my_file.h5ad \
--hierarchy '["class", "subclass", "cluster"]'
```
(Note the nested quotation marks around the specification of `--hierarchy`).

the tool will expect `obs` to contain columns `"class"`, `"subclass"`, and
`"cluster"` recording which of these taxons each cell belongs to. Additionally,
it is required that the taxonomy be a tree (i.e. every cell with the same
`"cluster"` has the same `"subclass"`; every cell with the same `"subclass"` has
the same `"class"`). If your taxonomy is not a strict tree, the code will
tell you as much and throw an error.

The cell-by-gene expression data in `X` can either be in raw counts or
`log2(CPM+1)`. Tell the code which it is with the `--normalization` command line
argument (either `"raw"` or `"log2CPM"`).


### Creating `precomputed_stats.h5` from an ABC Atlas release

The `precompute_stats_abc` command line tool, invoked via
```
python -m cell_type_mapper.cli.precompute_stats_abc --help
```
assumes a different data model than `precompute_stats_scrattch` defined above.

- cell by gene data can be contained in several h5ad files. However, these files **only** contain cell by gene data (again in the `X` matrices; again either
`"raw"` or `"log2CPM"` normalization). They do not
contain any cell type taxonomy data. `obs` and `var` are only used to find
cell labels and gene labels.
- cell type taxonomy data is encoded in a series of CSV files described below

#### Taxonomy-containing CSV files

##### `cluster_to_cluster_annotation_membership.csv`

A CSV in which each row is a node in the cell type taxonomy and the
columns are (there may be extra columns; these are the required columns)

- `cluster_annotation_term_set_label`: the machine-readable identifier of
the level in the taxonomy hierarchy (e.g. `CCN20230722_CLAS`,
`CCN20230722_SUBC`, and `CCN2020722_CLUS` for classes, subclasses,
and clusters).
- `cluster_annotation_term_set_name`: the human-readable name corresponding
to the term set label (e.g. `class`, `subclass`, and `cluster`).
- `cluster_annotation_term_label`: the machine-readable identifier of
individual node in the taxonomy tree (e.g. `CS20230722_CLAS_01` for a specific
class).
- `cluster_annotation_term_name`: the human readable name corresponding to
the annotation term label (e.g. `L6 IT CTX Glut_3`).
- `cluster_alias`: the globally unique alias (probably an integer)
corresponding to the node in the taxonomy tree (this is only used for leaf
nodes in the taxonomy tree).

#### `cluster_annotation_term.csv`

A CSV in which each row is a node in the cell type taxonomy and the
columns are (there may be extra columns; these are the required columns)

- `label`: the machine readable label of the node in the taxonomy tree (must
correspond to the `cluster_annotation_term_label` column in
`cluster_to_cluster_annotation_membership.csv`).
- `cluster_annotation_term_set_label`:  same as the similarly named column in
`cluster_to_cluster_annotation_membership.csv`.
- `parent_term_label`: the `label` of the node in the taxonomy tree that is
the direct parent of the current node.
- `parent_term_set_label`: the `cluster_annotation_term_set_label` of the
node in the taxonomy tree that is the direct parent of the current node.

#### `cell_metadata.csv`

A CSV in which each row is a cell in the dataset and the columns are (there may
be extra columns; these are the required columns)

- `cell_label`: the unique identifier of the cell. Must correspond to the index
of the `obs` dataframe in the H5AD file containing cell-by-gene data for the
cell.
- `cluster_alias`: the alias of the leaf node in the taxonomy to which the
cell was assigned. This must correspond to a `cluster_alias` in
`cluster_to_cluster_annotation_membership.csv`.

Once the tool has ingested the taxonomy, it will scan the list of provided
H5AD files to find the cells that belong to each cell type and assemble the
required statistics. It is not required that cells of the same cell type
be in the same H5AD file. Each cell in `cell_metadata.csv` must be in
one of the provided H5AD files. The H5AD files may also contain cells not
mentioned in `cell_metadata.csv`. These cells will be ignored when populating
the `precomputed_stats.h5` file.

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
documented [here.](taxonomy_tree.md)

### Numerical datasets

The numerical datasets encode the cell-type-cluster-by-gene statistics
characterizing the cell type clusters in the taxonomy. With the exception
of `n_cells` they are all `(n_cluster, n_gene)` in shape. The columns must
be in the same order as the gene names encoded in `col_names` above.
The mapping from cell type cluster to row is encoded in the `cluster_to_row`
dataset above.

#### `n_cells`

This is a `(n_clusters,)` array of ints. Each element records the number
of cells in the reference dataset that were assigned to that particular
cell type cluster.

#### `sum` and `sumsq`

These are `(n_clusters, n_genes)` arrays of floats aggregating the sum of the
gene expression values and the sum of the squares of gene expression values
per cell type cluster per gene (we collect the sum of the squares for use
when computing variances during marker gene selection). Values are assumed
to be in `log2(CPM+1)` normalization before squaring and/or summing.

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
