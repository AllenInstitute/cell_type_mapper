# Running the cell type mapping pipeline

This document explains how to run the code in this library to map unlabeled
cell by gene expression data onto the Allen Institute cell type taxonomy.
There are several steps in the pipeline that, in theory, only need to be run
once per (reference data, cell type taxonomy) pair. Those steps will be
pointed out and explained.

## General pipeline

There are four steps to running this pipeline

1. Computing the average gene expression profile per cell type cluster in
the cell type taxonomy. **This step only needs to be run once per reference
dataset and produces a file that can be reused multiple times.**
2. Identifying marker genes corresponding to the parent nodes in the cell type
taxonomy. **This step only needs to be run once per reference dataset and
produces a file that can be used multiple times.**
3. "Validating" the unlabeled dataset. This step converts gene symbols to
Ensembl IDs and (optionally) validates that the cell by gene expression
matrix is a set of raw count integers. There is a configuration you can
use if you already trust the normalization of your data; more on that below.
4. Mapping unlabeled data onto the reference taxonomy using the executable
provided [here.](../src/cell_type_mapper/cli/from_specified_markers.py)


### (1) Computing the average gene expression profile per cell type cluster

**Note:** This step only needs to be run once for each
(reference dataset, taxonomy) pair. The result is an HDF5 file which is
used as an input for steps (2) and (4).

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
taxonomy tree to which data will be mapped. The contents of this dict match
that of the taxonomy tree encoded in the cell type mapping output file and
documented
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
raw expression greater than or equal to unity in each geen for each cluster.

### (2) Encoding marker genes

**Note:** This step only needs to be run once for each
(reference dataset, taxonomy) pair. The result is a JSON file
that is used as an input for step (4).

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
[here.](../src/cell_type_mapper/cli/marker_cache_from_csv_dir.py)
To see the call signature and config parameters for that tool, run
```
python -m cell_type_mapper.cli.marker_cache_from_csv_dir --help
```
**Note:** Because we encode the cell type taxonomy in the precomputed stats
HDF5 file documented
[above](#computing-the-average-gene-expression-profile-per-cell-type-cluster),
the path to that file is a config parameter of this tool.


### (3) "Validating" the unlabeled dataset

This step takes the H5AD file containing the unlabeled data,
transforms it to meet the expectations of the mapping algorithm,
and saves the transformed data to another H5AD file. This new
file is what should be passed in as the `--query_path` parameter
of the CLI tool in step (4) below. This step was primarily written
to serve the requirements of the Brain Knowledge Platform's `MapMyCell`
tool, namely that

- All genes are identified with Ensembl IDs
- Cell by gene expression data to be mapped are stored in the `X`
layer of the unlabeled h5ad file
- Cell by gene expression data are raw counts expressed as integers

The first two requirements are non-negotiable. The third requirement can
be circumvented if your data is already log2 normalized and you want to use
your normalization.

There is a command line tool to take an arbitrary H5AD file and transform
it so that it meets the above requirements. To see its call signature, run

```
python -m cell_type_mapper.cli.validate_h5ad
```

The default behavior of this tool is to read data from the input
H5AD file's `X` layer, round it to the nearest integer, and save the rounded
data to the validated H5AD file's `X` layer. If you want to read data
from a different layer in the input H5AD file, specify that layer
with the `--layer` config parameter. If you do not wish your data to
be rounded to the nearest integer (e.g. if your cell by gene data is not
raw counts and has already been log2 normalized), run with `--round_to_int false`.

Regardless, the output cell by gene data will be saved to the `X` layer of
the validated H5AD file written by this tool, which is where the mapping tool
expects the cell by gene data to be.

**Note:** if you do not specify an output path with `--valid_h5ad_path`, the
tool will automatically create one from `--output_dir` and the name of the
input H5AD file. The resulting output file will be recorded in the output
manifest specified with `--output_json`.


### (4) Mapping unlabeled data onto the reference taxonomy

This is the step where we actually map the unlabeled data onto the
cell types taxonomy. It takes as inputs the files produced in steps (1), (2),
and (3). The CLI tool for this step can be run using
```
python -m cell_type_mapper.cli.from_specified_markers --help
```
For historical reasons, there are a lot of configuration parameters that are
not actually used any more (these will get cleaned up in a future version).
The parameters that are used are:

- `query_path`: the path to the validated H5AD file produced in step (3).
- `extended_result_path`: the path to the [extended output file.](output.md#json-output-file)
- `csv_result_path`: the optional path to the [CSV output file](output.md#csv-output-file)
- `tmp_dir`: directory where temporary data files can be written. This is especially important if your data is stored in a slow network mounted file system. Specifying a faster drive here will tell the code to copy the data into the faster drive before working on it, which can increase speed significantly. Just make sure that your `tmp_dir` has enough space to store an entire copy of your query data.
- `log_path`: the optional path to a text file containing log messages from the mapping run
(log messages will also be recorded in the extended output file).
- `max_gb`: available GB of memory for use when converting the input H5AD from a CSC sparse
matrix to a CSR sparse matrix (irrelevant if the input H5AD file is already in CSR or dense
form).
- `drop_level`: a level to drop from the cell type taxonomy before doing the mapping. This is
necessary because the Allen Institute taxonomy includes a "supertype" level that is not actually
used during hierarchical mapping. **Note:** be sure to use the same level naming scheme as was
used in the taxonomy you created in step (1).
- `flatten`: a boolean. If `true`, then flatten the cell type taxonomy and fit directly to the
leaf level nodes without traversing the tree.
- `precomputed_stats.path`: the path to the HDF5 file created in step (1).
- `query_markers.serialized_lookup`: the path to the JSON file created in step (2).
- `type_assignment.normalization`: either 'raw' or 'log2CPM'. Indicates the normalization of
the cell by gene data in `query_path`. If 'raw', the code will convert it to `log2(CPM+1)`
internally before actually mapping.
- `type_assignment.bootstrap_iteration`: the number of bootstrapping iterations to run
at each node of the taxonomy tree.
- `type_assignment.bootstrap_factor`: The factor by which to downsample the population of
marker genes for each bootstrapping iteration.
- `type_assignment.n_processors`: The number of independent worker processes to spin up.
- `type_assignment.chunk_size`: The number of cells to pass to each independent worker
process.
- `type_assignment.rng_seed`: An integer for seeding the random number generator.

These parameters may be written into a config file that looks like

```
{
"query_path": "path/to/file.h5ad",
"extended_result_path" "path/to/output.json",
...
"precomputed_stats": {
    "path": "path/to/stats.h5"
},
"query_markers": {
    "serialized_marker_lookup": "path/to/marker/file.json"
},
"type_assignment": {
    "normalization": "raw",
    "flatten": False,
    ...
}
```

and passed in using
```
python -m cell_type_mapper.cli.from_specified_markers --input_json path/to/config.json
```

#### "Flat" Correlation mapping

To run the `MapMyCell` Correlation Mapping algorithm (i.e. mapping onto a tree with only one taxonomic
level; no bootstrapping), run the above code with
```
--flatten true
--type_assignment.bootstrap_iteration 1
--type_assignment.bootstrap_factor 1.0
```
This will be an order of magnitude faster than running full hierarchical mapping because there
are many fewer nearest neighbor searches to do.

To run "legacy" flat mapping (flat mapping that uses bootstrapping) run with
```
--flatten true
--type_assignment.bootstrap_iteration 100
--type_assignment.bootstrap_factor 0.9
```

**Note:** because flattening the taxonomy causes every bootstrap iteration to
use every marker gene, running legacy flat mapping is no faster (and can even
be slower) than hierarchcal mapping. This because, with hierarchical mapping, as
you descend the taxonomy tree, you need drammatically fewer marker genes, which
speeds up the nearest neighbor searches going on within the algorithm.

#### Running programmatically

This module uses the
[argschema library](https://github.com/AllenInstitute/argschema)
to manage configuration parameters. If you want to run this mapping tool
from within a python script, run

```
from hierarchical_mapping.cli.from_specified_markers import (
    FromSpecifiedMarkersRunner)

runner = FromSpecifiedMarkersRunner(
    args=[],
    input_data=dict_containing_config_parameters)

runner.run()
```

The `args=[]` is important to prevent the `runner` from trying to grab
configuration parameters from the command line.
