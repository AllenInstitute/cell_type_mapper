# Running the cell type mapping pipeline

This document explains how to run the code in this library to map unlabeled
cell by gene expression data onto a reference cell type taxonomy.

If you wish to map to one of the taxonomies supported by the
[on-line MapMyCells tool](https://portal.brain-map.org/atlases-and-data/bkp/mapmycells),
you will need to download the data files that describe the desired taxonomy.
Instructions for downloading those files and running them through this
codebase can be found
[here.](input_data_files/running_online_taxonomies_locally.md)

Below, you will find instructions for using this codebase to map
data to an arbitrary taxonomy.

## Defining the taxonomy on which to map

In order to map an unlabeled dataset onto a taxonomy, you need two supporting
datafiles:

- an HDF5 file defining your taxonomy and the average gene expression
profile for the cell types in your taxonomy and
- a JSON file defining the marker genes to be used for mapping given your taxonomy.

These files may be costly (~ a few hours) to generate, but should only
need to be generated
once for a given taxonomy. The specific schemas
of these files and the tools provided in this codebase to help create them
are [documented here](ingesting_new_taxonomies.md).

If you are an internal Allen Institute user and just want to map to one of
the taxonomies currently supported by the MapMyCells online app, email Scott
Daniel, and he can provide you with the appropriate files. If you are an
external user and want to map to one of the MapMyCells-supported taxonomies
and the
[online MapMyCells app](https://portal.brain-map.org/atlases-and-data/bkp/mapmycells)
is for some reason insufficient for your purposes, please
file an issue on this repository and we can discuss how to get you the relevant
files.

The specific algorithm used for cell type mapping is described
[here.](algorithms/hierarchical_mapping.md)

**Note:** for small taxonomies (~ a few hundred; maybe 1,000 leaf nodes), there
is an alternative mapping tool documented
[here](#finding-markers-at-runtime)
which finds marker genes while running the mapping algorithm, as opposed
to as a precomputed step. The advantage of this code is that, if your
unlabeled dataset does not contain all of the genes that your reference dataset
contained, the code can adapt its marker selection strategy to accommodate the
limited gene list. The disadvantage of this code is that, for large taxonomies,
marker selection can take hours (for the
[Allen Institute whole mouse brain
taxonomy](https://doi.org/10.1038/s41586-023-06812-z)
with ~ 5,000 cell type clusters marker selection takes 3 hours).
Users can decide on their own tolerance for delay.

## Requirements on unlabeled data

The only requirements on the unlabeled data being mapped are

- Data is stored in an
[H5AD file](https://anndata.readthedocs.io/en/latest/fileformat-prose.html)
- cell-by-gene expression data is stored in the `X` matrix of the H5AD file
- `var.index.values` correspond to the gene identifiers used in the marker
gene lookup JSON file.

## Mapping unlabeled data onto the reference taxonomy

Once you have created the necessary input files, mapping unlabeled data
to your taxonomy can proceed using the command line interface tool
invoked via
```
python -m cell_type_mapper.cli.from_specified_markers --help
```
Running with `--help` will output detailed documentation of the command
line parameters accepted by this tool. The important parameters are

- `query_path`: the path to the H5AD file containing your unlabeled data.
- `extended_result_path`: the path to the [extended output file.](output.md#json-output-file)
- `csv_result_path`: the optional path to the [CSV output file](output.md#csv-output-file)
- `precomputed_stats.path`: the path to the [`precomputed_stats.h5` file as documented here.](input_data_files/precomputed_stats_file.md)
- `log_path`: the optional path to a text file containing log messages from the mapping run
(log messages will also be recorded in the extended output file).
- `query_markers.serialized_lookup`: the path to the [marker gene lookup file as
documented here.](input_data_files/marker_gene_lookup.md)
- `drop_level`: a level to drop from the cell type taxonomy before doing the mapping. This is
necessary because the Allen Institute taxonomy includes a "supertype" level that is not actually
used during hierarchical mapping.
- `flatten`: a boolean. If `True`, then flatten the cell type taxonomy and fit directly to the
leaf level nodes without traversing the tree.
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
- `max_gb`: available GB of memory for use when converting the input H5AD from a CSC sparse
matrix to a CSR sparse matrix (irrelevant if the input H5AD file is already in CSR or dense
form).
- `tmp_dir`: directory where temporary data files can be written. This is especially important if your data is stored in a slow network mounted file system. Specifying a faster drive here will tell the code to copy the data into the faster drive before working on it, which can increase speed significantly. Just make sure that your `tmp_dir` has enough space to store an entire copy of your query data.
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

### "Flat" Correlation mapping

To run the `MapMyCells` Correlation Mapping algorithm (i.e. mapping onto a tree with only one taxonomic
level; no bootstrapping), run the above code with
```
--flatten True
--type_assignment.bootstrap_iteration 1
--type_assignment.bootstrap_factor 1.0
```
This will be an order of magnitude faster than running full hierarchical mapping because there
are many fewer nearest neighbor searches to do.

To run "legacy" flat mapping (flat mapping that uses bootstrapping) run with
```
--flatten True
--type_assignment.bootstrap_iteration 100
--type_assignment.bootstrap_factor 0.9
```

**Note:** because flattening the taxonomy causes every bootstrap iteration to
use every marker gene, running legacy flat mapping is no faster (and can even
be slower) than hierarchcal mapping. This because, with hierarchical mapping, as
you descend the taxonomy tree, you need drammatically fewer marker genes, which
speeds up the nearest neighbor searches going on within the algorithm.

## Running programmatically

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

## "Validating" the unlabeled dataset

The online MapMyCells app demands that users identify genes with Ensembl IDs.
To facilitate this process, we provie a command line tool that will take
as input the H5AD file of unlabeled data and write out a new H5AD file with
the same contents, except that the genes are all identified by Ensembl IDs.

**This step may not be required if you are mapping to taxonomies other than
those already supported in the online version of MapMyCells.** It is also not
required if your H5AD file already refers to genes via Ensembl IDs.

This tool can be invoked via

```
python -m cell_type_mapper.cli.validate_h5ad --help
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


## Finding markers at runtime

If users do not wish to pre-calculate their marker gene lookup table,
there is a tool provided which requires only a `precomputed_stats.h5` file
and will find marker genes at runtime. That tool can be invoked via
```
python -m cell_type_mapper.cli.map_to_on_the_fly_markers --help
```
It uses all of the config parameters for the
[marker gene selection code](input_data_files/marker_gene_lookup.md)
with parameters for reference marker finding passed in with a
`reference_markers.` prefix and parameters for query marker
selection passed in with a `query_markers.` prefix.

Users should read both the
[surface level](input_data_files/marker_gene_lookup.md)
and the
[in depth](algorithms/marker_gene_selection.md)
marker gene selection documentation before deciding
to use this tool. As mentioned above, depending on the size
of the cell type taxonomy tree, it could be a time-consuming
calculation. Detailed profiling has not yet been performed. We will
report findings on the relationship between the number of cell type
clusters and the time it takes to find marker genes as they become
available.
