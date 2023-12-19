# Marker gene lookup table

The `from_specified_markers` command line tool expects you to pass in the path
to a file telling the code which marker genes to use at which node in the
cell type taxonomy. Below we will discuss the [schema](#Schema) of the expected
file and
[tools provided in this codebase](#creating-the-marker-gene-lookup-table) to
create the marker gene lookup table. **Note:** any file that adheres to the
schema below will work. Marker gene discovery is an open research question in
the field. The tools in this codebase use a specific algorithm (documentation
forthcoming). Users should feel free to select marker genes however they
are comfortable doing so.

## Schema

This file is a text file that contains the JSON
serialization of a dict. The keys of the dict are the parent nodes in the
cell type taxonomy tree (i.e. the nodes with children that need to be chosen
between). The root of the tree (i.e. the node from which the first cell type
choice is made) is specified with the key `'None'`. All other keys are of the
form `"level"/"node"` where `"level"` is the the node of the cell type taxonomy
(e.g. "class", "subclass", "cluster") and `"node"` is the specific node
at that level. The values of the dict are lists of marker genes to be used
when selecting between that parent node's child nodes.

A cartoon example of a marker gene lookup table is below. Parent nodes
that have only one child node may either be left out of the lookup table
or represented with an empty list of genes (because they only have on child,
there is no choice to be made when you come to that node).

**Note:** It is very important that the `"level"`, `"node"` and gene identifiers
correspond to the contents of your `precomputed_stats.h5` file. Specifically,
if you use the `precompute_stats_abc` command line tool to create your
`precomputed_stats.h5` file, `"level"` and `"node"`
must be the machine readable names encoded in `cluster_annotation_term_set_label`
and `cluster_annotation_term_label`
(see [here](precomputed_stats_file.md#taxonomy-containing-csv-files)).
This also means that if your `precomputed_stats.h5` file refers to genes
with, for example, Ensembl IDs, your marker gene lookup file must also refer
to genes with their Ensembl IDs.

Here is the example cartoon lookup table
```
{
  "None": [
    "PKNOX2",
    "REEP1",
    "TUSC3",
    "SYT16",
    "CADPS2",
    "A2M"
  ],
  "class/GABAergic": [
    "NFIB",
    "GRIK3",
    "TANC1",
    "LHX6",
    "SAMD5",
    "COBL"
  ],
  "class/Glutamatergic": [
    "FEZF2",
    "PDZRN3",
    "EGFEM1P"
  ],
  "class/Non-neuronal": [
    "PPP2R2B",
    "DOCK10",
    "SLC12A2",
    "CLDND1",
    "RASGRF2",
    "EYA2"
  ],
  "subclass/MGE": [
    "FGFR2",
    "FRMD6",
    "TRPC5",
    "SST",
    "SULF1",
  ],
  "subclass/Astrocyte": [],
  "subclass/Micro-PVM": [],
  "subclass/Oligo-OPC": [],
  "subclass/Other NN": [],
  "subclass/CGE_PoA": [
    "PCDH11X",
    "FBXL7",
    "PRKG2",
    "PDGFD",
    "NXPH2",
    "GOLIM4",
    "KMO",
    "CHST15",
    "LTBP1"
  ]
}
```

## Creating the marker gene lookup table

Detailed documentation of the marker gene discovery algorithm used by this
codebase will be recorded elsewhere. Here we present a user's guide to the
tools in this codebase to go from a `precomputed_stats.h5` file to a
`marker_gene_lookup.json` file.

There are two steps in finding marker genes (according to this codebase).

1. Finding the "reference marker genes." In this step we find, for every pair of leaf nodes in the cell type taxonomy tree, every gene that could conceivably be
a marker gene for discriminating between those two cell types.
2. Subselecting the genes found in (1) to give a set of marker genes that is
more manageable in size.

These steps are carried out by different executables with different outputs.

### Finding the reference markers

Because (1) involves finding all of the marker genes for `(n_clusters choose 2)`
pairs of cell types, it can take a long time and produce a lot of data. If
there are 5,000 cell type clusters in your taxonomy (as with the Allen Institute
whole mouse brain taxonomy), `(5,000 choose 2)` means you are finding the marker genes for 12.5 million `(clusterA, clusterB)` pairs. This not only takes hours,
it can produce a list of marker genes that is several tens of gigabytes in size.
The output of this step is therefore saved to an HDF5 file (which can be 30 GB
in size). This HDF5 file is then taken as the input to (2), which actually
produces the expected `marker_gene_lookup.json` file.

Step (1) is carried out by the command line tool
```
python -m cell_type_mapper.cli.reference_markers --help
```
This tool has many configurable parameters that inform the specific algorithm
being carried out. Those will be documented elsewhere. For the purposes of
this document, it is important for the user to specify

- `precomputed_path_list`: the path to the `precomputed_stats.h5` file or files
representing the taxonomy for which we are finding marker genes.
- `n_processors`: the number of independent worker processes to spin up.
- `tmp_dir`: a directory where temporary scratch files can be written (the
code will clean these up after itself).
- `output_dir`: the directory where the resulting HDF5 file will be written.

In order to support cases where reference datasets contain multiple modalities
that should not be mixed (e.g. 10Xv3 and 10Xv2), we have the user specify and
output directory rather than an output file. The idea is that a user can then
pass all of the reference marker files in the output directory into the next
step and the code will be smart enough to select marker genes without mixing
modalities. Given a precomputed stats file named like
```
my_precomputed_stats.this_particular_version.h5
```
the reference markers code will try to write a file named
```
output_dir/reference_markers.this_particular_version.h5
```
(i.e. it will replace the first `.` delimited block of the precomputed stats
file name with `reference_markers`). If that file already exists, an integer
will be added to the name as a "salt" to make the file name unique. If you do
not want to deal with this ambiguity, either specify an empty `output_dir` or
run the code with `--clobber True`, in which case existing files will be
overwritten.

The reference marker HDF5 file contains the path to the `precomputed_stats.h5` file from whichit was  created in its `metadata` field, allowing step (2)
of the process to correctly link `reference_maker.h5` and `precomputed_stats.h5`
files as needed.

### Subselecting the reference markers

The tool that takes the `reference_markers.h5` file produced in step (1) above
and converts it into the final JSON lookup table of marker genes is invoked via
```
python -m cell_type_mapper.cli.query_markers --help
```
It accepts a list of of `reference_marker.h5` files and writes out the expected
JSON lookup table.

The key configuration parameters affecting the output are `n_per_utility` and
`n_per_utility_override`. They function as follows:

For every `(clusterA, clusterB)` pair being compared, the code will try to
select marker genes such that there are `n_per_utility` marker genes
up-regulated in `clusterA` and `n_per_utility` marker genes up-regulated in
`clusterB`. The default value of `30` is probably a reasonable choice, with the
exception that it can result in a few thousand marker genes at the root node,
depending on how many clusters there are in the taxonomy.
`n_per_utility_override` allows you to set a different value of `n_per_utility`
for specific parent nodes in the taxonomy, for instance
```
--n_per_utility_override '[("None", 5), ("class/classA", 10)]'
```
would tell the code to treat `n_per_utility` as being equal to 5 when selecting
markers at the root node and as being equal to 10 when selecting marker genes
at the parent node `classA`.
