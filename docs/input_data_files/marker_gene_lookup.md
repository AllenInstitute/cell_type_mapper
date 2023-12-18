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
at that level. The values of the dicts are lists of marker genes to be used
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
codebase will be recorded elsewhere. Here we present a user guide the tools
in this codebase to go from a `precomputed_stats.h5` file to a
`marker_gene_lookup.json` file.

There are two steps in finding marker genes (according to this codebase).

1. Finding the "reference marker genes." In this step we find, for every pair of leaf nodes in the cell type taxonomy tree, every gene that could conceivably be
a marker gene for discriminating between those two cell types.
2. Subselecting the genes found in (1) to give a set of marker genes that is
more manageable in size.
