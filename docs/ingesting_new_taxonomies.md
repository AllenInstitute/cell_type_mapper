# Ingesting new taxonomies into the cell type mapper

We assume that any cell type taxonomy is based on a reference set of cells,
i.e. a set of cells that have been used to discover the cell
types in the taxonomy and that serve as the basis for answering questions like
"what is the average gene expression profile of all of the cells in classA?" or
"how many reference cells in classA express geneB at greater than 1 CPM?".

In order to run this cell type mapping code on a new taxonomy, you need
to create some supporting data files.

- The `precomputed_stats.h5` file is an HDF5 file that contains a serialization
of the cell type taxonomy tree onto which you are mapping, as well as statistics
about the reference cells that have been assigned to the cell types in your
taxonomy. The contents of that file and guidance for creating it is
[here](input_data_files/precomputed_stats_file.md).
- The marker gene lookup table tells the cell type mapper which marker genes to
use when selecting cell types at different levels in the cell type taxonomy.
This file, as well as tools to create it, are documented
[here](input_data_files/marker_gene_lookup.md).

**Note:** the marker gene lookup file is not
strictly required. If your cell type taxonomy is simple enough, you can
tell the cell type mapper to discover the marker genes at runtime, given the
context of your query datasets. This is only recommended for small taxonomies.
The cell type taxonomy backing the December 2023 SEA-AD release on MapMyCells
has only a few hundred clusters at the leaf level. Marker gene discovery takes
a few tens of seconds. The
[Allen Institute whole mouse brain
taxonomy](https://doi.org/10.1038/s41586-023-06812-z)
has 5,000
clusters at the leaf level. Marker gene discovery takes several hours. It
is up to the user to decide whether they want to precompute the marker gene
lookup table or discover marker genes at runtime.

To map cell types using a pre-computed marker gene lookup table, use the
command line interface
```
python -m cell_type_mapper.cli.from_specified_markers --help
```

To discover marker genes at runtime, use the command line interface
```
python -m cell_type_mapper.cli.map_to_on_the_fly_markers --help
```

(The `--help` argument will print out the specific call signature of these
tools and exit.)
