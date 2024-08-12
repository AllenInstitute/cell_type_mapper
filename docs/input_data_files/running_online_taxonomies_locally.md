# Mapping to taxonomies supported by MapMyCells

In order to use this codebase to map unlabeled data onto the taxonomies
supported by the on-line MapMyCells tool, you will need the data files
defining those taxonomies. We have made them available for download from
AWS's S3 storage service. We describe where to find the files for each supported
taxonomy below.

**Note:** some of the files that need to be downloaded (specifically, the
marker gene lookup tables) are JSON files. When you click on the link to one
of those files, your browser will likely default to displaying the file
on-screen, rather than downloading it. To download the file to your local
machine, you will need to run a command-line downloading tool. For instance, to
copy the marker genes for the Whole Mouse Brain taxonomy, you will need to run
something like

```
wget https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/mapmycells/WMB-10X/20240831/mouse_markers_230821.json ./
```
or
```
curl -OL https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/mapmycells/WMB-10X/20240831/mouse_markers_230821.json
```

For the `.h5` files referenced below, clicking on the hyperlink will be
sufficient to initiate download.


**Note:** we provide examples below of the command line tools you can run
to perform a MapMyCells mapping on your own local machine. After running
these tools, you may see a warning appear that looks like
```
UserWarning: resource_tracker: There appear to be 4 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
```
this is an artifact of some of the libraries the `cell_type_mapper` depends
on and can be safely ignored. As long as you see messages like
```
BENCHMARK: spent 3.9620e+01 seconds assigning cell types
Writing marker genes to output file
MAPPING FROM SPECIFIED MARKERS RAN SUCCESSFULLY
CLEANING UP
```
(probably above this warning), the mapping successfully completed.


## 10x Whole Mouse Brain taxonomy (CCN20230722)

The data files required to map data onto the Whole Mouse Brain taxonomy
can be found [here](https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/index.html#mapmycells/WMB-10X/20240831/).

`mouse_markers_230821.json` is the JSON-serialized
[marker gene lookup table](marker_gene_lookup.md).

`precomputed_stats_ABC_revision_230821.h5` is the HDF5 file containing
[the precomputed statistics](precomputed_stats_file.md).

To map unlabeled data to this taxonomy, run the command-line tool

```
python -m cell_type_mapper.cli.from_specified_markers \
--query_path /path/to/unlabeled_data.h5ad \
--extended_result_path output.json \
--csv_result_path output.csv \
--drop_level CCN20230722_SUPT \
--cloud_safe False \
--query_markers.serialized_lookup /path/to/mouse_markers_230821.json \
--precomputed_stats.path /path/to/precomputed_stats_ABC_revision_230821.h5 \
--type_assignment.normalization raw \
--type_assignment.n_processors 4
```

The full call signature of this tool is [documented here.](../mapping_cells.md)


## 10x Whole Human Brain taxonomy (CCN202210140)

The data files required to map data onto the Whole Human Brain taxonomy
can be found [here](https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/index.html#mapmycells/WHB-10Xv3/20240831/).

`query_markers.n10.20240221800.json` is the JSON-serialized
[marker gene lookup table](marker_gene_lookup.md).

`precomputed_stats.siletti.training.h5` is the HDF5 file containing
[the precomputed statistics](precomputed_stats_file.md).

To map unlabeled data to this taxonomy, run the command-line tool

```
python -m cell_type_mapper.cli.from_specified_markers \
--query_path /path/to/unlabeled_data.h5ad \
--extended_result_path output.json \
--csv_result_path output.csv \
--cloud_safe False \
--query_markers.serialized_lookup /path/to/query_markers.n10.20240221800.json \
--precomputed_stats.path /path/to/precomputed_stats.siletti.training.h5 \
--type_assignment.normalization raw \
--type_assignment.bootstrap_factor 0.5 \
--type_assignment.n_processors 4
```

The full call signature of this tool is [documented here.](../mapping_cells.md)

## 10x Human MTG SEA-AD taxonomy (CCN20230505)

The data files required to map data onto the Human MTG taxonomy
can be found [here](https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/index.html#mapmycells/SEAAD/20240831/).

There is only one file for this taxonomy.
`precomputed_stats.20231120.sea_ad.MTG.h5` is the HDF5 file containing
[the precomputed statistics](precomputed_stats_file.md). Because the
Human MTG taxonomy is so small, MapMyCells finds marker genes for it
at run time, taking into account the genes available in the unlabeled
dataset. In this case, the command line tool to run is

```
python -m cell_type_mapper.cli.map_to_on_the_fly_markers \
--query_path /path/to/unlabeled_data.h5ad \
--extended_result_path output.json \
--csv_result_path output.csv \
--n_processors 4 \
--cloud_safe False \
--precomputed_stats.path /path/to/precomputed_stats.20231120.sea_ad.MTG.h5 \
--type_assignment.normalization raw \
--query_markers.n_per_utility 15 \
--reference_markers.log2_fold_min_th 0.5
```

A brief discussion of this tool, with links to documentation discussing how
it selects marker genes,
[is here](https://github.com/AllenInstitute/cell_type_mapper/blob/main/docs/mapping_cells.md#finding-markers-at-runtime) under "Finding markers at runtime".
