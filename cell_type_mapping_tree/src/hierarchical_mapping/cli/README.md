## from_specified_markers.py

This is a CLI tool for running hierarchical mapping with a pre-specified
set of marker genes. It can be run like

```
python -m hierarchical_mapping.cli.from_specified_markers \
--input_json path/to/config.json
```

The config file is a JSON representation of a dict containing all of the
configuration parameters expected by the tool. Those parameters are:


```
{
  "result_path": "/optional/path/to/output/file/to/be/created.json",
  "query_path": "/path/to/h5ad/file/containing/query/dataset.h5ad",
  "tmp_dir": "/path/to/fast/tmp/dir/where/data/will/be/copied/for/processing/",
  "max_gb": gigabytes_available_for_matrix_transform,
  "precomputed_stats": {
    "reference_path": "/path/to/h5ad/file/containing/reference/dataset.h5ad",
    "path": "/path/to/hdf5/file/of/precomputed/stats/to/be/created.h5",
    "normalization": "raw" or "log2CPM", depending on the normalization of the reference dataset
    "taxonomy_tree": "/path/to/JSONized/representation/of/taxonomy/tree.json"
  },
  "query_markers": {
    "serialized_lookup": "/path/to/JSONized/representation/of/marker/genes.json"
  },
  "type_assignment": {
    "n_processors": number_of_cores_to_use (an int),
    "bootstrap_factor": factor_by_which_to_downsample_in_bootstrapping (0, 1],
    "bootstrap_iteration": number_of_boostrapping_terations (an int),
    "rng_seed": seed_for_rando_number_generator (an int),
    "chunk_size": number_of_query_rows_to_load_at_once (an int),
    "normalization": "raw" or "log2CPM" depending on normalization of query dataset
  }
}
```

**Note**:

- You can see a help message defining these parameters by running

```
python -m hierarchical_mapping.cli.from_specified_markers --help
```

- If the file pointed to by `precomputed_stats.path` exists, then that file
will just be copied to `tmp_dir` and used. If it does not exist, precomputed
stats will be generated from the reference dataset and stored in
`precomputed_stats.path`.

- Strictly speaking, the taxonomy tree is read from the precomputed stats file
to preserve consistency so, if `precomputed_stats.path` points to a file that
exists, then `precomputed_stats.taxonom_tree` is superflous.

### Running programmatically

This modules uses the
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

## hierarchical_mapping.py

This is the CLI tool to be run in the case where marker genes have not been
pre-specified. It is currently under development and should **not** be used.
