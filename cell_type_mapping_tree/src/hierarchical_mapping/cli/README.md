## from_specified_markers.py

This is a CLI tool for running hierarchical mapping with a pre-specified
set of marker genes. It can be run like

```
python -m hierarchical_mapping.cli.from_specified_markers \
--config_path path/to/config.json \
--result_path path/to/output/file.json \
--log_path /optional/path/to/log/file.json
```

**Note**:

- `result_path` can be specified in the config file. If `result_path` is specified at the command line, it will override whatever value is in the config file.
- The log recorded in `log_path` is currently also stored in the output file, to it is not strictly necessary to specify a `log_path`.

The config file is a JSON representation of a dict containing all of the configuration
parameters expected by the tool. Those parameters are:


```
{
  "result_path": "/optional/path/to/output/file/to/be/created.json",
  "query_path": "/path/to/h5ad/file/containing/query/dataset.h5ad",
  "tmp_dir": "/path/to/fast/tmp/dir/where/data/will/be/copied/for/processing/",
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

- If the file pointed to by `precomputed_stats.path` exists, then that file
will just be copied to `tmp_dir` and used. If it does not exist, precomputed
stats will be generated from the reference dataset and stored in
`precomputed_stats.path`.

- Strictly speaking, the taxonomy tree is read from the precomputed stats file
to preserve consistency so, if `precomputed_stats.path` points to a file that
exists, then `precomputed_stats.taxonom_tree` is superflous.

## hierarchical_mapping.py

This is the CLI tool to be run in the case where marker genes have not been
pre-specified. It is currently under development and should **not** be used.
