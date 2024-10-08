# Output file contents

**Note:** There is an example Jupyter notebook in this repository at

[examples/explore_mapping_results.ipynb](https://github.com/AllenInstitute/cell_type_mapper/blob/main/examples/explore_mapping_results.ipynb)

which walks the user through downloading actual data from the NEMO
archive, formatting it for submission to the on-line MapMyCells tool,
and examining the results of the mapping. Running that notebook
will provide a more practical, hands-on complement to reading this
documentation.

=====

`cell_type_mapper.cli.from_specified_markers` produces two output files.
An optional CSV output file, and an "extended" JSON output file (though this
file is actually the one that is always created). The JSON file contains all
of the data in the CSV file along with some extra confidence metrics and
metadata from the mapping run. Below, we document the contents of each file.

## CSV output file

If the `csv_result_path` config parameter is specified, then a CSV file
will be written with some basic mapping results. The first three (or four,
depending) lines contain metadata about the mapping run. These lines are
prefixed with a `#`. The first line indicates the name of the extended
JSON output file associated with this CSV file. The second line denotes
the hierarchy of taxonomy levels to which the data was mapped. The third
line denotes the human-readable hierarchy of taxonomy levels to which the
data was mapped (assuming that differs from the machine-readable hierarchy).
The final line contains metadata describing the version of the software
used to generate these results. An example of these three lines would be
something like this

```
# metadata = hier.json
# taxonomy hierarchy = ["CCN20230504_CLAS", "CCN20230504_SUBC", "CCN20230504_CLUS"]
# readable taxonomy hierarchy = ["class", "subclass", "cluster"]
# algorithm: 'hierarchical'; codebase: http://github.com/AllenInstitute/cell_type_mapper; version: 0.0.1
```

The next line defines the column headers for the CSV. Subsequent lines contain
the actual mapping results. Each row is a cell in the mapped dataset. The first
column is the ID of the cell as read directly from the input dataset. Subsequent
columns are, for each level in the taxonomy tree

```
thisLevel_label, thisLevel_name, thisLevel_bootstrapping_probability
```

where `label` is the machine-readable identifier of the taxonomic node assigned
to the cell at this level of the taxonomy, `name` is the human-readable name
of the assigned node, and `bootstrapping_probability` is a metric of confidence
in the assignment. It is the fraction of bootstrap iterations that chose
the assigned node at that level of the taxonomy. **Note:** in cases where
`bootstrapping_iteration=1`, there is no "fraction of bootstrap iterations" and
`thisLevel_bootstrapping_probability` is replaced with
`thisLevel_correlation_coefficient`, which is the Pearson's correlation
coefficient between the gene expression profile of the cell and the
gene expression profile of the assigned node *in the marker genes appropriate
to that taxonomic node.*

For the leaf node of the taxonomy (`cluster` in our example above), we also
return `thisLevel_alias`, which is another, theortically universally unique
identifier for that taxonomic node.

## JSON output file

The extended JSON output file is written to the location specified by the config
parameter `extended_result_path`. It is the JSON serialization of a dict
containing both the results of the mapping and metadata associated with the
mapping run. In Python, it can be loaded into memory with

```
import json
result_dict = json.load(open('path/to/result/file.json', 'rb'))
```

The key-vale pairs of the resulting dict are

- `results`: the actual results of the mapping
- `config`: a dict containing the config parameters for this mapping run
- `log`: a list of strings containing log messages produced by this mapping run
- `marker_genes`: a dict encoding the marker genes used for this mapping run
- `taxonomy_tree`: a dict encoding the taxonomy to which the data was mapped

Below we further document these objects (except for `log` and `config`, which
should be self explanatory).

### taxonomy_tree


This is a dict that is a serialization of the taxonomy tree to which the
data was mapped. It can be interpreted as is or used to instantiate a
`TaxonomyTree` object with. The particulars of this representation
of a taxonomy tree are documented
[here.](input_data_files/taxonomy_tree.md)

### marker_genes

`result_dict['marker_genes']` is a dict indicating the marker genes used to
discriminate between the children of each parent in the taxonomy tree. The
keys of this dict are the parents in the form `taxonomic_level/taxonomic_node`
e.g.

```
'CCN20230504_CLAS/CCN20230504_CLAS_15'
'CCN20230504_SUBC/CCN20230504_SUBC_259'
'CCN20230504_SUBC/CCN20230504_SUBC_269'
```
they key `'None'` indicates the root of the taxonomy tree. The values in the
dict are lists of genes (identified with the gene identifiers used in the
reference dataset, probably EnsemblID) used as markers when choosing between
the children of that parent node.

### results

`result_dict['results']` points to a list of dicts. Each dict represents
a cell in the data that was mapped and encodes the result of the mapping.
For each level in the taxonomy tree there is recorded

- `'assignment'`: the taxonomic node chosen for this cell
- `'bootstrapping_probability'`: the fraction of bootstrap iterations that
selected the assigned taxonomic node
- `'aggregate_probability'`: this is the product of `boostrapping_probability`
for all levels of the taxonomy starting at the grossest level and
ending at the current level, i.e. if your taxonomy has levels
`['class', 'subclass', 'cluster']`, then the `aggregate_probability` at
the `subclass` level is the product of the `class` level
`bootstrapping_probability` with the `subclass` level
`bootstrapping_probability`; the `aggregate_probability` at the `cluster`
level is the product of the `bootstrapping_probability` at all three levels.
- `'avg_correlation`': the average Pearson's correlation coefficient between
the gene profile of the cell and the average gene profile of the chosen
taxonomic node *in the marker genes appropriate for that node.* The average
is taken over only those bootstrap iterations that selected the assigned
node.
- `'directly_assigned'`: this is a boolean. If `True`, then the cell type
was assigned directly by the cell type mapper. If `False`, the cell type
was inferred from a directly assigned child. This may occur if, for instance,
you run the cell type mapper with `flatten = True`, in which case each cell
is mapped directly to the leaf node of the taxonomy tree. In this case,
higher level nodes of the tree are still assigned, but they are inferred
based on the inheritance of the assigned leaf node (e.g. "I know the cell
has been assigned directly to subtypeA; it was not directly assigned to
a supertype; however, since subtypeA is a child of supertypeB in my taxonomy,
I can infer that the cell must also be a member of supertypeB").
In this case, the `bootstrapping_probability`, 	`aggregate_probability`,
and `avg_correlation` values are propagated up the tree from the
directly assigned child node for convenience.

The cell is identified by the `cell_id` key in the dict, e.g.

```
{
  "CCN20230504_CLAS": {
    "assignment": "CCN20230504_CLAS_25",
    "bootstrapping_probability": 0.74,
    "aggregate_probability": 0.74,
    "avg_correlation": 0.5735289275007436,
    "directly_assigned": True
  },
  "CCN20230504_SUBC": {
    "assignment": "CCN20230504_SUBC_269",
    "bootstrapping_probability": 0.59,
    "aggregate_probability": 0.44,
    "avg_correlation": 0.6041588697199276,
    "directly_assigned": True
  },
  "CCN20230504_CLUS": {
    "assignment": "CCN20230504_CLUS_4975",
    "bootstrapping_probability": 0.56,
    "aggregate_probability": 0.24,
    "avg_correlation": 0.6123913092110298,
    "directly_assigned": True
  },
  "cell_id": "1015221640100510476"
}
```

At each level, there will also be optional fields `runner_up_assignment`,
`runner_up_correlation`, and `runner_up_probability`. These will map
to lists of the N (if requested) next most likely assignments as
ranked according to `bootstrapping_probability`. If there were no
runners up, these will be empty lists.

**Note:** the runner up fields will be absent for any levels in the
taxonomy tree that were not directly assigned.
