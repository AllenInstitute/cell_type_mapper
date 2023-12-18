# Taxonomy Tree serialization

A serialized representation of the cell type taxonomy tree is important
both as a part of the cell type mapper's input
([see the documentation for the precomputed_stats file](precomputed_stats_file.md)) and as a component of the extended JSON
ouptut produced by the cell type mapper (accessed via
`results['taxonomy_tree']`). Here we describe the serialized
representation of the taxonomy tree used by this codebase.

Ultimately, the taxonomy tree is represented as a dict. This dict can be
interpreted as is or used to instantiate a `TaxonomyTree` object with

```
from cell_type_mapper.taxonomy.taxonomy_tree import TaxonomyTree
tree = TaxonomyTree(data=results_dict['taxonomy_tree'])
```

**Note:** for size considerations, the part of the tree that maps individual
cells in the reference data to cell type clusters has been removed from the
output file produced by the cell type mapper. This will cause a warning to be
emitted when you instantiate the tree as above. You may neglect the warning.

The key-value pairs of the taxonomy tree dict are

- `metadata`: metadata about how the taxonomy tree was created (OPTIONAL)
- `hierarchy`: list of taxonomic levels from grossest to finest (e.g.
`["class", "subclass", "cluster"]`). (REQUIRED)
- `hierarchy_mapper`: a dict mapping machine-readable level names to
human-readable level names (OPTIONAL)
- `name_mapper`: a dict mapping machine-readable node names to human readable node names. (OPTIONAL)
- A key for each machine-readable level in the taxonomy that points to a dict
mapping nodes on that level of the taxonomy to their children. (REQUIRED)

## Required fields

Aside from `'hierarchy'` which maps to a list indicating, from most gross to
most fine, the order of the levels in the taxonomy tree, the only other
**required** fields in the taxonomy tree dict are those that encode the
parent-child relationships between the taxonomy nodes. Those fields are
themselves nested dicts, i.e. evel listed under `'hierarchy'` is itself
a key of the taxonomy tree dict pointing towards a dict which maps indivdiual
node names in the taxonomy to lists of their child nodes. For instance

```
{
    "hierarchy": ["class", "subclass", "cluster"],
    "class": {
        "class_01": ["subclass_01", "subclass_03"],
        "class_02": ["subclass_02"]
    },
    "subclass":{
        "subclass_01": ["cluster_01", "cluster_02"],
        "subclass_02": ["cluster_03", "cluster_05"],
        "subclass_03": ["cluster_04"]
    },
    "cluster": {
        "cluster_01": [],
        "cluster_02": [],
        "cluster_03": [],
        "cluster_04": [],
        "cluster_05": []
    }
}

```

Indicates that cluster_01 is a child of subclass_01 which is a child of
class_01. clusetr_02 is also a child of subclass_01. cluster_03 is a child of
subclass_02 which is a child of class_02. etc.

The code does not care about the mapping of leaf nodes ("clusters" in the
above example) to individual cells, but there does need to be a `"cluster"`
dict with the appropriate keys.


## Optional fields

Here we will document the optional fields in the schema by example. This section
is probably only useful for interpreting the output of the online MapMyCells
tool. Users who are ingesting their own taxonomy into this codebase should not
concern themselves with these fields.

Looking at `hierarchy_mapper` we see something like

```
>>> result_dict['taxonomy_tree']['hierarchy_mapper']
{'CCN20230504_CLUS': 'cluster', 'CCN20230504_SUPT': 'supertype', 'CCN20230504_SUBC': 'subclass', 'CCN20230504_CLAS': 'class'}
```

corresponding to

```
>>> result_dict['taxonomy_tree']['hierarchy']
['CCN20230504_CLAS', 'CCN20230504_SUBC', 'CCN20230504_SUPT', 'CCN20230504_CLUS']
```

which shows the order, from largest parent down to leaf, of inheritance of
levels in this taxonomy tree. Looking at `name_mapper`, we might see something like this

```
>>> result_dict['taxonomy_tree']['name_mapper']['CCN20230504_CLUS']['CCN20230504_CLUS_5046']
{'name': '5046 MOB-mi Frmd7 Gaba', 'alias': '1376'}
```

indicating that the node at level `CCN20230504_CLUS` with machine-readable
label `CCN20230504_CLUS_5046` can also be referred to by
"name" `5046 MOB-mi Frmd7 Gaba` and by "alias" `1376`.
