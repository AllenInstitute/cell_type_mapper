### Taxonomy Tree serialization

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

Looking into the encoding of the tree itself, we might see something like

```
>>> result_dict['taxonomy_tree']['CCN20230504_SUBC']
{
  "CCN20230504_SUBC_305": [
    "CCN20230504_SUPT_1041"
  ],
  "CCN20230504_SUBC_306": [
    "CCN20230504_SUPT_1042",
    "CCN20230504_SUPT_1043",
    "CCN20230504_SUPT_1044",
    "CCN20230504_SUPT_1045"
  ]
...
}
```

indicating that, at the taxonomic level `CCN20230504_SUBC`, the node
`CCN20230504_SUBC_305` has only one child node, while the node
`CCN20230504_SUBC_306` has four child nodes.
