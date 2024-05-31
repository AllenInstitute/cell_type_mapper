# Marker gene selection

In this page we document the algorithm used to select marker genes
for the cell type mapper. Code implementing this algorithm
is provided in this codebase as discussed
[here.](../input_data_files/marker_gene_lookup.md#creating-the-marker-gene-lookup-table)
In this page we will discuss the meaning of the various config
parameters used by the code explained in that page.

As discussed in the page linked above, marker gene selection
occurs in two stages, "reference marker selection," which
finds all of the marker genes for all of the leaf node pairs
in the cell type taxonomy, and "query marker selection" which
subsamples the reference markers in an intelligent way so as
to produce a manageably small set of marker genes.
We discuss those algorithms separately below.

## Reference marker selection

This algorithm is based on the description of
"finding differentially expressed genes" in the Methods
section of
[Tasic et al. 2018.](https://doi.org/10.1038/s41586-018-0654-5)
The command line tool implementing this algorithm is
```
python -m cell_type_mapper.cli.reference_markers --help
```
Configuration parameters referenced below are arguments
of this tool.

For any given pair of clusters (i.e. leaf nodes in the cell
type taxonomy) clusterA and clusterB, genes are designated
as either markers or non-markers according to two filters.
A gene must successfully pass both filters to be considered
a marker gene discriminating between clusterA and clusterB.

### P-value filter

Using Student's t-test corrected for multiple hypotheses
according to the
[Holm-Bonferroni method](https://en.wikipedia.org/wiki/Holm-Bonferroni_method),
only those genes with a p-value < 0.01 indicating that they are
drawn from different distributions in clusterA and clusterB
are allowed to be markers that discriminate between the
two clusters.

The threshold for this test (0.01 above) is set with the
`p_th` configuration parameter.

### Penetrance filter

Define `P_iA` as the fraction of cells in clusterA that express
gene `i` at greater than 1 CPM. Define `P_iB` as the fraction of
cells in clusterB that express gene `i` at greater than 1 CPM.
The "penetrance filter" for marker genes considers a few statistics
```
q1 = max(P_iA, P_iB)
q_diff = |P_iA-P_iB|/q1
```
and the `log2` fold change of the gene expression between
the two clusters (i.e. the absolute value of the difference
between the mean `log2(CPM+1)` expression of gene `i` in
clusterA and the mean `log2(CPM+1)` expression of gene `i`
in clusterB).

A gene is considered a marker gene for discriminating between
clusterA and clusterB if

```
q1 >= 0.5
qdiff >= 0.7
fold_change > = 1
```

These thresholds are set using the config parameters
`q1_th`, `qdiff_th`, and `log2_fold_th`.

#### Approximate penetrance filter

In practice, this "exact penetrance filter" is too restrictive.
It does not permit enough genes to be designated as markers, so
the code includes an approximate penetrance filter that can be
turned on (and is turned on by default) by passing in the
config parameter
```
--exact_penetrance false \
--n_valid N
```
where `N` is some number greater than 0.

In the event that the approximate penetrance filter is turned on,
the code will take each gene that passes the P-value filter and
place it in a three dimensional `(q1, q_diff, fold_change)` space.
It will then find a sphere centered on `(0.5, 0.7, 1.0)` that contains
at least `n_valid` genes and accept all genes that fall within
that sphere, subject to the absolute requirement that

```
q1 > q1_min_th
&& qdiff > qdiff_min_th
&& fold_change > log2_fold_min_th
```
where the `*_min_th` are also config parameters.
(Obviously, all genes that pass the exact penetrance filter
are also accepted as markers.)

## Query marker selection

Now that we have found all of the marker genes in the
reference dataset, we need to downsample them to produce
a set of markers that is tractable for use in querying
(i.e. for use in mapping an actual unlabeled dataset.)
The command line tool implementing this algorithm is
```
python -m cell_type_mapper.cli.query_markers --help
```
Configuration parameters referenced below are arguments
of this tool. The key config parameter for this tool
is `n_per_utility`. At each parent node in the taxonomy
tree, there is a subset of cell type clusters (leaf nodes)
that needs to be considered (this will be defined below).
The algorithm tries to select markers at that parent node
such that for each of those (clusterA, clusterB) pairs,
there are `n_per_utility` marker genes that are up-regulated
in clusterA and `n_per_utility` marker genes that are
up-regulated in clusterB.

### Cluster pairs of relevance

Here we define the cell type cluster pairs that
are considered relevant for marker gene selection
given a parent node in the cell type taxonomy.

Consider the following example taxonomy 
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
(see [here](../input_data_files/taxonomy_tree.md) for the
interprentation of this taxonomy tree).

Walk down the different parent nodes in the tree,
```
['None', 'class_01', 'class_02', 'subclass_01', 'subclass_02', 'subclass_03']
```
At each node, consider the "children of interest" that you are
selecting between
```
'None' -> ['class_01', 'class_02']
'class_01' -> ['subclass_01', 'subclass_03']
...
'subclass_01' -> ['cluster_01', 'cluster_02']
```
Consider only those cell type cluster pairs that descend from different
children of interest. So, if the parent node is `'None'`, then the cluster
pairs that need to be considered are
```
(cluster_01, cluster_03)
(cluster_02, cluster_03)
(cluster_04, cluster_03)
(cluster_01, cluster_05)
(cluster_02, cluster_05)
(cluster_04, cluster_05)
```
We donot consider, for instance,  the pair `(cluster_01, cluster_04)`
because those clusters both descend from `class_01`. However, if the
parent node is `class_01`, we consider the cluster pairs
```
(cluster_01, cluster_04)
(cluster_02, cluster_04)
```
ignoring `(cluster_01, cluster_02)` because they both descend from
`subclass_01`.

### Selecting query markers

Now that we have the lists of relevant cluster pairs for each
parent node in the taxonomy tree, we can select marker genes
for those parent nodes. Selection proceeds as follows

1. For each gene, count the number of `(clusterA, clusterB, direction)`
combinations for which it is a marker (`direction=up` if the gene is a
marker and is up-regulated in clusterA; `direction=down` if the gene is a
marker and is up-regulated in clusterB). This count is the gene's
"utility score."
2. Initialize a counter for each `(clusterA, clusterB, direction)`
combination.
3. Select the gene with the highest utility score and add it to the list
of marker genes for the parent node being considered. Increment the
counter for every `(clusterA, clusterB, direction)` combination for which
that gene is a marker.
4. Whenever a `(clusterA, clusterB, direction)` combination achieves
`n_per_utility` selected markers, find all the remaining genes which
are markers for that combination and decrement their utility score,
since we no longer need to worry about selecting markers for that
combination.
5. Continue until all `(clusterA, clusterB, direction)` combinations
have achieved `n_per_utility` markers, or all genes have utility scores
of zero (indicating that it was physically impossible to fulfill the
compement of marker genes given the data we have).

There are special conditions opposed on edge cases

- In cases where there are not `2*n_per_utility` reference markers for
a `(clusterA, clusterB)` pair, all of the reference markers for that
pair are chosen, regardless of which cluster they are up-regulated in.
- Similarly, if `(clusterA, clusterB, up)` achieves `n_per_utility`
markers but there are not `n_per_utility` reference markers for
`(clusterA, clusterB, down)`, step (4) will not trigger.
- If, in the regular course of running the algorithm,
a `(clusterA, clusterB)` pair achieves `2*n_per_utility` markers,
regardless of distribution along the `direction` axis, trigger
step (4) for all genes that are markers for that `(clusterA, clusterB)`
pair.

Effectively, priority is given to attempting to assign
`2*n_per_utility` markers to each `(clusterA, clusterB)` pair,
not to making sure markers are evenly distributed among
`(clusterA, clusterB, up)` and `(clusterA, clusterB, down)`.

### Special considerations

One of the end conditions in query marker selection is "we cannot find
enough markers for every `(clusterA, clusterB, direction)` combination."
This is affected by the total number of marker genes there are in the
reference marker dataset. The total number of markers in the reference
marker dataset is, in turn, affected by the interaction between the data
and the `reference_markers` config parameters
```
n_valid
q1_th
q1_min_th
qdiff_th
qdiff_min_th
log2_fold_th
log2_fold_min_th
```
(assuming `exact_penetrance = false`). At the very least, the `n_valid`
parameter passed to the reference marker tool should be twice the
`n_per_utility` parameter passed to the query marker tool.

It should also be noted that the reference marker dataset is effectively
just a binary mask. A gene either is or is not a marker. There is no saved
metric indicating how good a marker it is. This should give users caution
before they set `n_valid` too high or `*_min_th` too low, as these actions
could permit a lot of low quality markers to enter the reference marker set
and the query marker selection code has no way to know to ignore those low
quality markers.
