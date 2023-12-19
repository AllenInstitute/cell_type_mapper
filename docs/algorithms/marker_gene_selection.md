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
clusterA and the mean `log2(CPM+1) expression of gene `i`
in cluterB).

A gene is considered a marker gene for discriminating between
clusterA and clusterB if

```
q1 >= 0.5
qdiff >= 0.7
fold_change > = 1
```

These thresholds are set using the config parameters
`q1_th`, `qdiff_th`, and `log2_fold_th`.

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
place it in a three dimensional (`q1`, `q_diff`, `fold_change`) space.
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
