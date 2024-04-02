# Marker gene selection (alternative pipeline)

This page documents an alternative pipeline for creating the marker gene
lookup table used by the cell type mapper. It essentially follows the
same algorithm described [here,](marker_gene_selection.md) but with a different
computational workflow that is, in some respects, more efficient. We will
attempt to compare the original workflow with this workflow
[at the bottom of this page.](#advantages-to-the-new-workflow)

As described in the algorithm documentation page linked to above, there
are two steps to marker gene selection:

1. Selecting the "reference markers", i.e. finding every marker gene between
every pair of cell types in the cell type taxonomy
2. Selecting the "query markers", i.e. combinatorially finding the minimal
set of reference markers necessary to map data onto the cell type taxonomy

The statistics involved in delineating marker genes are all calculated and
applied in the reference marker step. We introduced the concept of
[approximate filters](marker_gene_selection.md/#approximate-penetrance-filter)
to ensure that enough marker genes are selected, even if they do not all
strictly comport with the statistical tests being used. The difficulty with
this scheme is that applying the approximate filter can potentially
depend on the unlabeled data being mapped (maybe you want to relax the
filters more in cases where not many strict marker genes have been measured
in the unlabeled set). Applying the approximate filter at the reference
marker step becomes a burden because this step is slower than query marker
selection by a factor of a few (for taxonomies of a few thousand cell types,
reference marker selection takes 2-3 hours; query marker selection takes
30-60 minutes). The new pipeline replaces the reference marker selection
step with a step we call "P-value mask calculation." Instead of storing a
sparse matrix of booleans indicating whether or not a gene is a marker for
discriminating between a pair of cell types, we compute and store a sparse
matrix of floats recording the parameter space distance of the (gene, cell type pair)
pair from the point in `(q1, q_diff, fold_change)` space (see the
[approximate penetrance filter section](marker_gene_selection.md/#approximate-penetrance-filter))
corresponding to genes that pass the strict statistical test. Genes that
pass the strict statistical test are given a parameter space distance of
`-1.0`. Genes that fail the minimum statistical thresholds and thus are
not to be considered marker genes have no parameter space distance recorded
at all. Query marker selection can adjust the threshold in this parameter
space distance that it accepts as valid to incorporate more or fewer
marker genes into the lookup table as desired.


## Computational workflow

The P-value mask file is computed with the command line tool
```
python -m cell_type_mapper.cli.compute_p_value_mask
```
which accepts most of the same command line arguments as the original
`reference_markers` command line tool.

Once the P-value mask file has been created, the marker gene lookup
table can be built using
```
python -m cell_type_mapper.cli.query_markers_from_p_value_mask
```

As with all command line tools in this repository, running with
the `--help` argument will print out the tool's call signature.

## Advantages to the new workflow
