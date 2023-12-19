# The hierarchical mapping algorithm

This page describes the cell type mapping algorithm implemented in this
codebase. We will discuss it in the context of the command line tool
```
python -m cell_type_mapper.cli.from_specified_markers
```
which maps unlabeled data to a cell type taxonomy using
a specified lookup table of marker genes.

The mapping algorithm operates on the assumption that the
cell type taxonomy is a strict tree. The root node of the
tree is the set of all cells. The cells are then divided
into taxonomic types of different levels. Our canonical
example is to divide the cells into "classes," divide each
class into "subclasses," and finally divide each sublcass
into "clusters," though there can be any number
of levels in the taxonomy and their names are arbitrary.
The tree-like structure of the taxonomy means that all
cells that belong to the same subclass also belong
to the same class and all cells that belong to the same cluster
also belong to the same subclasses. Many subclasses can belong
to the same class. Many clusters can belong to the same subclass.
A subclass that belongs to a class is said to be a "child node"
of that class, etc.

The set of marker genes used by the algorithm is arranged such
that each parent node in the taxonomy (i.e. each node with more than
one child node) has a set of marker genes assigned to it that have
been deemed good at discriminating between that parent node's
children. This is documented in more detail
[here.](../input_data_files/marker_gene_lookup.md)

To map an unlabeled cell to the taxonomy, the algorithm
starts with the root node as the "designated parent" and

1. Take a random subset of 90% of the marker genes for the
designated parent node.
2. Correlate the cell's gene expression profile (`log2(CPM+1)`
normalization) with the average gene expression profile of
every leaf node ("cluster" in our canonical example) in the
taxonomy tree.
3. Select the cluster with which the cell has the highest
correlation coefficient.
4. Determine the child node of the designated parent from which
that cluster descends (i.e. if the designated parent is the root
node, transform the chosen cluster into its corresponding class)
and assign that child node 1 vote.
5. Repeat steps (1)-(4) 100 times using a different random 90% of
marker genes each time. Assign the cell to the child node that
receives the plurality of votes.
6. Repeat steps (1)-(5) using the child node chosen in (5) as the new
designated parent. Repeat this entire process until the cell has been
assigned to every level ("class", "subclass", "cluster") in the
taxonomy.
