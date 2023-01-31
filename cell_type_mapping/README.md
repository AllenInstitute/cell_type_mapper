The scripts in this directory represent an implementation of a very simple
algorithm to map cells with known gene expression and unknown cell type
onto a known set of cell type clusters.

First, run aggregate_cell_clusters.py on the dataset with the known
cell types. This script will take the cells in that dataset and use
them to create a cluster-by-gene dataset in which each cluster is
represented by the average gene expression profile of all cells in that
cluster.

Then run map_to_clusters.py using the cells of unknown type as one input
and the cluter-by-gene dataset produced by aggregate_cell_clusters.py
as another. This script will produce an .h5 file which contains a
cell-by-cluster array which is the Pearson's correlation coefficient
(in gene space) of the unknown cells with the average cluster gene
exprssion profiles.
