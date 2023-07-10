# Running the cell type mapping pipeline

This document explains how to run the code in this library to map unlabeled
cell by gene expression data onto the Allen Institute cell type taxonomy.
There are several steps in the pipeline that, in theory, only need to be run
once per (reference data, cell type taxonomy) pair. Those steps will be
pointed out and explained.

## General pipeline

There are three steps to running this pipeline

- Computing the average gene expression profile per cell type cluster in
the cell type taxonomy. **This step only needs to be run once per reference
dataset and produces a file that can be reused multiple times.**
- Identifying marker genes corresponding to the parent nodes in the cell type
taxonomy. **This step only needs to be run once per reference dataset and
produces a file that can be used multiple times.**
- Mapping unlabeled data onto the reference taxonomy using the executable
provided [here](../src/hierarchical_mapping/cli/from_specified_markers.py)
