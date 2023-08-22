# Cell Type Mapper

## Overview

This code provides a python package for mapping single sell RNA sequencing data
onto a cell type taxonomy such as that provided by the Allen Institute for Brain
Science.

## Installation

To install this library, clone the repository, then (ideally from a clean
python environment)

- Run `pip install -r requirements.txt` to install the required packages.
- Run `pip install -e .` from the root directory of this repository to install
this package itself.

This package has been tested extensively with python 3.9. We have no reason to
believe that it will not also run with python 3.8 and 3.10.

## Detailed documentation

The recommended workflow for running this code is
[here.](docs/mapping_cells.md)

Documentation of the output produced by this code can be found
[here.](docs/output.md)

## Level of support

We are providing this tool to the community and any and all who want to use it.
Issues and pull requests are welcome, however, this code is also intended
as part of the backend for the Allen Institute Brain Knowledge Platform. As
such, issues and pull requests may be declined if they interfere with
the functionality required to support that service.
