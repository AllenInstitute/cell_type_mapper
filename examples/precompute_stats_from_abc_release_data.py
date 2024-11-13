"""
This script creates the precomputed_stats.h5 file from the ABC Atlas
release data as it was store on the local Allen Institute cluster
circa June 30 2023.

It takes an hour or two to run.
"""

from cell_type_mapper.cli.precompute_stats_abc import (
    PrecomputationABCRunner)

from cell_type_mapper.utils.utils import get_timestamp

import argparse
import pathlib
import time

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()
    if args.output_path is None:
        raise RuntimeError("Must specify --output_path")

    data_dir = pathlib.Path(
        "/allen/programs/celltypes/workgroups/rnaseqanalysis/lydian/ABC_handoff/metadata")
    assert data_dir.is_dir()

    # Paths to CSV files encoding the cell types taxonomy

    cluster_annotation = data_dir / "WMB-taxonomy/20230630/cluster_annotation_term.csv"
    assert cluster_annotation.is_file()

    cluster_membership = cluster_annotation.parent / "cluster_to_cluster_annotation_membership.csv"
    assert cluster_membership.is_file()

    cell_metadata = data_dir / "WMB-10X/20230630/cell_metadata.csv"
    assert cell_metadata.is_file()

    cell_metadata = str(cell_metadata.resolve().absolute())
    cluster_membership = str(cluster_membership.resolve().absolute())
    cluster_annotation = str(cluster_annotation.resolve().absolute())

    # hierarchy of the levels in the cell types taxonomy
    hierarchy=[
            "CCN20230504_CLAS",
            "CCN20230504_SUBC",
            "CCN20230504_SUPT",
            "CCN20230504_CLUS"]

    # assemble a list of all of the h5ad files containing all of the
    # raw counts cell by gene expression matrices for the data release
    h5ad_dir_1 = pathlib.Path(
        "/allen/programs/celltypes/workgroups/rnaseqanalysis/lydian/ABC_handoff/expression_matrices/WMB-10Xv2/20230630")
    h5ad_list_1 = [
        str(n.resolve().absolute())
        for n in h5ad_dir_1.iterdir()
        if n.name.endswith('raw.h5ad')]

    h5ad_dir_2 = pathlib.Path(
        "/allen/programs/celltypes/workgroups/rnaseqanalysis/lydian/ABC_handoff/expression_matrices/WMB-10Xv3/20230630")
    h5ad_list_2 = [
        str(n.resolve().absolute())
        for n in h5ad_dir_2.iterdir()
        if n.name.endswith('raw.h5ad')]

    h5ad_list = h5ad_list_1+h5ad_list_2

    output_path = f"precompute_abc_{get_timestamp().replace('-','')}.h5"

    config = {
        'h5ad_path_list': h5ad_list,
        'normalization': 'raw',
        'cell_metadata_path': cell_metadata,
        'cluster_annotation_path': cluster_annotation,
        'cluster_membership_path': cluster_membership,
        'hierarchy': hierarchy,
        'output_path': args.output_path}

    t0 = time.time()
    runner = PrecomputationABCRunner(args=[], input_data=config)
    runner.run()
    dur = time.time()-t0
    print(f"wrote {args.output_path}")
    print(f"that took {dur:.2e} seconds")


if __name__ == "__main__":
    main()
