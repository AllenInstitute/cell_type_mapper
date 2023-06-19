from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)

import pathlib

def main():

    data_dir = pathlib.Path(
        "/allen/programs/celltypes/workgroups/rnaseqanalysis/lydian/ABC_handoff/dataframes")
    assert data_dir.is_dir()

    cluster_annotation = data_dir / "WMB-taxonomy/20230630/cluster_annotation_term.csv"
    assert cluster_annotation.is_file()

    cluster_membership = cluster_annotation.parent / "cluster_to_cluster_annotation_membership.csv"
    assert cluster_membership.is_file()


    cell_metadata = data_dir / "WMB-10X/20230630/cell_metadata.csv"
    assert cell_metadata.is_file()

    tree = TaxonomyTree.from_data_release(
        cell_metadata_path=cell_metadata,
        cluster_annotation_path=cluster_annotation,
        cluster_membership_path=cluster_membership,
        hierarchy=[
            "CCN20230504_CLAS",
            "CCN20230504_SUBC",
            "CCN20230504_SUPT",
            "CCN20230504_CLUS"])

    print("got tree")

if __name__ == "__main__":
    main()
