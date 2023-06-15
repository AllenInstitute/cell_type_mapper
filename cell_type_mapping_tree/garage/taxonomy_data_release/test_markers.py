import pathlib

from hierarchical_mapping.gene_id.gene_id_mapper import (
    GeneIdMapper)

def main():
    marker_dir = pathlib.Path(
        '/allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/Taxonomies/AIT17.0_mouse/Templates/marker_list_on_nodes')
    assert marker_dir.is_dir()

    data_dir = pathlib.Path(
        "/allen/programs/celltypes/workgroups/rnaseqanalysis/lydian/ABC_handoff/metadata/WMB-taxonomy/20230630")
    assert data_dir.is_dir()

    cluster_membership = data_dir / "cluster_to_cluster_annotation_membership.csv"
    assert cluster_membership.is_file()

    hierarchy=[
        "CCN20230504_CLAS",
        "CCN20230504_SUBC",
        "CCN20230504_CLUS"]

if __name__ == "__main__":
    main()
