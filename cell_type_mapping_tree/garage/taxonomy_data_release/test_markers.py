import pathlib

from hierarchical_mapping.gene_id.gene_id_mapper import (
    GeneIdMapper)

from hierarchical_mapping.taxonomy.data_release_utils import (
    get_alias_mapper)

from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)

def main():
    marker_dir = pathlib.Path(
        '/allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/Taxonomies/AIT17.0_mouse/Templates/marker_list_on_nodes')
    assert marker_dir.is_dir()

    data_dir = pathlib.Path(
        "/allen/programs/celltypes/workgroups/rnaseqanalysis/lydian/ABC_handoff/metadata/")
    assert data_dir.is_dir()

    cluster_membership = data_dir / "WMB-taxonomy/20230630/cluster_to_cluster_annotation_membership.csv"
    assert cluster_membership.is_file()

    cluster_annotation = data_dir / "WMB-taxonomy/20230630/cluster_annotation_term.csv"
    assert cluster_annotation.is_file()

    cell_metadata = data_dir / "WMB-10X/20230630/cell_metadata.csv"
    assert cell_metadata.is_file()

    hierarchy=[
        "CCN20230504_CLAS",
        "CCN20230504_SUBC",
        "CCN20230504_CLUS"]

    level_to_idx = {n:ii+1 for ii, n in enumerate(hierarchy)}

    alias_mapper = get_alias_mapper(
        csv_path=cluster_membership,
        valid_term_set_labels=hierarchy,
        alias_column_name='cluster_annotation_term_name',
        strict_alias=True)

    taxonomy_tree = TaxonomyTree.from_data_release(
        cell_metadata_path=cell_metadata,
        cluster_annotation_path=cluster_annotation,
        cluster_membership_path=cluster_membership,
        hierarchy=hierarchy)

    #parent_list =

    print("all done")

if __name__ == "__main__":
    main()
