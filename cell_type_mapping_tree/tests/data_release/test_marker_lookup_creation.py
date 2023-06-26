from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from hierarchical_mapping.marker_lookup.marker_lookup import (
    marker_lookup_from_tree_and_csv)

def test_marker_creation_function(
        marker_gene_csv_dir,
        expected_marker_lookup_fixture,
        cluster_membership_fixture,
        cell_metadata_fixture,
        cluster_annotation_term_fixture):

    taxonomy_tree = TaxonomyTree.from_data_release(
        cell_metadata_path=cell_metadata_fixture,
        cluster_membership_path=cluster_membership_fixture,
        cluster_annotation_path=cluster_annotation_term_fixture,
        hierarchy=['class', 'subclass', 'supertype', 'cluster'])

    actual = marker_lookup_from_tree_and_csv(
        taxonomy_tree=taxonomy_tree,
        csv_dir=marker_gene_csv_dir)

    assert actual == expected_marker_lookup_fixture
