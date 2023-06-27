import pytest

import h5py
import json

from hierarchical_mapping.utils.utils import (
    mkstemp_clean)

from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from hierarchical_mapping.marker_lookup.marker_lookup import (
    marker_lookup_from_tree_and_csv)

from hierarchical_mapping.cli.marker_cache_from_csv_dir import (
    MarkerCacheRunner)


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


def test_marker_creation_function_bad_dir(
        bad_marker_gene_csv_dir,
        cluster_membership_fixture,
        cell_metadata_fixture,
        cluster_annotation_term_fixture):

    taxonomy_tree = TaxonomyTree.from_data_release(
        cell_metadata_path=cell_metadata_fixture,
        cluster_membership_path=cluster_membership_fixture,
        cluster_annotation_path=cluster_annotation_term_fixture,
        hierarchy=['class', 'subclass', 'supertype', 'cluster'])

    with pytest.raises(RuntimeError, match="does not exist"):
        marker_lookup_from_tree_and_csv(
            taxonomy_tree=taxonomy_tree,
            csv_dir=bad_marker_gene_csv_dir)


def test_marker_creation_cli(
        marker_gene_csv_dir,
        expected_marker_lookup_fixture,
        cluster_membership_fixture,
        cell_metadata_fixture,
        cluster_annotation_term_fixture,
        tmp_dir_fixture):

    taxonomy_tree = TaxonomyTree.from_data_release(
        cell_metadata_path=cell_metadata_fixture,
        cluster_membership_path=cluster_membership_fixture,
        cluster_annotation_path=cluster_annotation_term_fixture,
        hierarchy=['class', 'subclass', 'supertype', 'cluster'])

    precompute_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precomputed_stats_',
        suffix='.h5')

    with h5py.File(precompute_path, 'w') as dst:
        dst.create_dataset(
            'taxonomy_tree',
            data=taxonomy_tree.to_str().encode('utf-8'))

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='marker_genes_',
        suffix='.json')

    config = {
        'precomputed_file_path': precompute_path,
        'marker_dir': str(marker_gene_csv_dir.resolve().absolute()),
        'output_path': output_path}

    runner = MarkerCacheRunner(args=[], input_data=config)
    runner.run()

    actual = json.load(open(output_path, 'rb'))

    # the +1 is for 'metadata'
    assert len(actual) == len(expected_marker_lookup_fixture) + 1
    assert 'metadata' in actual


def test_marker_creation_cli_mangled_gene(
        bad_marker_gene_csv_dir_2,
        expected_marker_lookup_fixture,
        cluster_membership_fixture,
        cell_metadata_fixture,
        cluster_annotation_term_fixture,
        tmp_dir_fixture):
    """
    Test that an error is raised if a marker gene
    cannot be mapped to EnsemblID
    """

    taxonomy_tree = TaxonomyTree.from_data_release(
        cell_metadata_path=cell_metadata_fixture,
        cluster_membership_path=cluster_membership_fixture,
        cluster_annotation_path=cluster_annotation_term_fixture,
        hierarchy=['class', 'subclass', 'supertype', 'cluster'])

    precompute_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precomputed_stats_',
        suffix='.h5')

    with h5py.File(precompute_path, 'w') as dst:
        dst.create_dataset(
            'taxonomy_tree',
            data=taxonomy_tree.to_str().encode('utf-8'))

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='marker_genes_',
        suffix='.json')

    config = {
        'precomputed_file_path': precompute_path,
        'marker_dir': str(bad_marker_gene_csv_dir_2.resolve().absolute()),
        'output_path': output_path}

    runner = MarkerCacheRunner(args=[], input_data=config)
    with pytest.raises(RuntimeError, match="genes had no mapping"):
        runner.run()
