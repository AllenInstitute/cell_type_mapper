"""
Test merger of precomputed stats files
"""

import pytest

import anndata
import h5py
import json
import numpy as np
import pandas as pd

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up
)

from cell_type_mapper.taxonomy.taxonomy_tree import TaxonomyTree
from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad
)
from cell_type_mapper.diff_exp.merge_precompute import (
    merge_precomputed_stats_files
)


@pytest.fixture(scope='session')
def tmp_dir_fixture(
        tmp_path_factory):
    result = tmp_path_factory.mktemp('precompute_file_merger_')
    yield result
    _clean_up(result)


def create_precompute_file(
        obs,
        var,
        column_hierarchy,
        rng,
        tmp_dir):

    n_genes = len(var)
    n_cells = len(obs)
    adata_path = mkstemp_clean(
        dir=tmp_dir,
        suffix='.h5ad'
    )
    adata = anndata.AnnData(
        var=var,
        obs=obs,
        X=rng.integers(0, 10, (n_cells, n_genes))
    )
    adata.write_h5ad(adata_path)
    precompute_path = mkstemp_clean(
        dir=tmp_dir,
        suffix='.h5ad'
    )
    precompute_summary_stats_from_h5ad(
        data_path=adata_path,
        column_hierarchy=column_hierarchy,
        normalization='raw',
        taxonomy_tree=None,
        output_path=precompute_path,
        n_processors=1
    )

    return precompute_path


@pytest.fixture
def taxonomyA_fixture(tmp_dir_fixture):

    gene_data = [
        {'gene_id': f'g_{ii}'} for ii in range(40)
    ]

    var = pd.DataFrame(gene_data).set_index('gene_id')

    taxonomy_lookup = {
        'aa': {
            'subclass': 'b',
            'class': 'A'
        },
        'bb': {
            'subclass': 'b',
            'class': 'A'
        },
        'cc': {
            'subclass': 'c',
            'class': 'A'
        },
        'dd': {
            'subclass': 'd',
            'class': 'B'
        },
        'ee': {
            'subclass': 'e',
            'class': 'B'
        }
    }
    n_cells = 47
    rng = np.random.default_rng(77123)
    cluster_list = sorted(taxonomy_lookup.keys())
    cell_records = []
    for ii in range(n_cells):
        cluster = rng.choice(cluster_list)
        cell = {
            'cell_id': f'c_{ii}',
            'cluster': cluster,
            'subclass': taxonomy_lookup[cluster]['subclass'],
            'class': taxonomy_lookup[cluster]['class']
        }
        cell_records.append(cell)
    obs = pd.DataFrame(cell_records).set_index('cell_id')

    precompute_path = create_precompute_file(
        obs=obs,
        var=var,
        column_hierarchy=['class', 'subclass', 'cluster'],
        rng=rng,
        tmp_dir=tmp_dir_fixture
    )

    return precompute_path


@pytest.fixture
def taxonomyB_fixture(tmp_dir_fixture):

    gene_data = [
        {'gene_id': f'g_{ii}'} for ii in range(12, 61, 1)
    ]

    var = pd.DataFrame(gene_data).set_index('gene_id')

    taxonomy_lookup = {
        'aa': {
            'cluster': 'b',
            'supercluster': 'A'
        },
        'bb': {
            'cluster': 'b',
            'supercluster': 'A'
        },
        'cc': {
            'cluster': 'c',
            'supercluster': 'A'
        },
        'dd': {
            'cluster': 'd',
            'supercluster': 'B'
        },
        'ee': {
            'cluster': 'e',
            'supercluster': 'B'
        }
    }
    n_cells = 118
    rng = np.random.default_rng(227573123)
    cluster_list = sorted(taxonomy_lookup.keys())
    cell_records = []
    for ii in range(n_cells):
        subcluster = rng.choice(cluster_list)
        cell = {
            'cell_id': f'c_{ii}',
            'subcluster': subcluster,
            'cluster': taxonomy_lookup[subcluster]['cluster'],
            'supercluster': taxonomy_lookup[subcluster]['supercluster']
        }
        cell_records.append(cell)
    obs = pd.DataFrame(cell_records).set_index('cell_id')

    precompute_path = create_precompute_file(
        obs=obs,
        var=var,
        column_hierarchy=['supercluster', 'cluster', 'subcluster'],
        rng=rng,
        tmp_dir=tmp_dir_fixture
    )

    return precompute_path


@pytest.fixture
def taxonomyC_fixture(tmp_dir_fixture):

    rng = np.random.default_rng(871100992)

    gene_data = [
        {'gene_id': f'g_{ii}'} for ii in range(61)
    ]
    rng.shuffle(gene_data)

    var = pd.DataFrame(gene_data).set_index('gene_id')

    taxonomy_lookup = {
        'aa': {
            'supercluster': 'A'
        },
        'bb': {
            'supercluster': 'A'
        },
        'cc': {
            'supercluster': 'A'
        },
        'dd': {
            'supercluster': 'B'
        },
        'ee': {
            'supercluster': 'B'
        },
        'ff': {
            'supercluster': 'B'
        },
        'gg': {
            'supercluster': 'C'
        }
    }
    n_cells = 213
    cluster_list = sorted(taxonomy_lookup.keys())
    cell_records = []
    for ii in range(n_cells):
        subcluster = rng.choice(cluster_list)
        cell = {
            'cell_id': f'c_{ii}',
            'leaf': subcluster,
            'supercluster': taxonomy_lookup[subcluster]['supercluster']
        }
        cell_records.append(cell)
    obs = pd.DataFrame(cell_records).set_index('cell_id')

    precompute_path = create_precompute_file(
        obs=obs,
        var=var,
        column_hierarchy=['supercluster', 'leaf'],
        rng=rng,
        tmp_dir=tmp_dir_fixture
    )

    return precompute_path


def test_merging_precomputed_stats_files(
        taxonomyA_fixture,
        taxonomyB_fixture,
        taxonomyC_fixture,
        tmp_dir_fixture):
    """
    Test merger of precomputed stats files
    """

    src_lookup = {
        'taxonomyA': taxonomyA_fixture,
        'taxonomyB': taxonomyB_fixture,
        'taxonomyC': taxonomyC_fixture
    }

    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='merged_stats_',
        suffix='.h5'
    )

    merge_precomputed_stats_files(
        src_lookup=src_lookup,
        dst_path=dst_path,
        clobber=True
    )

    expected_gene_set = set(
        [f'g_{ii}' for ii in range(12, 40, 1)]
    )

    # assemble set of expected leaf clusters
    expected_leaf_set = set()
    for taxonomy_name in src_lookup:
        with h5py.File(src_lookup[taxonomy_name], 'r') as src:
            cluster_to_row = json.loads(
                src['cluster_to_row'][()].decode('utf-8')
            )
            tree = TaxonomyTree.from_precomputed_stats(
                src_lookup[taxonomy_name]
            )
            leaf_level = tree.leaf_level
            for cl in cluster_to_row.keys():
                expected_leaf_set.add(f'{taxonomy_name}:{leaf_level}:{cl}')

    with h5py.File(dst_path, 'r') as actual:

        # make sure expected genes are prsent
        gene_names = json.loads(actual['col_names'][()].decode('utf-8'))
        assert set(gene_names) == expected_gene_set

        # make sure expected leaf nodes are present
        cluster_to_row = json.loads(
            actual['cluster_to_row'][()].decode('utf-8')
        )
        assert set(cluster_to_row.keys()) == expected_leaf_set

        actual_n_arr = actual['n_cells'][()]

        # loop over source files, verifying numerical contents
        for taxonomy_name in src_lookup:
            with h5py.File(src_lookup[taxonomy_name], 'r') as expected:

                expected_rows = json.loads(
                    expected['cluster_to_row'][()].decode('utf-8')
                )

                tree = TaxonomyTree(
                    data=json.loads(
                        expected['taxonomy_tree'][()].decode('utf-8')
                    )
                )
                leaf_level = tree.leaf_level

                # construct a map from gene name to idx in the expected file
                expected_gene_to_idx = {
                   g: ii
                   for ii, g in enumerate(
                       json.loads(expected['col_names'][()].decode('utf-8')))
                }

                # which columns do we expect to have made it through to the
                # merged data
                gene_idx = np.array(
                    [expected_gene_to_idx[g] for g in gene_names]
                )

                # loop over numerical datasets, verifying contents row by row
                for key in expected:
                    if key in ('col_names',
                               'cluster_to_row',
                               'metadata',
                               'taxonomy_tree'):
                        continue

                    assert key in actual

                    if key == 'n_cells':
                        nn = expected[key][()]
                        expected_n = {
                            cl: nn[expected_rows[cl]]
                            for cl in expected_rows
                        }

                        actual_n = {
                            cl: actual_n_arr[
                                cluster_to_row[
                                    f'{taxonomy_name}:{leaf_level}:{cl}'
                                ]
                            ]
                            for cl in expected_rows
                        }

                        assert actual_n == expected_n
                    else:
                        for cl in expected_rows.keys():
                            expected_data = (
                                expected[key][expected_rows[cl], :][gene_idx]
                            )

                            i_actual = cluster_to_row[
                                f'{taxonomy_name}:{leaf_level}:{cl}'
                            ]

                            actual_data = actual[key][i_actual, :]

                            np.testing.assert_allclose(
                                expected_data,
                                actual_data,
                                atol=0.0,
                                rtol=1.0e-6
                            )
