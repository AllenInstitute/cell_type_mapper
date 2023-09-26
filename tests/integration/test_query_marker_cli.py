"""
Test the CLI tool for finding query markers
"""
import pytest

import anndata
import itertools
import json
import numpy as np
import pandas as pd


from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.cli.query_markers import (
    QueryMarkerRunner)



@pytest.mark.parametrize(
    "n_per_utility,drop_level,downsample_genes",
    itertools.product(
        (5, 3, 7, 11),
        (None, 'subclass'),
        (True, False)))
def test_query_marker_cli_tool(
        query_gene_names,
        ref_marker_path_fixture,
        precomputed_path_fixture,
        full_marker_name_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture,
        n_per_utility,
        drop_level,
        downsample_genes):

    if downsample_genes:
        rng = np.random.default_rng(76123)
        valid_gene_names = rng.choice(
            query_gene_names,
            len(query_gene_names)*3//4,
            replace=False)
    else:
        valid_gene_names = query_gene_names

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='h5ad_for_finding_query_markers_',
        suffix='.h5ad')

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='query_markers_',
        suffix='.json')

    var = pd.DataFrame(
        [{'gene_name': g}
         for g in valid_gene_names]).set_index('gene_name')
    adata = anndata.AnnData(var=var)
    adata.write_h5ad(h5ad_path)

    config = {
        'query_path': h5ad_path,
        'precomputed_path': precomputed_path_fixture,
        'reference_marker_path': ref_marker_path_fixture,
        'n_processors': 3,
        'n_per_utility': n_per_utility,
        'drop_level': drop_level,
        'output_path': output_path,
        'tmp_dir': str(tmp_dir_fixture.resolve().absolute())}

    runner = QueryMarkerRunner(
        args=[],
        input_data=config)
    runner.run()

    with open(output_path, 'rb') as src:
        actual = json.load(src)

    assert 'log' in actual
    log = actual['log']
    for level in taxonomy_tree_dict['hierarchy'][:-1]:
        for node in taxonomy_tree_dict[level]:
            log_key = f'{level}/{node}'
            if level == drop_level:
                assert log_key not in log
            else:
                assert log_key in log

    gene_ct = 0
    levels_found = set()
    actual_genes = set()
    for k in actual:
        if k == 'metadata':
            continue
        if k == 'log':
            continue
        if drop_level is not None:
            assert drop_level not in k
        levels_found.add(k.split('/')[0])
        for g in actual[k]:
            actual_genes.add(g)
            assert g in valid_gene_names
            gene_ct += 1
    assert gene_ct > 0

    expected_levels = set(['None'])
    for level in taxonomy_tree_dict['hierarchy'][:-1]:
        if level != drop_level:
            expected_levels.add(level)
    assert expected_levels == levels_found

    if not downsample_genes and n_per_utility == 7 and drop_level is None:
        assert actual_genes == set(full_marker_name_fixture)
    elif downsample_genes:
        assert actual_genes != set(full_marker_name_fixture)

    assert 'metadata' in actual
    assert 'timestamp' in actual['metadata']
    assert 'config' in actual['metadata']
    for k in config:
        assert k in actual['metadata']['config']
        assert actual['metadata']['config'][k] == config[k]
