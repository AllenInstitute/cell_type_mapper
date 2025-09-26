import pytest

import anndata
import marshmallow
import h5py
import json
import numpy as np
import pandas as pd
import warnings

import cell_type_mapper.utils.utils as ctm_utils
import cell_type_mapper.utils.anndata_utils as anndata_utils
import cell_type_mapper.taxonomy.taxonomy_tree as tree_module

from cell_type_mapper.cli.from_specified_markers import (
    FromSpecifiedMarkersRunner
)


def test_hann_cli(
        noisy_precomputed_stats_fixture,
        noisy_marker_gene_lookup_fixture,
        noisy_raw_query_h5ad_fixture,
        tmp_dir_fixture):
    """
    Mostly a smoketest to make sure we can run the HANN algorithm
    through the CLI tool.
    """
    bootstrap_iteration = 57
    dst_path = ctm_utils.mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='hann_from_cli_',
        suffix='.h5'
    )
    config = {
        "query_path": str(noisy_raw_query_h5ad_fixture),
        "precomputed_stats": {
            "path": str(noisy_precomputed_stats_fixture)
        },
        "query_markers": {
            "serialized_lookup": str(noisy_marker_gene_lookup_fixture)
        },
        "type_assignment": {
            "n_processors": 3,
            "bootstrap_factor": 0.5,
            "bootstrap_iteration": bootstrap_iteration,
            "algorithm": "hann",
            "normalization": "raw"
        },
        "hdf5_result_path": dst_path
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=config
        )
        runner.run()

    obs = anndata_utils.read_df_from_h5ad(
        config['query_path'],
        df_name='obs'
    )
    tree = tree_module.TaxonomyTree.from_precomputed_stats(
        config['precomputed_stats']['path']
    )

    with h5py.File(dst_path, "r") as src:
        votes = src["votes"][()]
        cell_id = src["cell_identifiers"][()]
        cluster_id = src["cluster_identifiers"][()]
        _ = json.loads(src['metadata'][()].decode('utf-8'))

    np.testing.assert_array_equal(
        desired=obs.index.values,
        actual=[c.decode('utf-8') for c in cell_id]
    )
    vote_row_count = votes.sum(axis=1)
    np.testing.assert_array_equal(
        actual=vote_row_count,
        desired=bootstrap_iteration*np.ones(len(obs))
    )
    unq_votes = np.unique(votes)
    assert len(unq_votes) > 10

    np.testing.assert_array_equal(
        desired=tree.nodes_at_level(tree.leaf_level),
        actual=[c.decode('utf-8') for c in cluster_id]
    )


@pytest.fixture(scope='module')
def query_for_err_fixture(tmp_dir_fixture):
    """
    query h5ad file for testing config errors
    (to make sure we do not write to obsm)
    """
    h5ad_path = ctm_utils.mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad'
    )

    n_cells = 5
    n_genes = 12
    obs = pd.DataFrame(
        [{'cell': f'c{ii}'} for ii in range(n_cells)]).set_index('cell')
    var = pd.DataFrame(
        [{'gene': f'g{ii}'} for ii in range(n_genes)]).set_index('gene')
    adata = anndata.AnnData(
        obs=obs,
        var=var,
        X=np.zeros((n_cells, n_genes), dtype=float)
    )

    adata.write_h5ad(h5ad_path)
    return h5ad_path


@pytest.mark.parametrize(
    "use_csv_path, use_json_path, use_obsm_key, use_hdf5_path, msg_suffix",
    ([True, False, False, True, "; you specified csv_result_path"],
     [False, True, False, True, "; you specified extended_result_path"],
     [False, False, True, True, "; you specified obsm_key"],
     [True, True, False, True,
      "; you specified csv_result_path"
      "; you specified extended_result_path"],
     [True, True, True, True,
      "; you specified csv_result_path"
      "; you specified extended_result_path"
      "; you specified obsm_key"],
     [True, False, False, False,
      "; you specified csv_result_path; you did not specify hdf5_result_path"],
     [False, True, False, False,
      "; you specified extended_result_path"
      "; you did not specify hdf5_result_path"],
     [False, False, True, False,
      "; you specified obsm_key; you did not specify hdf5_result_path"]
     )
)
def test_hann_cli_config_errors(
        noisy_precomputed_stats_fixture,
        noisy_marker_gene_lookup_fixture,
        query_for_err_fixture,
        tmp_dir_fixture,
        use_csv_path,
        use_json_path,
        use_obsm_key,
        use_hdf5_path,
        msg_suffix):
    """
    Test that correct errors are raised when specifying the wrong
    outputs for HANN CLI
    """

    if use_csv_path:
        csv_path = ctm_utils.mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.csv'
        )
    else:
        csv_path = None

    if use_json_path:
        json_path = ctm_utils.mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.json'
        )
    else:
        json_path = None

    if use_obsm_key:
        obsm_key = 'test_obsm'
    else:
        obsm_key = None

    if use_hdf5_path:
        hdf5_path = ctm_utils.mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5'
        )
    else:
        hdf5_path = None

    bootstrap_iteration = 57

    config = {
        "query_path": str(query_for_err_fixture),
        "precomputed_stats": {
            "path": str(noisy_precomputed_stats_fixture)
        },
        "query_markers": {
            "serialized_lookup": str(noisy_marker_gene_lookup_fixture)
        },
        "type_assignment": {
            "n_processors": 3,
            "bootstrap_factor": 0.5,
            "bootstrap_iteration": bootstrap_iteration,
            "algorithm": "hann",
            "normalization": "raw"
        },
        "hdf5_result_path": hdf5_path,
        "csv_result_path": csv_path,
        "extended_result_path": json_path,
        "obsm_key": obsm_key
    }

    match = "HANN algorithm can only output to hdf5_result_path"
    match += msg_suffix

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        with pytest.raises(marshmallow.ValidationError, match=match):
            FromSpecifiedMarkersRunner(
                args=[],
                input_data=config
            )
