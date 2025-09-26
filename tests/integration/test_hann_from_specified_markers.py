import h5py
import json
import numpy as np
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
        metadata = json.loads(src['metadata'][()].decode('utf-8'))
        
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
