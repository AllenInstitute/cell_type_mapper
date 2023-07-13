"""
test_specified_markers.py tested the from_specified_markers CLI on a
very "clean" dataset (i.e. each cell had one and only one cluster it
conceivable mapped onto.

In this module, we are going to generate some very noisy data and run
it through the from_specified_markers CLI. We won't do any checking of
the validity of the outputs, only their self-consistency. This test
was written after adding the 'runners_up' to the output. It is mostly
aimed at ensuring that the runner up assignments are consistent with
the taxonomy.
"""

import pytest

import anndata
import copy
import h5py
import json
import numpy as np
import os
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse
import tempfile

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.utils.torch_utils import (
    is_torch_available)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from cell_type_mapper.diff_exp.markers import (
    find_markers_for_all_taxonomy_pairs)

from cell_type_mapper.type_assignment.marker_cache_v2 import (
    create_marker_cache_from_reference_markers,
    serialize_markers)

from cell_type_mapper.type_assignment.election_runner import (
    run_type_assignment_on_h5ad)

from cell_type_mapper.cli.hierarchical_mapping import (
    run_mapping as ab_initio_mapping)

from cell_type_mapper.cli.from_specified_markers import (
    run_mapping as from_marker_run_mapping)

from cell_type_mapper.cli.from_specified_markers import (
    FromSpecifiedMarkersRunner)


@pytest.fixture(scope='module')
def noisy_raw_reference_h5ad_fixture(
        obs_records_fixture,
        reference_gene_names,
        tmp_dir_fixture):
    rng = np.random.default_rng(223123)
    n_cells = len(obs_records_fixture)
    n_genes = len(reference_gene_names)
    data = np.zeros(n_cells*n_genes, dtype=int)
    chosen = rng.choice(np.arange(len(data)), len(data)//3, replace=False)
    data[chosen] = rng.integers(100, 1000)
    data = data.reshape((n_cells, n_genes))

    var_data = [{'gene_name': g, 'garbage': ii}
                for ii, g in enumerate(reference_gene_names)]

    var = pd.DataFrame(var_data)
    var = var.set_index('gene_name')

    obs = pd.DataFrame(obs_records_fixture)
    obs = obs.set_index('cell_id')

    a_data = anndata.AnnData(
        X=scipy_sparse.csr_matrix(data),
        obs=obs,
        var=var,
        dtype=int)

    h5ad_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='noisy_reference_',
            suffix='.h5ad'))
    a_data.write_h5ad(h5ad_path)
    return h5ad_path

@pytest.fixture(scope='module')
def noisy_precomputed_stats_fixture(
        tmp_dir_fixture,
        taxonomy_tree_dict,
        noisy_raw_reference_h5ad_fixture):
    taxonomy_tree = TaxonomyTree(data=taxonomy_tree_dict)
    output_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='noisy_precomputed_stats_',
            suffix='.h5'))
    precompute_summary_stats_from_h5ad(
        data_path=noisy_raw_reference_h5ad_fixture,
        column_hierarchy=None,
        taxonomy_tree=taxonomy_tree,
        output_path=output_path,
        rows_at_a_time=10000,
        normalization='raw')

    return output_path


@pytest.fixture(scope='module')
def noisy_raw_query_h5ad_fixture(
        query_gene_names,
        expected_cluster_fixture,
        tmp_dir_fixture):

    rng = np.random.default_rng(77665544)

    n_cells = 5000
    n_genes = len(query_gene_names)

    data = rng.integers(100, 110, (n_cells, n_genes))

    var_data = [
        {'gene_name': g, 'garbage': ii}
         for ii, g in enumerate(query_gene_names)]

    var = pd.DataFrame(var_data)
    var = var.set_index('gene_name')

    a_data = anndata.AnnData(
        X=data,
        var=var,
        uns={'AIBS_CDM_gene_mapping': {'a': 'b', 'c': 'd'}},
        dtype=int)

    h5ad_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5ad'))
    a_data.write_h5ad(h5ad_path)
    return h5ad_path

@pytest.fixture(scope='module')
def noisy_marker_gene_lookup_fixture(
        tmp_dir_fixture,
        reference_gene_names,
        taxonomy_tree_dict):

    output_path = pathlib.Path(
        mkstemp_clean(
           dir=tmp_dir_fixture,
           prefix='marker_lookup_',
           suffix='.json'))

    rng = np.random.default_rng(77123)
    markers = dict()
    markers['None'] = list(rng.choice(reference_gene_names, 27, replace=False))
    for level in taxonomy_tree_dict['hierarchy'][:-1]:
        for node in taxonomy_tree_dict[level]:
            if len(taxonomy_tree_dict[level][node]) == 1:
                continue
            node_key = f"{level}/{node}"
            markers[node_key] = list(
                rng.choice(
                    reference_gene_names,
                    rng.integers(11, 34),
                    replace=False))

    with open(output_path, 'w') as dst:
        dst.write(json.dumps(markers))

    return output_path

@pytest.mark.parametrize(
        'flatten,use_csv,use_tmp_dir,use_gpu,just_once,drop_subclass',
        [(True, True, True, False, False, False),
         (True, False, True, False, False, False),
         (False, True, True, False, False, False),
         (False, False, True, False, False, False),
         (False, True, True, False, False, False),
         (False, True, True, True, False, False),
         (True, True, True, True, False, False),
         (True, True, True, True, True, False),
         (False, True, True, True, True, False),
         (False, True, True, False, True, False),
         (True, True, True, False, True, False),
         (False, True, True, True, True, True),
         (True, True, True, False, False, False)])
def test_mapping_from_markers(
        noisy_precomputed_stats_fixture,
        noisy_marker_gene_lookup_fixture,
        noisy_raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture,
        flatten,
        use_csv,
        use_tmp_dir,
        use_gpu,
        just_once,
        drop_subclass):
    """
    just_once sets type_assignment.bootstrap_iteration=1

    drop_subclass will drop 'subclass' from the taxonomy
    """

    if use_gpu and not is_torch_available():
        return

    env_var = 'AIBS_BKP_USE_TORCH'
    if use_gpu:
        os.environ[env_var] = 'true'
    else:
        os.environ[env_var] = 'false'

    this_tmp = tempfile.mkdtemp(dir=tmp_dir_fixture)

    if use_csv:
        csv_path = mkstemp_clean(
            dir=this_tmp,
            suffix='.csv')
    else:
        csv_path = None

    result_path = mkstemp_clean(
        dir=this_tmp,
        suffix='.json')

    config = dict()
    if use_tmp_dir:
        config['tmp_dir'] = this_tmp
    else:
        config['tmp_dir'] = None

    config['query_path'] = str(
        noisy_raw_query_h5ad_fixture.resolve().absolute())

    config['extended_result_path'] = result_path
    config['csv_result_path'] = csv_path
    config['max_gb'] = 1.0

    config['precomputed_stats'] = {
        'path': str(
            noisy_precomputed_stats_fixture.resolve().absolute())}

    config['flatten'] = flatten

    config['query_markers'] = {
        'serialized_lookup': str(
            noisy_marker_gene_lookup_fixture.resolve().absolute())}

    if drop_subclass:
        config['drop_level'] = 'subclass'

    config['type_assignment'] = {
        'bootstrap_iteration': 50,
        'bootstrap_factor': 0.75,
        'rng_seed': 1491625,
        'n_processors': 3,
        'chunk_size': 1000,
        'normalization': 'raw'
    }

    if just_once:
        config['type_assignment']['bootstrap_iteration'] = 1

    runner = FromSpecifiedMarkersRunner(
        args= [],
        input_data=config)

    runner.run()

    actual = json.load(open(result_path, 'rb'))

    # make sure taxonomy tree was recorded in metadata
    taxonomy_tree = TaxonomyTree(
        data=taxonomy_tree_dict)
    expected_tree = taxonomy_tree.to_str(drop_cells=True)
    expected_tree = json.loads(expected_tree)
    assert actual['taxonomy_tree'] == expected_tree

    gpu_msg = 'Running GPU implementation of type assignment.'
    cpu_msg = 'Running CPU implementation of type assignment.'
    found_gpu = False
    found_cpu = False
    for line in actual['log']:
        if gpu_msg in line:
            found_gpu = True
        if cpu_msg in line:
            found_cpu = True

    if found_cpu:
        assert not found_gpu
    if found_gpu:
        assert not found_cpu

    if use_gpu:
        assert found_gpu
    else:
        assert found_cpu

    query_adata = anndata.read_h5ad(
        noisy_raw_query_h5ad_fixture,
        backed='r')
    n_query_cells = query_adata.X.shape[0]

    actual = json.load(open(result_path, 'rb'))
    assert len(actual['results']) == n_query_cells

    input_uns = query_adata.uns
    assert actual['gene_identifier_mapping'] == input_uns['AIBS_CDM_gene_mapping']
    os.environ[env_var] = ''

    if flatten:
        taxonomy_tree = taxonomy_tree.flatten()

    with_runners_up = 0
    without_runners_up = 0
    is_different = 0

    # check consistency of runners up
    for cell in actual['results']:
        for level in cell:
            if level == 'cell_id':
                continue
            this_level = cell[level]
            family_tree = taxonomy_tree.parents(
                level=level,
                node=this_level['assignment'])

            n_runners_up = len(this_level['runner_up_assignments'])
            assert len(this_level['runner_up_correlation']) == n_runners_up
            if n_runners_up == 0:
                without_runners_up += 1
                np.testing.assert_allclose(
                    this_level['bootstrapping_probability'],
                    1.0,
                    atol=0.0,
                    rtol=1.0e-6)
            else:
                with_runners_up += 1
                for rup in this_level['runner_up_assignments']:
                    if rup != this_level['assignment'] and level != taxonomy_tree.leaf_level:
                        is_different += 1
                    if level == taxonomy_tree.leaf_level:
                        assert rup != this_level['assignment']
                    other_tree = taxonomy_tree.parents(
                        level=level,
                        node=rup)
                    assert other_tree == family_tree

    if not just_once:
        assert with_runners_up > 0
        if not flatten:
            assert is_different > 0

    if just_once:
        assert with_runners_up == 0

    assert without_runners_up > 0
