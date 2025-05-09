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
import hashlib
import itertools
import json
import numpy as np
import numbers
import os
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse
import tempfile
import warnings

from cell_type_mapper.test_utils.comparison_utils import (
    assert_blobs_equal
)

from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.utils.torch_utils import (
    is_torch_available)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad
)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from cell_type_mapper.diff_exp.precompute_utils import (
    drop_nodes_from_precomputed_stats
)

from cell_type_mapper.cli.from_specified_markers import (
    FromSpecifiedMarkersRunner)

from cell_type_mapper.utils.output_utils import (
    blob_to_hdf5,
    hdf5_to_blob)

from cell_type_mapper.test_utils.hierarchical_mapping import (
    assert_mappings_equal
)


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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

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
        tmp_dir_fixture):

    rng = np.random.default_rng(77665544)

    n_cells = 500
    n_genes = len(query_gene_names)

    data = rng.integers(100, 110, (n_cells, n_genes))

    var_data = [
        {'gene_name': g, 'garbage': ii}
        for ii, g in enumerate(query_gene_names)
    ]

    var = pd.DataFrame(var_data)
    var = var.set_index('gene_name')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

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
        'flatten,use_gpu,just_once,drop_subclass,n_runners_up,scalar_factor',
        itertools.product(
            (True, False),
            (True, False),
            (True, False),
            (True, False),
            (2, 4),
            (True, False)
        ))
def test_mapping_from_markers_smoke(
        noisy_precomputed_stats_fixture,
        noisy_marker_gene_lookup_fixture,
        noisy_raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture,
        flatten,
        use_gpu,
        just_once,
        drop_subclass,
        n_runners_up,
        scalar_factor):
    """
    just_once sets type_assignment.bootstrap_iteration=1

    drop_subclass will drop 'subclass' from the taxonomy
    """

    use_tmp_dir = True

    if use_gpu and not is_torch_available():
        return

    env_var = 'AIBS_BKP_USE_TORCH'
    if use_gpu:
        os.environ[env_var] = 'true'
    else:
        os.environ[env_var] = 'false'

    this_tmp = tempfile.mkdtemp(dir=tmp_dir_fixture)

    csv_path = mkstemp_clean(
        dir=this_tmp,
        suffix='.csv')

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

    if scalar_factor:
        bootstrap_factor = 0.75
        bootstrap_factor_lookup = None
    else:
        bootstrap_factor = None
        bootstrap_factor_lookup = [
            ('None', 0.75),
            ('class', 0.5),
            ('subclass', 0.33)
        ]

    config['type_assignment'] = {
        'bootstrap_iteration': 50,
        'bootstrap_factor': bootstrap_factor,
        'bootstrap_factor_lookup': bootstrap_factor_lookup,
        'rng_seed': 1491625,
        'n_processors': 3,
        'chunk_size': 1000,
        'normalization': 'raw',
        'n_runners_up': n_runners_up
    }

    if just_once:
        config['type_assignment']['bootstrap_iteration'] = 1

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=config)

        runner.run()

    actual = json.load(open(result_path, 'rb'))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

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
    assert (
        actual['gene_identifier_mapping']
        == input_uns['AIBS_CDM_gene_mapping']
    )
    os.environ[env_var] = ''

    with_runners_up = 0
    without_runners_up = 0

    max_runners_up = 0
    # check consistency of runners up

    backfilled_levels = set()
    if flatten:
        backfilled_levels = set(['class', 'subclass'])
    elif drop_subclass:
        backfilled_levels = set(['subclass'])

    for cell in actual['results']:

        for level in taxonomy_tree.hierarchy:

            # since we are backfilling missing levels into results,
            # this should hold
            assert level in cell

            # check that backfilled levels are correctly flagged
            if level in backfilled_levels:
                assert not cell[level]['directly_assigned']
            else:
                assert cell[level]['directly_assigned']

        # check inheritance (i.e. that assigned cell types are
        # descended from each other)
        this_leaf = cell[taxonomy_tree.leaf_level]['assignment']
        these_parents = taxonomy_tree.parents(
            level=taxonomy_tree.leaf_level,
            node=this_leaf)

        for parent_l in these_parents:
            assert cell[parent_l]['assignment'] == these_parents[parent_l]

        for level in cell:
            if level == 'cell_id':
                continue

            this_level = cell[level]
            if 'runner_up_assignment' not in this_level:
                continue

            family_tree = taxonomy_tree.parents(
                level=level,
                node=this_level['assignment'])

            # check consistency of inheritance
            for parent_level in family_tree:

                assert (
                    cell[parent_level]['assignment']
                    == family_tree[parent_level]
                )

            n_runners_up_actual = len(this_level['runner_up_assignment'])

            # make sure runners up are unique and do not include the assigned
            # taxon
            assert (
                this_level['assignment']
                not in this_level['runner_up_assignment']
            )

            assert (
                len(set(this_level['runner_up_assignment']))
                == n_runners_up_actual
            )

            if n_runners_up_actual > max_runners_up:
                max_runners_up = n_runners_up_actual
            assert n_runners_up_actual <= n_runners_up

            assert (
                len(this_level['runner_up_correlation'])
                == n_runners_up_actual
            )
            assert (
                len(this_level['runner_up_probability'])
                == n_runners_up_actual
            )
            if n_runners_up_actual == 0:
                without_runners_up += 1
                np.testing.assert_allclose(
                    this_level['bootstrapping_probability'],
                    1.0,
                    atol=0.0,
                    rtol=1.0e-6)
            else:
                with_runners_up += 1
                if n_runners_up_actual > 1:

                    # check that runners up are ordered by probability
                    for ir in range(1, n_runners_up_actual, 1):
                        r0 = this_level['runner_up_probability'][ir]
                        r1 = this_level['runner_up_probability'][ir-1]
                        assert r0 > 0.0
                        assert r0 <= r1

                # if level was not directly assigned, then the assigned
                # node/probability will have been forced by inheritance,
                # not chosen by majority vote
                if this_level['directly_assigned']:
                    assert this_level['runner_up_probability'][0] <= \
                        this_level['bootstrapping_probability']

                # check that probability sums to <= 1
                assert this_level['bootstrapping_probability'] < 1.0
                p_sum = (
                    this_level['bootstrapping_probability']
                    + sum(this_level['runner_up_probability'])
                )
                eps = 1.0e-6
                assert p_sum <= (1.0+eps)

                if 'runner_up_assignment' in this_level:
                    for rup in this_level['runner_up_assignment']:
                        assert rup != this_level['assignment']

                        if this_level['directly_assigned']:
                            # if levels were backfilled, runners up might have
                            # odd inheritance relationship to actual
                            # assignments
                            if not flatten and not drop_subclass:
                                other_tree = taxonomy_tree.parents(
                                    level=level,
                                    node=rup)

                                assert other_tree == family_tree

    if not just_once:
        assert with_runners_up > 0
        assert max_runners_up == n_runners_up

    if just_once:
        assert with_runners_up == 0

    assert without_runners_up > 0

    # check aggregate probability
    actual_agg = []
    expected_agg = []
    for cell in actual['results']:
        prob = 1.0
        prob_lookup = dict()

        # calculate aggregate probability for directly calculated
        # levels
        for level in ['class', 'subclass', 'cluster']:
            this = prob*cell[level]['bootstrapping_probability']
            prob_lookup[level] = this
            if cell[level]['directly_assigned']:
                prob = this

        for level in ['class', 'subclass', 'cluster']:
            actual_agg.append(cell[level]['aggregate_probability'])
            expected_agg.append(prob_lookup[level])

    np.testing.assert_allclose(
        actual_agg,
        expected_agg,
        atol=0.0,
        rtol=1.0e-6)

    # check that hierarchy_consistent is set correctly
    if flatten and not just_once:
        expected_consistency = []
        cell_id = []
        for cell in actual['results']:
            is_consistent = True
            for level in cell:
                if level == 'cell_id':
                    continue
                prob = cell[level]['bootstrapping_probability']
                rup = cell[level]['runner_up_probability']
                if len(rup) > 0:
                    if prob < max(rup):
                        is_consistent = False
                        break
            expected_consistency.append(is_consistent)
            cell_id.append(cell['cell_id'])
        expected_consistency = np.array(expected_consistency)
        cell_id = np.array(cell_id)
        n_consistent = expected_consistency.sum()
        assert n_consistent > 0
        assert n_consistent < len(actual['results'])

        csv_results = pd.read_csv(csv_path, comment='#')
        np.testing.assert_array_equal(
            expected_consistency,
            csv_results.hierarchy_consistent.values
        )
        np.testing.assert_array_equal(
            cell_id,
            csv_results.cell_id.values.astype(str)
        )


@pytest.mark.parametrize(
        'flatten,use_gpu,just_once,drop_subclass,'
        'clobber,use_tmp_dir',
        itertools.product(
            (True, False),
            (True, False),
            (True, False),
            (True, False),
            (True, False),
            (True, False)
        ))
def test_mapping_from_markers_to_query_h5ad(
        noisy_precomputed_stats_fixture,
        noisy_marker_gene_lookup_fixture,
        noisy_raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture,
        flatten,
        use_gpu,
        just_once,
        drop_subclass,
        use_tmp_dir,
        clobber):
    """
    Test that we correctly write output to query_path.obsm when
    specified to do so

    just_once sets type_assignment.bootstrap_iteration=1

    drop_subclass will drop 'subclass' from the taxonomy

    clobber will make sure that the query h5ad file already has
    a field in obsm where we are trying to write and will
    overwrite it
    """

    use_csv = True
    obsm_key = 'cdm_mapping'

    if use_gpu and not is_torch_available():
        return

    # copy query file to a new location so that we can
    # test the option to write the mapping to query_path.obsm
    query_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='query_file_for_obsm_',
        suffix='.h5ad')

    src = anndata.read_h5ad(noisy_raw_query_h5ad_fixture, backed='r')

    if clobber:
        obsm = {obsm_key: src.obs}
    else:
        obsm = None

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        dst = anndata.AnnData(
                X=src.X[()],
                obs=src.obs,
                var=src.var,
                obsm=obsm)

    dst.write_h5ad(query_path)

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

    config['query_path'] = query_path

    config['extended_result_path'] = result_path
    config['obsm_key'] = obsm_key
    config['obsm_clobber'] = clobber
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
        'normalization': 'raw',
        'n_runners_up': 10
    }

    if just_once:
        config['type_assignment']['bootstrap_iteration'] = 1

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=config)

        runner.run()

    json_results = json.load(open(result_path, 'rb'))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        taxonomy_tree = TaxonomyTree(
            data=json_results['taxonomy_tree'])

    a_data = anndata.read_h5ad(query_path, backed='r')
    obs = a_data.obs
    pd_results = a_data.obsm[obsm_key]
    assert list(pd_results.index.values) == list(obs.index.values)

    pd_results = pd_results.to_dict(orient='index')
    baseline_results = {cell['cell_id']: cell
                        for cell in json_results['results']}

    assert set(pd_results.keys()) == set(baseline_results.keys())
    assert len(pd_results) > 0

    # test that results recorded in JSON and results recorded
    # in obsm are identical
    ct_runners_up = 0
    for cell_id in pd_results:
        pd_cell = pd_results[cell_id]
        baseline_cell = baseline_results[cell_id]
        for level in baseline_cell:
            if level == 'cell_id':
                continue
            readable_level = taxonomy_tree.level_to_name(level)
            assn_key = f'{readable_level}_label'
            assert pd_cell[assn_key] == baseline_cell[level]['assignment']
            node_name = taxonomy_tree.label_to_name(
                level=level,
                label=baseline_cell[level]['assignment'])
            name_key = f'{readable_level}_name'
            assert pd_cell[name_key] == node_name
            if level == taxonomy_tree.leaf_level:
                alias = taxonomy_tree.label_to_name(
                    level=level,
                    label=baseline_cell[level]['assignment'],
                    name_key='alias')
                alias_key = f'{readable_level}_alias'
                assert pd_cell[alias_key] == alias

            for k in ('bootstrapping_probability',
                      'avg_correlation',
                      'directly_assigned'):
                pd_key = f'{readable_level}_{k}'
                np.testing.assert_allclose(
                    pd_cell[pd_key],
                    baseline_cell[level][k])

            for column in baseline_cell[level]:
                if 'runner_up' not in column:
                    continue
                baseline_runners = baseline_cell[level][column]
                for idx in range(len(baseline_runners)):
                    ct_runners_up += 1
                    pd_key = f'{readable_level}_{column}_{idx}'
                    if isinstance(baseline_runners[idx], numbers.Number):
                        np.testing.assert_allclose(
                            pd_cell[pd_key],
                            baseline_runners[idx])
                    else:
                        assert pd_cell[pd_key] == baseline_runners[idx]

    os.environ[env_var] = ''
    if not just_once:
        assert ct_runners_up > 0


@pytest.mark.parametrize(
    'extended_result_path,obsm_key,obsm_clobber,error_msg',
    [
     (None, None, False, 'at least one of'),
     (None, 'cdm_mapping', False, 'already has key cdm_mapping'),
     ('junk_file', 'cdm_mapping', False, 'already has key cdm_mapping'),
     (None, 'cdm_mapping', True, None),
     (None, 'hooplah', False, None),
     ('junk_file', None, False, None)
    ]
)
def test_mapping_from_markers_to_query_h5ad_config_errors(
        noisy_precomputed_stats_fixture,
        noisy_marker_gene_lookup_fixture,
        noisy_raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture,
        extended_result_path,
        obsm_key,
        obsm_clobber,
        error_msg):
    """
    Test that we get appropriate config errors when we do not
    specify a location for the extended output
    """

    # copy query file to a new location so that we can
    # test the option to write the mapping to query_path.obsm
    query_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='query_file_for_obsm_errors_',
        suffix='.h5ad')

    src = anndata.read_h5ad(noisy_raw_query_h5ad_fixture, backed='r')
    assert len(src.obsm) == 0

    obsm = {'cdm_mapping': src.obs}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        dst = anndata.AnnData(
                X=src.X[()],
                obs=src.obs,
                var=src.var,
                obsm=obsm)

    dst.write_h5ad(query_path)

    if extended_result_path is not None:
        result_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix=extended_result_path,
            suffix='.json')
    else:
        result_path = None

    config = dict()
    config['tmp_dir'] = None

    config['query_path'] = query_path

    config['extended_result_path'] = result_path
    config['obsm_key'] = obsm_key
    config['obsm_clobber'] = obsm_clobber
    config['csv_result_path'] = None
    config['max_gb'] = 1.0

    config['precomputed_stats'] = {
        'path': str(
            noisy_precomputed_stats_fixture.resolve().absolute())}

    config['flatten'] = False

    config['query_markers'] = {
        'serialized_lookup': str(
            noisy_marker_gene_lookup_fixture.resolve().absolute())}

    config['type_assignment'] = {
        'bootstrap_iteration': 50,
        'bootstrap_factor': 0.75,
        'rng_seed': 1491625,
        'n_processors': 3,
        'chunk_size': 1000,
        'normalization': 'raw',
        'n_runners_up': 10
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        if error_msg is not None:
            with pytest.raises(RuntimeError, match=error_msg):
                FromSpecifiedMarkersRunner(
                    args=[],
                    input_data=config)
        else:
            runner = FromSpecifiedMarkersRunner(
                    args=[],
                    input_data=config)
            runner.run()


@pytest.mark.parametrize(
        'flatten,just_once,drop_subclass,n_runners_up',
        itertools.product(
            (True, False),
            (True, False),
            (True, False),
            (2, 4)
        ))
def test_compression_noisy_markers(
        noisy_precomputed_stats_fixture,
        noisy_marker_gene_lookup_fixture,
        noisy_raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture,
        flatten,
        just_once,
        drop_subclass,
        n_runners_up):
    """
    just_once sets type_assignment.bootstrap_iteration=1

    drop_subclass will drop 'subclass' from the taxonomy
    """

    use_tmp_dir = True
    csv_path = None

    this_tmp = tempfile.mkdtemp(dir=tmp_dir_fixture)

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
        'normalization': 'raw',
        'n_runners_up': n_runners_up
    }

    if just_once:
        config['type_assignment']['bootstrap_iteration'] = 1

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=config)

        runner.run()

        output_blob = json.load(open(result_path, 'rb'))

        hdf5_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='blob_to_hdf5_',
            suffix='.h5')

        blob_to_hdf5(
            output_blob=output_blob,
            dst_path=hdf5_path)

        roundtrip = hdf5_to_blob(
            src_path=hdf5_path)

        assert_blobs_equal(
            blob0=roundtrip,
            blob1=output_blob)


@pytest.mark.parametrize(
        'flatten,just_once,drop_subclass,n_runners_up',
        itertools.product(
            (True, False),
            (True, False),
            (True, False),
            (2, 4)
        ))
def test_cli_compression_noisy_markers(
        noisy_precomputed_stats_fixture,
        noisy_marker_gene_lookup_fixture,
        noisy_raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture,
        flatten,
        just_once,
        drop_subclass,
        n_runners_up):
    """
    Test whether output written to HDF5 file is identical to
    output written to JSON file.

    just_once sets type_assignment.bootstrap_iteration=1

    drop_subclass will drop 'subclass' from the taxonomy
    """

    this_tmp = tempfile.mkdtemp(dir=tmp_dir_fixture)

    json_result_path = mkstemp_clean(
        dir=this_tmp,
        suffix='.json')

    hdf5_result_path = mkstemp_clean(
        dir=this_tmp,
        suffix='.h5')

    config = dict()
    config['tmp_dir'] = this_tmp

    config['query_path'] = str(
        noisy_raw_query_h5ad_fixture.resolve().absolute())

    config['extended_result_path'] = json_result_path
    config['hdf5_result_path'] = hdf5_result_path
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
        'normalization': 'raw',
        'n_runners_up': n_runners_up
    }

    if just_once:
        config['type_assignment']['bootstrap_iteration'] = 1

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=config)

        runner.run()

    output_blob = json.load(open(json_result_path, 'rb'))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        from_hdf5 = hdf5_to_blob(
            src_path=hdf5_result_path)

    assert len(output_blob['results']) > 0
    assert_blobs_equal(
        blob0=from_hdf5,
        blob1=output_blob)


@pytest.mark.parametrize(
        'flatten,just_once,drop_subclass,n_runners_up',
        itertools.product(
            (True, False),
            (True, False),
            (True, False),
            (2, 4)
        ))
def test_failure_cli_compression_of_noisy_markers(
        noisy_precomputed_stats_fixture,
        noisy_marker_gene_lookup_fixture,
        noisy_raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture,
        flatten,
        just_once,
        drop_subclass,
        n_runners_up):
    """
    Test whether output written to HDF5 file is identical to
    output written to JSON file when the job fails.

    just_once sets type_assignment.bootstrap_iteration=1

    drop_subclass will drop 'subclass' from the taxonomy
    """

    nonsense_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    rng = np.random.default_rng(22111)
    n_cells = 55
    n_genes = 14
    var = pd.DataFrame(
        [{'g': f'garbage_{ii}'} for ii in range(n_genes)]).set_index('g')
    obs = pd.DataFrame(
        [{'c': f'c_{ii}'} for ii in range(n_cells)]).set_index('c')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        a_data = anndata.AnnData(
            obs=obs,
            var=var,
            X=rng.random((n_cells, n_genes)))

    a_data.write_h5ad(nonsense_path)

    this_tmp = tempfile.mkdtemp(dir=tmp_dir_fixture)

    json_result_path = mkstemp_clean(
        dir=this_tmp,
        suffix='.json')

    hdf5_result_path = mkstemp_clean(
        dir=this_tmp,
        suffix='.h5')

    config = dict()
    config['tmp_dir'] = this_tmp

    config['query_path'] = nonsense_path

    config['extended_result_path'] = json_result_path
    config['hdf5_result_path'] = hdf5_result_path
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
        'normalization': 'raw',
        'n_runners_up': n_runners_up
    }

    if just_once:
        config['type_assignment']['bootstrap_iteration'] = 1

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=config)

        try:
            runner.run()
        except Exception:
            pass

    output_blob = json.load(open(json_result_path, 'rb'))
    assert 'results' not in output_blob

    from_hdf5 = hdf5_to_blob(
        src_path=hdf5_result_path)

    assert from_hdf5 == output_blob


@pytest.mark.parametrize(
        'flatten,just_once,drop_subclass,n_runners_up',
        itertools.product(
            (True, False),
            (True, False),
            (True, False),
            (2, 4)
        ))
def test_hdf5_output_from_cli(
        noisy_precomputed_stats_fixture,
        noisy_marker_gene_lookup_fixture,
        noisy_raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture,
        flatten,
        just_once,
        drop_subclass,
        n_runners_up):
    """
    Test that from_specified_markers CLI can successfully
    write an HDF5 output file with expected output

    just_once sets type_assignment.bootstrap_iteration=1

    drop_subclass will drop 'subclass' from the taxonomy
    """

    use_tmp_dir = True
    csv_path = None

    this_tmp = tempfile.mkdtemp(dir=tmp_dir_fixture)

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
    config['cloud_safe'] = False

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
        'normalization': 'raw',
        'n_runners_up': n_runners_up
    }

    if just_once:
        config['type_assignment']['bootstrap_iteration'] = 1

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=config)

        runner.run()

    json_output = json.load(open(result_path, 'rb'))

    hdf5_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='hdf5_output_',
        suffix='.h5')

    config.pop('extended_result_path')
    config['hdf5_result_path'] = hdf5_path

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=config)

        runner.run()

        hdf5_output = hdf5_to_blob(
            src_path=hdf5_path)

    # remove fields that will be different because
    # of the different configs/runtimes
    hdf5_config = hdf5_output.pop('config')
    assert hdf5_config == config

    json_output.pop('config')

    hdf5_output.pop('metadata')
    json_output.pop('metadata')

    hdf5_output.pop('log')
    json_output.pop('log')

    assert_blobs_equal(
        blob0=hdf5_output,
        blob1=json_output)


@pytest.mark.parametrize(
        'use_gpu,drop_nodes',
        itertools.product(
            (True, False),
            ([('class', 'a'), ('subclass', 'subclass_5')],
             [('class', 'a'), ('class', 'b')])
        )
)
def test_mapping_from_markers_with_drop_nodes(
        noisy_precomputed_stats_fixture,
        noisy_marker_gene_lookup_fixture,
        noisy_raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture,
        use_gpu,
        drop_nodes):
    """
    Test that the FromSpecifiedMarkersRunner correctly drops
    nodes from the taxonomy
    """

    hasher = hashlib.md5()
    with open(noisy_precomputed_stats_fixture, 'rb') as src:
        hasher.update(src.read())
    precompute_hash = hasher.hexdigest()

    use_tmp_dir = True

    if use_gpu and not is_torch_available():
        return

    env_var = 'AIBS_BKP_USE_TORCH'
    if use_gpu:
        os.environ[env_var] = 'true'
    else:
        os.environ[env_var] = 'false'

    this_tmp = tempfile.mkdtemp(dir=tmp_dir_fixture)

    config = dict()
    if use_tmp_dir:
        config['tmp_dir'] = this_tmp
    else:
        config['tmp_dir'] = None

    config['query_path'] = str(
        noisy_raw_query_h5ad_fixture.resolve().absolute())

    config['csv_result_path'] = None
    config['max_gb'] = 1.0

    config['precomputed_stats'] = {
        'path': str(
            noisy_precomputed_stats_fixture.resolve().absolute())}

    config['flatten'] = False

    config['query_markers'] = {
        'serialized_lookup': str(
            noisy_marker_gene_lookup_fixture.resolve().absolute())}

    bootstrap_factor = 0.5

    config['type_assignment'] = {
        'bootstrap_iteration': 50,
        'bootstrap_factor': bootstrap_factor,
        'bootstrap_factor_lookup': None,
        'rng_seed': 1491625,
        'n_processors': 3,
        'chunk_size': 1000,
        'normalization': 'raw',
        'n_runners_up': 5
    }

    baseline_config = copy.deepcopy(config)

    drop_nodes_config = copy.deepcopy(config)
    drop_nodes_config['nodes_to_drop'] = drop_nodes

    trimmed_precompute_path = mkstemp_clean(
        dir=this_tmp,
        prefix='trimmed_precompute_',
        suffix='.h5'
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        drop_nodes_from_precomputed_stats(
            src_path=noisy_precomputed_stats_fixture,
            dst_path=trimmed_precompute_path,
            node_list=drop_nodes,
            clobber=True
        )

    alt_baseline_config = copy.deepcopy(config)
    alt_baseline_config['precomputed_stats']['path'] = trimmed_precompute_path

    baseline_output = mkstemp_clean(
        dir=this_tmp,
        prefix='baseline_',
        suffix='.json'
    )
    baseline_config['extended_result_path'] = baseline_output

    alt_output = mkstemp_clean(
        dir=this_tmp,
        prefix='pre_trimmed_',
        suffix='.json'
    )
    alt_baseline_config['extended_result_path'] = alt_output

    drop_output = mkstemp_clean(
        dir=this_tmp,
        prefix='drop_nodes_',
        suffix='.json'
    )
    drop_nodes_config['extended_result_path'] = drop_output

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=baseline_config)
        runner.run()

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=alt_baseline_config)
        runner.run()

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=drop_nodes_config)
        runner.run()

    baseline = json.load(open(baseline_output, 'rb'))
    alt = json.load(open(alt_output, 'rb'))
    drop = json.load(open(drop_output, 'rb'))

    assert baseline['results'] != alt['results']
    assert alt['results'] == drop['results']
    assert baseline['taxonomy_tree'] != alt['taxonomy_tree']
    assert alt['taxonomy_tree'] == drop['taxonomy_tree']

    # make sure precomputed stats file did not change
    hasher = hashlib.md5()
    with open(noisy_precomputed_stats_fixture, 'rb') as src:
        hasher.update(src.read())
    assert hasher.hexdigest() == precompute_hash
    os.environ[env_var] = ''


@pytest.mark.parametrize(
        'use_gpu,drop_nodes,flatten',
        itertools.product(
            (True, False),
            ([('class', 'a'), ('subclass', 'subclass_5')],
             [('class', 'a'), ('class', 'b')]),
            (True, False)
        )
)
def test_mapping_config_roundtrip(
        noisy_precomputed_stats_fixture,
        noisy_marker_gene_lookup_fixture,
        noisy_raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture,
        use_gpu,
        drop_nodes,
        flatten):
    """
    Test that the FromSpecifiedMarkersRunner correctly records
    the config parameters in the output JSON file
    """

    hasher = hashlib.md5()
    with open(noisy_precomputed_stats_fixture, 'rb') as src:
        hasher.update(src.read())

    use_tmp_dir = True

    if use_gpu and not is_torch_available():
        return

    env_var = 'AIBS_BKP_USE_TORCH'
    if use_gpu:
        os.environ[env_var] = 'true'
    else:
        os.environ[env_var] = 'false'

    this_tmp = tempfile.mkdtemp(dir=tmp_dir_fixture)

    config = dict()
    if use_tmp_dir:
        config['tmp_dir'] = this_tmp
    else:
        config['tmp_dir'] = None

    config['query_path'] = str(
        noisy_raw_query_h5ad_fixture.resolve().absolute())

    config['csv_result_path'] = None
    config['max_gb'] = 1.0

    config['precomputed_stats'] = {
        'path': str(
            noisy_precomputed_stats_fixture.resolve().absolute())}

    config['flatten'] = flatten
    config['cloud_safe'] = False

    config['query_markers'] = {
        'serialized_lookup': str(
            noisy_marker_gene_lookup_fixture.resolve().absolute())}

    config['type_assignment'] = {
        'bootstrap_iteration': 50,
        'bootstrap_factor': 0.5,
        'bootstrap_factor_lookup': None,
        'rng_seed': 1491625,
        'n_processors': 3,
        'chunk_size': 1000,
        'normalization': 'raw',
        'n_runners_up': 5
    }
    config['nodes_to_drop'] = drop_nodes

    baseline_output = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.json'
    )
    baseline_config = copy.deepcopy(config)
    baseline_config['extended_result_path'] = baseline_output

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=baseline_config
        )
        runner.run()

    baseline_mapping = json.load(open(baseline_output, 'rb'))

    test_config = copy.deepcopy(baseline_mapping['config'])
    test_output = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.json'
    )
    test_config['extended_result_path'] = test_output

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=test_config
        )
        runner.run()

    test_mapping = json.load(open(test_output, 'rb'))

    assert test_mapping['results'] == baseline_mapping['results']
    assert test_mapping['taxonomy_tree'] == baseline_mapping['taxonomy_tree']
    assert test_mapping['marker_genes'] == baseline_mapping['marker_genes']
    assert test_mapping['metadata'] != baseline_mapping['metadata']

    update_config_list = [
        {'rng_seed': 6677112},
        {'n_processors': 2},
        {'bootstrap_factor': 0.8},
        {'bootstrap_iteration': 34}
    ]

    # the GPU implementation does not actually
    # care about the value of n_processors
    if use_gpu:
        update_config_list.pop(1)

    for update_config in update_config_list:
        test_config = copy.deepcopy(config)
        for k in update_config:
            assert k in test_config['type_assignment']
            test_config['type_assignment'][k] = update_config[k]
        test_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.json'
        )
        test_config['extended_result_path'] = test_path
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            runner = FromSpecifiedMarkersRunner(
                args=[],
                input_data=test_config
            )
            runner.run()
        test_mapping = json.load(open(test_path, 'rb'))
        assert test_mapping['results'] != baseline_mapping['results']

        assert test_mapping['taxonomy_tree'] == \
            baseline_mapping['taxonomy_tree']

        assert test_mapping['marker_genes'] == baseline_mapping['marker_genes']
        assert test_mapping['metadata'] != baseline_mapping['metadata']

        retest_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.json'
        )
        retest_config = copy.deepcopy(test_mapping['config'])
        retest_config['extended_result_path'] = retest_path
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            runner = FromSpecifiedMarkersRunner(
                args=[],
                input_data=retest_config
            )
            runner.run()
        retest_mapping = json.load(open(retest_path, 'rb'))
        assert retest_mapping['results'] == test_mapping['results']
        assert retest_mapping['taxonomy_tree'] == test_mapping['taxonomy_tree']
        assert retest_mapping['marker_genes'] == test_mapping['marker_genes']
        assert retest_mapping['metadata'] != test_mapping['metadata']

    os.environ[env_var] = ''


@pytest.mark.parametrize(
    "specify_markers,collapse_markers",
    [(True, True),
     (False, True),
     (False, False)]
)
def test_marker_collapse_params(
        noisy_precomputed_stats_fixture,
        noisy_marker_gene_lookup_fixture,
        noisy_raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture,
        specify_markers,
        collapse_markers):
    """
    Test that correct mapping is done when collapse_markers
    is specified
    """

    this_tmp = tempfile.mkdtemp(dir=tmp_dir_fixture)

    csv_path = mkstemp_clean(
        dir=this_tmp,
        suffix='.csv')

    result_path = mkstemp_clean(
        dir=this_tmp,
        suffix='.json')

    config = dict()
    config['tmp_dir'] = this_tmp

    config['query_path'] = str(
        noisy_raw_query_h5ad_fixture.resolve().absolute())

    config['extended_result_path'] = result_path
    config['csv_result_path'] = csv_path
    config['max_gb'] = 1.0

    config['precomputed_stats'] = {
        'path': str(
            noisy_precomputed_stats_fixture.resolve().absolute())}

    config['flatten'] = False

    if specify_markers:
        marker_path = str(noisy_marker_gene_lookup_fixture)
    else:
        marker_path = None

    config['query_markers'] = {
        'serialized_lookup': marker_path,
        'collapse_markers': collapse_markers
    }

    config['type_assignment'] = {
        'bootstrap_iteration': 50,
        'bootstrap_factor': 0.5,
        'bootstrap_factor_lookup': None,
        'rng_seed': 1491625,
        'n_processors': 3,
        'chunk_size': 1000,
        'normalization': 'raw',
        'n_runners_up': 5
    }

    msg = (
        "Cannot have collapse_markers = False if you are not specifying "
        "a serialized_lookup for query_markers"
    )

    if (not specify_markers) and (not collapse_markers):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            with pytest.raises(RuntimeError, match=msg):
                runner = FromSpecifiedMarkersRunner(
                    args=[],
                    input_data=config)

                runner.run()
    else:

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            runner = FromSpecifiedMarkersRunner(
                args=[],
                input_data=config)

            runner.run()

        # Now run the mapping, specifying the collapsed markers
        # in a JSON file and confirm that the mappings come out
        # identical

        expected_config = copy.deepcopy(config)

        expected_json = mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='expected_output_',
            suffix='.json'
        )
        expected_csv = mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='expected_output_',
            suffix='.csv'
        )
        expected_config['extended_result_path'] = expected_json
        expected_config['csv_result_path'] = expected_csv

        taxonomy_tree = TaxonomyTree.from_precomputed_stats(
            noisy_precomputed_stats_fixture
        )

        expected_markers = dict()
        var = read_df_from_h5ad(config['query_path'], df_name='var')
        query_genes = set(var.index.values)
        if specify_markers:
            _src_path = config['query_markers']['serialized_lookup']
            with open(_src_path, 'rb') as src:
                src_markers = json.load(src)
            all_markers = set()
            for key in src_markers:
                if key in ('log', 'metadata'):
                    continue
                all_markers = all_markers.union(set(src_markers[key]))
            all_markers = sorted(all_markers)
        else:
            with h5py.File(config['precomputed_stats']['path'], 'r') as src:
                ref_genes = json.loads(
                    src['col_names'][()].decode('utf-8')
                )

            all_markers = sorted(
                query_genes.intersection(set(ref_genes))
            )

        for parent in taxonomy_tree.all_parents:
            if parent is None:
                key = 'None'
            else:
                key = f'{parent[0]}/{parent[1]}'
            expected_markers[key] = all_markers

        expected_marker_path = mkstemp_clean(
           dir=tmp_dir_fixture,
           prefix='expected_markers_',
           suffix='.json'
        )

        with open(expected_marker_path, 'w') as dst:
            dst.write(json.dumps(expected_markers, indent=2))

        expected_config['query_markers']['serialized_lookup'] = (
            expected_marker_path
        )
        expected_config['query_markers']['collapse_markers'] = False

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            runner = FromSpecifiedMarkersRunner(
                args=[],
                input_data=expected_config)

            runner.run()

        actual = json.load(
            open(config['extended_result_path'], 'rb')
        )

        if not specify_markers:
            # check for warning about genes that were in query
            # dataset but were not in reference dataset
            found_it = False
            match = (
                "genes in the query dataset were not present in the "
                "reference dataset."
            )
            for msg in actual['log']:
                if match in msg:
                    found_it = True
                    break
                assert not found_it

        expected = json.load(
            open(expected_config['extended_result_path'], 'rb')
        )
        assert actual != expected
        assert_mappings_equal(
            mapping0=actual['results'],
            mapping1=expected['results']
        )

        # check that the recorded marker genes in actual are
        # as expected
        actual_markers = actual['marker_genes']
        assert set(actual_markers.keys()) == set(expected_markers.keys())
        for k in actual_markers:
            act = set(actual_markers[k])
            expct = set(expected_markers[k])
            assert len(act-expct) == 0

            # make sure any difference is due to genes that are
            # not in the query set
            diff = (expct-act)
            assert len(query_genes.intersection(diff)) == 0


def test_marker_collapse_params_no_overlap(
        noisy_precomputed_stats_fixture,
        noisy_marker_gene_lookup_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture):
    """
    Test that the correct error is raised when you
    specify collapse_markers but there is no overlap
    between reference genes and your query set
    """

    this_tmp = tempfile.mkdtemp(dir=tmp_dir_fixture)

    query_path = mkstemp_clean(
        dir=this_tmp,
        prefix='query_',
        suffix='.h5ad'
    )

    var = pd.DataFrame(
        [{'gene_id': f'special_case_{ii}'}
         for ii in range(10)]
    ).set_index('gene_id')

    query_data = anndata.AnnData(
        var=var,
        X=np.random.random_sample((15, 10))
    )
    query_data.write_h5ad(query_path)

    csv_path = mkstemp_clean(
        dir=this_tmp,
        suffix='.csv')

    result_path = mkstemp_clean(
        dir=this_tmp,
        suffix='.json')

    config = dict()
    config['tmp_dir'] = this_tmp

    config['query_path'] = query_path

    config['extended_result_path'] = result_path
    config['csv_result_path'] = csv_path
    config['max_gb'] = 1.0

    config['precomputed_stats'] = {
        'path': str(
            noisy_precomputed_stats_fixture.resolve().absolute())}

    config['flatten'] = False

    config['query_markers'] = {
        'serialized_lookup': None,
        'collapse_markers': True
    }

    config['type_assignment'] = {
        'bootstrap_iteration': 50,
        'bootstrap_factor': 0.5,
        'bootstrap_factor_lookup': None,
        'rng_seed': 1491625,
        'n_processors': 3,
        'chunk_size': 1000,
        'normalization': 'raw',
        'n_runners_up': 5
    }

    msg = (
        "There was no overlap between the genes in the query "
        "dataset and the genes in the reference dataset"
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        with pytest.raises(RuntimeError, match=msg):
            runner = FromSpecifiedMarkersRunner(
                args=[],
                input_data=config)

            runner.run()
