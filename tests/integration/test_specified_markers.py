import pytest

import anndata
import copy
import h5py
import itertools
import json
import numbers
import numpy as np
import os
import pandas as pd
import pathlib
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

from cell_type_mapper.test_utils.hierarchical_mapping import (
    run_mapping as ab_initio_mapping)

from cell_type_mapper.cli.from_specified_markers import (
    run_mapping as from_marker_run_mapping)

from cell_type_mapper.cli.from_specified_markers import (
    FromSpecifiedMarkersRunner)


@pytest.fixture(scope='module')
def ab_initio_assignment_fixture(
        raw_reference_h5ad_fixture,
        raw_query_h5ad_fixture,
        expected_cluster_fixture,
        taxonomy_tree_dict,
        query_gene_names,
        tmp_dir_fixture):
    """
    Run the (tested) assignment pipeline.
    Store its results.
    Serialize its markers to a JSON file.
    Will later test that we get back the same results
    using the specified markers.
    """

    this_tmp_dir = pathlib.Path(
        tempfile.mkdtemp(
            dir=tmp_dir_fixture))

    precompute_out = this_tmp_dir / 'precomputed.h5'
    ref_marker_out = this_tmp_dir / 'ref.h5'

    config = dict()
    config['tmp_dir'] = str(this_tmp_dir.resolve().absolute())
    config['query_path'] = str(
        raw_query_h5ad_fixture.resolve().absolute())

    config['precomputed_stats'] = {
        'reference_path': str(raw_reference_h5ad_fixture.resolve().absolute()),
        'path': str(precompute_out),
        'normalization': 'raw'}

    tree_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.json')
    with open(tree_path, 'w') as out_file:
        out_file.write(json.dumps(taxonomy_tree_dict))
    config['precomputed_stats']['taxonomy_tree'] = tree_path

    config['reference_markers'] = {
        'n_processors': 3,
        'max_gb': 0.6,
        'path': str(ref_marker_out)}

    config["query_markers"] = {
        'n_per_utility': 5,
        'n_processors': 3}

    config["type_assignment"] = {
        'n_processors': 3,
        'bootstrap_factor': 0.9,
        'bootstrap_iteration': 27,
        'rng_seed': 66234,
        'chunk_size': 1000,
        'normalization': 'raw'}

    assignment_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir_fixture,
                suffix='.json'))

    ab_initio_mapping(
        config,
        output_path=str(assignment_path),
        log_path=None)

    # create query marker cache to serialize

    query_marker_path = mkstemp_clean(
        dir=this_tmp_dir,
        suffix='.h5')

    taxonomy_tree=TaxonomyTree(data=taxonomy_tree_dict)

    create_marker_cache_from_reference_markers(
        output_cache_path=query_marker_path,
        input_cache_path=ref_marker_out,
        query_gene_names=query_gene_names,
        taxonomy_tree=taxonomy_tree,
        n_per_utility=config['query_markers']['n_per_utility'],
        n_processors=3,
        behemoth_cutoff=5000000)

    marker_lookup = serialize_markers(
        marker_cache_path=query_marker_path,
        taxonomy_tree=taxonomy_tree)

    marker_lookup_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.json'))

    # make sure that the from_specified_markers CLI
    # successfully ignores any metadata entries in the
    # marker gene lookup file
    marker_lookup['metadata'] = ['nonsense', 'garbage']

    with open(marker_lookup_path, 'w') as out_file:
        out_file.write(json.dumps(marker_lookup))

    yield {'assignment': assignment_path,
           'markers': str(marker_lookup_path.resolve().absolute()),
           'ab_initio_config': config}


@pytest.fixture(scope='module')
def precomputed_stats_fixture(
        ab_initio_assignment_fixture,
        tmp_dir_fixture):
    """
    Remove 'ge1', 'gt0', 'gt1', 'sumsq' from
    the precomputed stats file and see if we can still
    run cell type assignment.
    """
    src_path = ab_initio_assignment_fixture['ab_initio_config']['precomputed_stats']['path']
    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precomputed_stats_cleaned_',
        suffix='.h5')

    with h5py.File(src_path, 'r') as src:
        with h5py.File(dst_path, 'w') as dst:
            for k in src.keys():
                if k in ('sumsq', 'gt0', 'gt1', 'ge1'):
                    continue
                dst.create_dataset(k, data=src[k][()])
    return dst_path

@pytest.mark.parametrize(
        'flatten,use_gpu,just_once,drop_subclass',
        itertools.product(
            (True, False),
            (True, False),
            (True, False),
            (True, False)
        ))
def test_mapping_from_markers(
        ab_initio_assignment_fixture,
        raw_query_cell_x_gene_fixture,
        raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        precomputed_stats_fixture,
        tmp_dir_fixture,
        flatten,
        use_gpu,
        just_once,
        drop_subclass):
    """
    just_once sets type_assignment.bootstrap_iteration=1

    drop_subclass will drop 'subclass' from the taxonomy
    """

    use_csv = True
    use_tmp_dir = True

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

    baseline_config = ab_initio_assignment_fixture['ab_initio_config']
    config = dict()
    if use_tmp_dir:
        config['tmp_dir'] = this_tmp
    else:
        config['tmp_dir'] = None
    config['query_path'] = baseline_config['query_path']

    # just reuse the precomputed stats file that has already been generated
    config['precomputed_stats'] = {'path': precomputed_stats_fixture}

    config['type_assignment'] = copy.deepcopy(baseline_config['type_assignment'])
    if just_once:
        config['type_assignment']['bootstrap_iteration'] = 1
    config['flatten'] = flatten

    config['query_markers'] = {
        'serialized_lookup': ab_initio_assignment_fixture['markers']}

    config['extended_result_path'] = result_path
    config['csv_result_path'] = csv_path
    config['max_gb'] = 1.0

    if drop_subclass:
        config['drop_level'] = 'subclass'

    runner = FromSpecifiedMarkersRunner(
        args= [],
        input_data=config)

    runner.run()

    actual = json.load(open(result_path, 'rb'))
    assert 'RAN SUCCESSFULLY' in actual['log'][-2]

    # make sure taxonomy tree was recorded in metadata
    expected_tree = TaxonomyTree(
        data=taxonomy_tree_dict).to_str(drop_cells=True)
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

    # only expect detailed matching with this dataset if
    # flatten is False
    expected = json.load(
        open(ab_initio_assignment_fixture['assignment'], 'rb'))

    if drop_subclass and not flatten:
        for k in list(expected['marker_genes'].keys()):
            if k.startswith('subclass'):
                expected['marker_genes'].pop(k)

    if not flatten:
        assert actual['marker_genes'] == expected['marker_genes']
        actual_lookup = {
            cell['cell_id']:cell for cell in actual['results']}
        for cell in expected['results']:
            actual_cell = actual_lookup[cell['cell_id']]
            assert set(cell.keys()) == set(actual_cell.keys())
            for k in cell.keys():
                if k == 'cell_id':
                    continue

                if k == 'subclass' and drop_subclass:
                    continue

                if config['type_assignment']['bootstrap_iteration'] > 1:
                    assert cell[k]['assignment'] == actual_cell[k]['assignment']
                    for sub_k in ('bootstrapping_probability', 'avg_correlation'):
                        np.testing.assert_allclose(
                            [cell[k][sub_k]],
                            [actual_cell[k][sub_k]],
                            atol=1.0e-3,
                            rtol=1.0e-3)
    else:
        all_markers = set()
        for k in expected['marker_genes']:
            all_markers = all_markers.union(set(expected['marker_genes'][k]))

        assert set(actual['marker_genes']['None']) == all_markers
        assert len(actual['marker_genes']['None']) == len(all_markers)
        assert len(actual['marker_genes']) == 1
        valid_clusters = set(taxonomy_tree_dict['cluster'].keys())
        for cell in actual['results']:
            assert 'cluster' in cell
            assert 'assignment' in cell['cluster']
            assert 'bootstrapping_probability' in cell['cluster']
            assert 'avg_correlation' in cell['cluster']
            assert cell['cluster']['assignment'] in valid_clusters

    assert len(actual['results']) == raw_query_cell_x_gene_fixture.shape[0]
    assert len(actual['results']) == len(expected['results'])

    # check that inheritance of assignments agrees with tree
    tree_obj = TaxonomyTree(data=taxonomy_tree_dict)
    for cell in actual['results']:
        this = cell[tree_obj.leaf_level]['assignment']
        parents = tree_obj.parents(level=tree_obj.leaf_level, node=this)
        for parent_level in parents:
            assert cell[parent_level]['assignment'] == parents[parent_level]

    # check consistency between extended and csv results
    if use_csv:
        if config['type_assignment']['bootstrap_iteration'] > 1:
            stat_label = 'bootstrapping_probability'
            stat_key = 'bootstrapping_probability'
        else:
            stat_label = 'correlation_coefficient'
            stat_key = 'avg_correlation'

        result_lookup = {
            cell['cell_id']: cell for cell in actual['results']}
        with open(csv_path, 'r') as in_file:
            assert in_file.readline() == f"# metadata = {pathlib.Path(result_path).name}\n"
            hierarchy = ['class', 'subclass', 'cluster']
            assert in_file.readline() == f"# taxonomy hierarchy = {json.dumps(hierarchy)}\n"
            version_line = in_file.readline()
            if flatten:
                assert "algorithm: 'correlation'" in version_line
            else:
                assert "algorithm: 'hierarchical'" in version_line
            assert 'codebase' in version_line
            assert 'version' in version_line

            header_line = 'cell_id'
            for level in hierarchy:
                if level == 'cluster':
                    header_line += (',cluster_label,cluster_name,cluster_alias,'
                                    f'cluster_{stat_label}')
                else:
                    header_line += f',{level}_label,{level}_name,{level}_{stat_label}'
            header_line += '\n'
            assert in_file.readline() == header_line
            found_cells = []
            for line in in_file:
                params = line.strip().split(',')
                assert len(params) == 3*len(hierarchy)+2  # +2 is for cluster alias and cell_id
                this_cell = result_lookup[params[0]]
                found_cells.append(params[0])
                for i_level, level in enumerate(hierarchy):
                    assn_idx = 1+3*i_level
                    conf_idx = 3+3*i_level
                    if level == 'cluster':
                        conf_idx += 1
                    assert params[assn_idx] == this_cell[level]['assignment']
                    print('params ',params)
                    delta = np.abs(this_cell[level][stat_key]-float(params[conf_idx]))
                    assert delta < 0.0001

            assert len(found_cells) == len(result_lookup)
            assert set(found_cells) == set(result_lookup.keys())

    query_adata = anndata.read_h5ad(raw_query_h5ad_fixture, backed='r')
    input_uns = query_adata.uns
    assert actual['gene_identifier_mapping'] == input_uns['AIBS_CDM_gene_mapping']

    os.environ[env_var] = ''


@pytest.mark.parametrize('use_gpu', [False, True])
def test_mapping_from_markers_when_some_markers_missing_from_lookup(
        ab_initio_assignment_fixture,
        raw_query_cell_x_gene_fixture,
        raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        precomputed_stats_fixture,
        tmp_dir_fixture,
        use_gpu):
    """
    Test what happens when one element in the marker lookup
    contains no marker genes. Should raise an exception.
    """

    if use_gpu and not is_torch_available():
        return

    env_var = 'AIBS_BKP_USE_TORCH'
    if use_gpu:
        os.environ[env_var] = 'true'
    else:
        os.environ[env_var] = 'false'

    this_tmp = tempfile.mkdtemp(dir=tmp_dir_fixture)

    csv_path = None

    result_path = mkstemp_clean(
        dir=this_tmp,
        suffix='.json')

    baseline_config = ab_initio_assignment_fixture['ab_initio_config']
    config = dict()
    config['tmp_dir'] = this_tmp
    config['query_path'] = baseline_config['query_path']

    # just reuse the precomputed stats file that has already been generated
    config['precomputed_stats'] = {'path': precomputed_stats_fixture}

    config['type_assignment'] = copy.deepcopy(baseline_config['type_assignment'])
    config['flatten'] = False

    new_query_lookup = mkstemp_clean(
        dir=this_tmp,
        prefix='markers_',
        suffix='.json')

    with open(ab_initio_assignment_fixture['markers'], 'rb') as src:
        markers = json.load(src)

    assert len(markers['subclass/subclass_5']) > 0
    markers['subclass/subclass_5'] = []
    with open(new_query_lookup, 'w') as dst:
        dst.write(json.dumps(markers, indent=2))

    config['query_markers'] = {
        'serialized_lookup': new_query_lookup}

    config['extended_result_path'] = result_path
    config['csv_result_path'] = csv_path
    config['max_gb'] = 1.0
    config['drop_level'] = None

    runner = FromSpecifiedMarkersRunner(
        args= [],
        input_data=config)

    msg = "'subclass/subclass_5' has no valid markers in marker_lookup"
    with pytest.raises(RuntimeError, match=msg):
        runner.run()

    os.environ[env_var] = ''



@pytest.mark.parametrize('use_gpu, delete_to_none',
    itertools.product([False, True], [False, True]))
def test_mapping_from_markers_when_some_markers_missing_from_query(
        ab_initio_assignment_fixture,
        raw_query_cell_x_gene_fixture,
        raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        precomputed_stats_fixture,
        tmp_dir_fixture,
        use_gpu,
        delete_to_none):
    """
    Test what happens when the query set is missing markers
    from a parent node. Should pass, having backfilled the markers
    from that marker's parent.

    if delete_to_none is True, then remove all of the markers
    along the designated branch of the taxonomy tree, except
    the markers at the root node
    """

    if use_gpu and not is_torch_available():
        return

    env_var = 'AIBS_BKP_USE_TORCH'
    if use_gpu:
        os.environ[env_var] = 'true'
    else:
        os.environ[env_var] = 'false'

    this_tmp = tempfile.mkdtemp(dir=tmp_dir_fixture)

    csv_path = None

    result_path = mkstemp_clean(
        dir=this_tmp,
        suffix='.json')

    baseline_config = ab_initio_assignment_fixture['ab_initio_config']
    config = dict()
    config['tmp_dir'] = this_tmp

    # just reuse the precomputed stats file that has already been generated
    config['precomputed_stats'] = {'path': precomputed_stats_fixture}

    config['type_assignment'] = copy.deepcopy(baseline_config['type_assignment'])
    config['flatten'] = False

    with open(ab_initio_assignment_fixture['markers'], 'rb') as src:
        markers = json.load(src)

    assert len(markers['subclass/subclass_5']) > 0
    taxonomy_tree = TaxonomyTree(data=taxonomy_tree_dict)
    parents = taxonomy_tree.parents(
        level='subclass', node='subclass_5')

    # marker_parent is the parent node whose markers
    # we expect subclass/subclass_5 to match after we have
    # deleted the marker genes from the query file.
    if delete_to_none:
        marker_parent = 'None'
    else:
        marker_parent = f'class/{parents["class"]}'

    assert len(markers[marker_parent]) > 0

    assert set(markers['subclass/subclass_5']) != set(
        markers[marker_parent])

    # overwrite the marker genes for subclass_5 in the query set
    new_query_path = mkstemp_clean(
        dir=tmp_dir_fixture, suffix='.h5ad')
    old_query = anndata.read_h5ad(baseline_config['query_path'], backed='r')
    old_var = old_query.var.reset_index().to_dict(orient='records')
    to_overwrite = set(markers['subclass/subclass_5'])
    if delete_to_none:
        for level in parents:
            these = set(markers[f'{level}/{parents[level]}'])
            to_overwrite = to_overwrite.union(these)

    ct = 0
    for record in old_var:
        if record['gene_name'] in to_overwrite:
            record['gene_name'] = f'overwritten_{ct}'
            ct += 1

    new_var = pd.DataFrame(old_var).set_index('gene_name')
    new_a = anndata.AnnData(
        X=old_query.X[()],
        var=new_var,
        obs=old_query.obs)
    new_a.write_h5ad(new_query_path)

    config['query_path'] = new_query_path

    config['query_markers'] = {
        'serialized_lookup': ab_initio_assignment_fixture['markers']}

    config['extended_result_path'] = result_path
    config['csv_result_path'] = csv_path
    config['max_gb'] = 1.0
    config['drop_level'] = None

    runner = FromSpecifiedMarkersRunner(
        args= [],
        input_data=config)

    runner.run()

    # subclass/subclass_5 marker genes should now be identical to
    # its parent's, since we removed its default markers from the
    # query dataset
    with open(result_path, 'rb') as src:
        results = json.load(src)
    result_markers = results['marker_genes']
    assert set(result_markers[marker_parent]) == set(
        result_markers['subclass/subclass_5'])

    # if we did not delete the entire branch of the taxonomy tree,
    # we should expect that subclass/subclass_5 has different markers
    # than the root node
    if not delete_to_none:
        assert set(result_markers['subclass/subclass_5']) != set(
            result_markers['None'])

    os.environ[env_var] = ''


@pytest.mark.parametrize('use_gpu', [False, True])
def test_mapping_from_markers_when_root_markers_missing_from_query(
        ab_initio_assignment_fixture,
        raw_query_cell_x_gene_fixture,
        raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        precomputed_stats_fixture,
        tmp_dir_fixture,
        use_gpu):
    """
    Test what happens when the query set is missing all of the
    markers along a branch of the taxonomy tree (i.e. all of the markers
    up to None)
    """

    if use_gpu and not is_torch_available():
        return

    env_var = 'AIBS_BKP_USE_TORCH'
    if use_gpu:
        os.environ[env_var] = 'true'
    else:
        os.environ[env_var] = 'false'

    this_tmp = tempfile.mkdtemp(dir=tmp_dir_fixture)

    csv_path = None

    result_path = mkstemp_clean(
        dir=this_tmp,
        suffix='.json')

    baseline_config = ab_initio_assignment_fixture['ab_initio_config']
    config = dict()
    config['tmp_dir'] = this_tmp

    # just reuse the precomputed stats file that has already been generated
    config['precomputed_stats'] = {'path': precomputed_stats_fixture}

    config['type_assignment'] = copy.deepcopy(baseline_config['type_assignment'])
    config['flatten'] = False

    with open(ab_initio_assignment_fixture['markers'], 'rb') as src:
        markers = json.load(src)

    # overwrite the markers in the root node
    new_query_path = mkstemp_clean(
        dir=tmp_dir_fixture, suffix='.h5ad')
    old_query = anndata.read_h5ad(baseline_config['query_path'], backed='r')
    old_var = old_query.var.reset_index().to_dict(orient='records')
    to_overwrite = set(markers['None'])
    ct = 0
    for record in old_var:
        if record['gene_name'] in to_overwrite:
            record['gene_name'] = f'overwritten_{ct}'
            ct += 1

    new_var = pd.DataFrame(old_var).set_index('gene_name')
    new_a = anndata.AnnData(
        X=old_query.X[()],
        var=new_var,
        obs=old_query.obs)
    new_a.write_h5ad(new_query_path)

    config['query_path'] = new_query_path

    config['query_markers'] = {
        'serialized_lookup': ab_initio_assignment_fixture['markers']}

    config['extended_result_path'] = result_path
    config['csv_result_path'] = csv_path
    config['max_gb'] = 1.0
    config['drop_level'] = None

    runner = FromSpecifiedMarkersRunner(
        args= [],
        input_data=config)

    msg = "'None' has no valid markers in query gene set"

    with pytest.raises(RuntimeError, match=msg):
        runner.run()

    os.environ[env_var] = ''
