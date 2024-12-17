import pytest

import anndata
import copy
import h5py
import itertools
import json
import numpy as np
import os
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse
import tempfile
import warnings

from cell_type_mapper.test_utils.cloud_safe import (
    check_not_file)

from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.test_utils.anndata_utils import (
    create_h5ad_without_encoding_type
)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad
)

from cell_type_mapper.utils.output_utils import (
    blob_to_hdf5,
    hdf5_to_blob)

from cell_type_mapper.utils.torch_utils import (
    is_torch_available)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.type_assignment.marker_cache_v2 import (
    create_marker_cache_from_reference_markers,
    serialize_markers)

from cell_type_mapper.test_utils.hierarchical_mapping import (
    run_mapping as ab_initio_mapping)

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
        'normalization': 'raw',
        'min_markers': 10}

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

    taxonomy_tree = TaxonomyTree(data=taxonomy_tree_dict)

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
    src_path = ab_initio_assignment_fixture[
        'ab_initio_config']['precomputed_stats']['path']

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
        'flatten,use_gpu,just_once,drop_subclass,keep_encoding',
        itertools.product(
            (True, False),
            (True, False),
            (True, False),
            (True, False),
            (True, False)
        ))
def test_mapping_from_markers_basic(
        ab_initio_assignment_fixture,
        raw_query_cell_x_gene_fixture,
        raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        precomputed_stats_fixture,
        tmp_dir_fixture,
        flatten,
        use_gpu,
        just_once,
        drop_subclass,
        keep_encoding):
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

    if keep_encoding:
        test_query_path = baseline_config['query_path']
    else:
        test_query_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='no_encoding_',
            suffix='.h5ad'
        )
        create_h5ad_without_encoding_type(
            src_path=baseline_config['query_path'],
            dst_path=test_query_path
        )

    config = dict()
    if use_tmp_dir:
        config['tmp_dir'] = this_tmp
    else:
        config['tmp_dir'] = None
    config['query_path'] = test_query_path
    # just reuse the precomputed stats file that has already been generated
    config['precomputed_stats'] = {'path': precomputed_stats_fixture}

    config['type_assignment'] = copy.deepcopy(
        baseline_config['type_assignment'])

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

    log_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.txt')
    config['log_path'] = log_path

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=config)

        runner.run()

        # make sure taxonomy tree was recorded in metadata
        expected_tree = TaxonomyTree(
            data=taxonomy_tree_dict).to_str(drop_cells=True)

    actual = json.load(open(result_path, 'rb'))
    assert 'RAN SUCCESSFULLY' in actual['log'][-2]

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
            cell['cell_id']: cell
            for cell in actual['results']
        }
        for cell in expected['results']:
            actual_cell = actual_lookup[cell['cell_id']]
            assert set(cell.keys()) == set(actual_cell.keys())
            for k in cell.keys():
                if k == 'cell_id':
                    continue

                if k == 'subclass' and drop_subclass:
                    continue

                if config['type_assignment']['bootstrap_iteration'] > 1:
                    assert cell[k][
                        'assignment'] == actual_cell[k]['assignment']

                    for sub_k in ('bootstrapping_probability',
                                  'avg_correlation'):

                        np.testing.assert_allclose(
                            [cell[k][sub_k]],
                            [actual_cell[k][sub_k]],
                            atol=1.0e-3,
                            rtol=1.0e-3)
    else:
        all_markers = set()
        for k in expected['marker_genes']:
            if k not in ('metadata', 'log'):
                all_markers = all_markers.union(
                    set(expected['marker_genes'][k])
                )

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
            assert in_file.readline() == (
                f"# metadata = {pathlib.Path(result_path).name}\n"
            )
            hierarchy = ['class', 'subclass', 'cluster']
            assert in_file.readline() == (
                f"# taxonomy hierarchy = {json.dumps(hierarchy)}\n"
            )
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
                    header_line += (
                        ',cluster_label,cluster_name,cluster_alias,'
                        f'cluster_{stat_label}'
                    )
                else:
                    header_line += (
                        f',{level}_label,{level}_name,{level}_{stat_label}'
                    )
            header_line += '\n'
            assert in_file.readline() == header_line
            found_cells = []
            for line in in_file:
                params = line.strip().split(',')

                # +2 is for cluster alias and cell_id
                assert len(params) == 3*len(hierarchy)+2

                this_cell = result_lookup[params[0]]
                found_cells.append(params[0])
                for i_level, level in enumerate(hierarchy):
                    assn_idx = 1+3*i_level
                    conf_idx = 3+3*i_level
                    if level == 'cluster':
                        conf_idx += 1
                    assert params[assn_idx] == this_cell[level]['assignment']
                    delta = np.abs(
                        this_cell[level][stat_key]-float(params[conf_idx])
                    )
                    assert delta < 0.0001

            assert len(found_cells) == len(result_lookup)
            assert set(found_cells) == set(result_lookup.keys())

    query_adata = anndata.read_h5ad(raw_query_h5ad_fixture, backed='r')
    input_uns = query_adata.uns

    assert actual['gene_identifier_mapping'] == input_uns[
        'AIBS_CDM_gene_mapping']

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
    contains no marker genes. Should emit an error.
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

    config['type_assignment'] = copy.deepcopy(
        baseline_config['type_assignment'])
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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=config)

        msg = "'subclass/subclass_5' has no valid markers in marker_lookup"
        with pytest.warns(UserWarning, match=msg):
            runner.run()

    os.environ[env_var] = ''


@pytest.mark.parametrize(
    'use_gpu, delete_to_none',
    itertools.product([False, True], [False, True])
)
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

    config['type_assignment'] = copy.deepcopy(
        baseline_config['type_assignment'])
    config['flatten'] = False

    with open(ab_initio_assignment_fixture['markers'], 'rb') as src:
        markers = json.load(src)

    assert len(markers['subclass/subclass_5']) > 0

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
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

    config['type_assignment'] = copy.deepcopy(
        baseline_config['type_assignment'])
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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=config)

        msg = "'None' has no valid markers in query gene set"

        with pytest.raises(RuntimeError, match=msg):
            runner.run()

    os.environ[env_var] = ''


@pytest.mark.parametrize(
        'flatten,use_gpu',
        itertools.product(
            (True, False),
            (True, False)
        ))
def test_mapping_when_there_are_no_markers(
        ab_initio_assignment_fixture,
        raw_query_cell_x_gene_fixture,
        raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        precomputed_stats_fixture,
        tmp_dir_fixture,
        flatten,
        use_gpu):
    """
    Test case when there are no markers
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

    config['type_assignment'] = copy.deepcopy(
        baseline_config['type_assignment'])
    config['flatten'] = flatten

    # overwrite markers with garbage
    with open(ab_initio_assignment_fixture['markers'], 'rb') as src:
        original_markers = json.load(src)
    for k in original_markers:
        original_markers[k] = ['garbage1', 'garbage2', 'garbage3']

    bad_marker_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.json')
    with open(bad_marker_path, 'w') as dst:
        dst.write(json.dumps(original_markers))

    config['query_markers'] = {
        'serialized_lookup': bad_marker_path}

    config['extended_result_path'] = result_path
    config['csv_result_path'] = csv_path
    config['max_gb'] = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        msg = ("After comparing query data to reference data, "
               "no valid marker genes could be found")

        with pytest.raises(RuntimeError, match=msg):

            runner = FromSpecifiedMarkersRunner(
                args=[],
                input_data=config)

            runner.run()

    os.environ[env_var] = ''


@pytest.mark.parametrize(
        'flatten,use_gpu,query_dtype,density',
        itertools.product(
            (True, False),
            (True, False),
            (np.uint16, np.uint32),
            ('csr', 'csc', 'array')
        ))
def test_mapping_uint16_data(
        ab_initio_assignment_fixture,
        raw_query_cell_x_gene_fixture,
        raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        precomputed_stats_fixture,
        tmp_dir_fixture,
        flatten,
        use_gpu,
        query_dtype,
        density):
    """
    Test mapping data that is saved as a uint16 (torch cannot convert
    any uint other than uint8 to a tensor)
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

    # recast query data using specified dtype
    src = anndata.read_h5ad(baseline_config['query_path'], backed='r')
    obs = src.obs[:10]
    var = src.var

    rng = np.random.default_rng(556611)
    iinfo = np.iinfo(query_dtype)
    if density == 'array':
        baseline_query_data = rng.integers(
            0, iinfo.max, (len(obs), len(var)), dtype=np.int64)
        query_data = baseline_query_data.astype(query_dtype)
    else:
        ntot = len(obs)*len(var)
        baseline_query_data = np.zeros(ntot, dtype=np.int64)
        chosen = rng.choice(np.arange(ntot), ntot//3, replace=False)
        baseline_query_data[chosen] = rng.integers(0, iinfo.max, len(chosen))
        baseline_query_data = baseline_query_data.reshape(len(obs), len(var))

        query_data = baseline_query_data.astype(query_dtype)

        if density == 'csc':
            query_data = scipy_sparse.csc_matrix(query_data)
        else:
            query_data = scipy_sparse.csr_matrix(query_data)

    dst = anndata.AnnData(X=query_data, obs=obs, var=var)
    query_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')
    dst.write_h5ad(query_path)

    config['query_path'] = query_path

    # just reuse the precomputed stats file that has already been generated
    config['precomputed_stats'] = {'path': precomputed_stats_fixture}

    config['type_assignment'] = copy.deepcopy(
        baseline_config['type_assignment'])
    config['flatten'] = flatten

    config['query_markers'] = {
        'serialized_lookup': ab_initio_assignment_fixture['markers']}

    config['extended_result_path'] = result_path
    config['csv_result_path'] = csv_path
    config['max_gb'] = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=config)

        runner.run()

    actual = json.load(open(result_path, 'rb'))
    assert 'RAN SUCCESSFULLY' in actual['log'][-2]

    # now save the data as an np.int64 and verify
    # that the results are exactly the same
    baseline_dst = anndata.AnnData(X=baseline_query_data, var=var, obs=obs)
    baseline_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')
    baseline_dst.write_h5ad(baseline_path)
    config['query_path'] = baseline_path
    baseline_output = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.json')
    config['extended_result_path'] = baseline_output

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=config)
        runner.run()

    expected = json.load(open(baseline_output, 'rb'))
    actual_results = {
        cell['cell_id']: cell for cell in actual['results']}
    expected_results = {
        cell['cell_id']: cell for cell in expected['results']}

    assert expected_results == actual_results

    # make sure that the data we ran was saved as an int64
    with h5py.File(config['query_path'], 'r') as src:
        assert src['X'].dtype == np.int64

    os.environ[env_var] = ''


@pytest.mark.parametrize(
        'flatten,use_gpu,just_once,drop_subclass',
        itertools.product(
            (True, False),
            (True, False),
            (True, False),
            (True, False)
        ))
def test_cloud_safe_mapping(
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
    Test that when cloud_safe is True, no absolute file paths
    get recorded in the output log
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

    config['type_assignment'] = copy.deepcopy(
        baseline_config['type_assignment'])

    if just_once:
        config['type_assignment']['bootstrap_iteration'] = 1

    config['flatten'] = flatten

    config['query_markers'] = {
        'serialized_lookup': ab_initio_assignment_fixture['markers']}

    config['extended_result_path'] = result_path
    config['csv_result_path'] = csv_path
    config['max_gb'] = 1.0
    config['cloud_safe'] = True

    if drop_subclass:
        config['drop_level'] = 'subclass'

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=config)

        runner.run()

    with open(result_path, 'rb') as src:
        data = json.load(src)
        check_not_file(data['log'])
        check_not_file(data['config'])


@pytest.mark.parametrize(
        'flatten,just_once,drop_subclass',
        itertools.product(
            (True, False),
            (True, False),
            (True, False)
        ))
def test_output_compression(
        ab_initio_assignment_fixture,
        raw_query_cell_x_gene_fixture,
        raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        precomputed_stats_fixture,
        tmp_dir_fixture,
        flatten,
        just_once,
        drop_subclass):
    """
    Use the fact that this test file already has a full mapping
    popeline implemented in it to test the
    blob_to_hdf5 < - > hdf5_to_blob
    output_utils roundtrip
    """

    use_tmp_dir = True
    this_tmp = tempfile.mkdtemp(dir=tmp_dir_fixture)
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

    config['type_assignment'] = copy.deepcopy(
        baseline_config['type_assignment'])

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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=config)

        runner.run()

        with open(result_path, 'rb') as src:
            output_blob = json.load(src)

        hdf5_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='blob_to_hdf5_',
            suffix='.h5')

        blob_to_hdf5(
            output_blob=output_blob,
            dst_path=hdf5_path)

        roundtrip = hdf5_to_blob(
            src_path=hdf5_path)

    assert roundtrip == output_blob


def test_integer_indexed_input(
        ab_initio_assignment_fixture,
        raw_query_cell_x_gene_fixture,
        raw_query_h5ad_fixture,
        taxonomy_tree_dict,
        precomputed_stats_fixture,
        tmp_dir_fixture):
    """
    Test that the mapper can handle an input h5ad file in
    which obs is indexed using numpy.int64 (which JSON cannot
    serialize)
    """

    flatten = False
    just_once = False
    drop_subclass = False

    use_tmp_dir = True
    this_tmp = tempfile.mkdtemp(dir=tmp_dir_fixture)
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

    new_query_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='int_indexed_',
        suffix='.h5ad')

    query_data = anndata.read_h5ad(baseline_config['query_path'])
    old_obs = query_data.obs
    old_idx = old_obs.index.values
    new_obs = []
    idx_array = np.arange(len(old_obs), dtype=np.int64)
    rng = np.random.default_rng(22113)
    rng.shuffle(idx_array)
    old_label_to_new_label = dict()

    for ii in range(len(old_obs)):
        new_label = idx_array[ii]
        old_label_to_new_label[old_idx[ii]] = new_label
        new_obs.append({'cell_label': new_label})

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        new_obs = pd.DataFrame(new_obs).set_index('cell_label')
        new_obs.index.astype(np.int64, copy=False)
        new_data = anndata.AnnData(
            obs=new_obs,
            var=query_data.var,
            X=query_data.X)
        new_data.write_h5ad(new_query_path)

        # doctor h5ad file to force the index to have integer values
        # (I'm not sure that anndata thinks this should be possible,
        # https://github.com/scverse/anndata/issues/35
        # but it has been encountered "in the wild")
        with h5py.File(new_query_path, 'a') as src:
            old_index = src['obs']['cell_label'][()]
            del src['obs']['cell_label']
            src['obs'].create_dataset(
                'cell_label',
                data=old_index.astype(np.int64)
            )

        checking = read_df_from_h5ad(new_query_path, df_name='obs')
        assert checking.index.dtype == np.int64

    config['query_path'] = str(new_query_path)

    # just reuse the precomputed stats file that has already been generated
    config['precomputed_stats'] = {'path': precomputed_stats_fixture}

    config['type_assignment'] = copy.deepcopy(
        baseline_config['type_assignment'])

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

    control_config = copy.deepcopy(config)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=config)
        runner.run()

    # re-run with the original anndata file (that doesn't use
    # integers to index obs) and make sure that results are identical
    control_output = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='control_mapping_',
        suffix='.json'
    )
    control_config['query_path'] = baseline_config['query_path']
    control_config['extended_result_path'] = control_output

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=control_config
        )
        runner.run()

    with open(config['extended_result_path'], 'rb') as src:
        actual = json.load(src)
    with open(control_config['extended_result_path'], 'rb') as src:
        expected = json.load(src)

    assert len(expected['results']) == len(actual['results'])

    actual = {
        c['cell_id']: c for c in actual['results']
    }
    for ii in range(len(expected['results'])):
        expected_cell = expected['results'][ii]
        actual_cell = actual[old_label_to_new_label[expected_cell['cell_id']]]
        expected_cell.pop('cell_id')
        actual_cell.pop('cell_id')
        assert actual_cell == expected_cell
