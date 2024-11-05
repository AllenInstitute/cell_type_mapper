"""
This module will test the CLI tool to compute a taxonomy tree
and precomputed stats file from a set of files that looks like
the June 2023 ABC Atlas data release
"""
import pytest

import anndata
import copy
import h5py
import itertools
import json
import numpy as np
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse
import tempfile

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)

from cell_type_mapper.cli.precompute_stats_abc import (
    PrecomputationABCRunner)

from cell_type_mapper.cli.reference_markers import (
    ReferenceMarkerRunner)

from cell_type_mapper.diff_exp.precompute_utils import (
    merge_precompute_files)

from cell_type_mapper.diff_exp.markers import (
    find_markers_for_all_taxonomy_pairs)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)


def _create_word(rng):
    alphabet=[
        n for n in 'abcdefghijklmnopqrstuvwxyz']
    return ''.join(rng.choice(alphabet, 5))


@pytest.fixture(scope='module')
def tmp_dir_fixture(
        tmp_path_factory):

    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('abc_taxonomy_'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture(scope='module')
def cluster_names_fixture():
    result = []
    for ii in range(133):
        result.append(f'cluster_{ii}')
    return result

@pytest.fixture(scope='module')
def cluster_to_supertype_fixture(cluster_names_fixture):
    result = dict()
    n_super = len(cluster_names_fixture)//3
    assert n_super > 2
    super_type_list = [f'supertype_{ii}'
                       for ii in range(n_super)]
    rng = np.random.default_rng()
    for cl in cluster_names_fixture:
        chosen_super = rng.choice(super_type_list)
        result[cl] = chosen_super
    return result

@pytest.fixture(scope='module')
def supertype_to_subclass_fixture(cluster_to_supertype_fixture):
    supertypes = list(set(cluster_to_supertype_fixture.values()))
    supertypes.sort()
    n_sub = len(supertypes) // 2
    assert n_sub > 2
    rng = np.random.default_rng(22314)
    result = dict()
    subclasses = [f'subclass_{ii}' for ii in range(n_sub)]
    for super_t in supertypes:
        chosen_sub = rng.choice(subclasses)
        result[super_t] = chosen_sub
    return result


@pytest.fixture(scope='module')
def subclass_to_class_fixture(supertype_to_subclass_fixture):
    subclasses = list(set(supertype_to_subclass_fixture.values()))
    subclasses.sort()
    n_classes = len(subclasses) // 2
    assert n_classes > 2
    classes = [f"class_{ii}" for ii in range(n_classes)]
    result = dict()
    rng = np.random.default_rng(667788)
    for sub_c in subclasses:
        chosen_c = rng.choice(classes)
        result[sub_c] = chosen_c
    return result


@pytest.fixture(scope='module')
def cell_to_cluster_fixture(cluster_names_fixture):
    result = dict()
    rng = np.random.default_rng(2233311)

    # make sure every cluster has at least one cell
    for ii in range(len(cluster_names_fixture)):
        cell_name = f"cheating_cell_{ii}"
        result[cell_name] = cluster_names_fixture[ii]

    for ii in range(5556):
        cell_name = f"cell_{ii}"
        chosen_cluster = rng.choice(cluster_names_fixture)
        result[cell_name] = chosen_cluster
    return result

@pytest.fixture(scope='module')
def missing_subclass_fixture():
    """
    Name of the subclass to omit from incomplete_cell_metadata
    """
    return "subclass_1"


@pytest.fixture(scope='module')
def alias_fixture(
        cluster_to_supertype_fixture,
        supertype_to_subclass_fixture,
        subclass_to_class_fixture):
    """
    Return a lookup between (arbitrary) label and alias.
    Make sure aliases are unique within levels, but not
    across levels.
    """
    result = dict()
    class_lookup = {n: None
                    for n in set(subclass_to_class_fixture.values())}
    for lookup in [cluster_to_supertype_fixture,
                   supertype_to_subclass_fixture,
                   subclass_to_class_fixture,
                   class_lookup]:
        alias = 0
        for label in lookup:
            assert label not in result
            result[label] = alias
            alias += 1
    return result


@pytest.fixture(scope='module')
def dataset_list_fixture():
    return ['dataset1', 'dataset2', 'dataset3']


@pytest.fixture(scope='module')
def cell_to_dataset_fixture(
        cell_to_cluster_fixture,
        dataset_list_fixture):
    rng = np.random.default_rng(2212)
    lookup = dict()
    for cell_id in cell_to_cluster_fixture.keys():
        chosen = rng.choice(dataset_list_fixture)
        lookup[cell_id] = chosen
    return lookup


@pytest.fixture(scope='module')
def cell_metadata_fixture(
        tmp_dir_fixture,
        cell_to_cluster_fixture,
        cell_to_dataset_fixture,
        alias_fixture):
    """
    Simulates CSV that associates cell_name with cluster alias
    """

    tmp_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='cell_metadata_',
        suffix='.csv')

    rng = np.random.default_rng(5443388)
    with open(tmp_path, 'w') as out_file:
        out_file.write(
            'nonsense,cell_label,more_nonsense,cluster_alias,woah,dataset_label\n')
        for cell_name in cell_to_cluster_fixture:
            cluster_name = cell_to_cluster_fixture[cell_name]
            dataset_label = cell_to_dataset_fixture[cell_name]
            alias = alias_fixture[cluster_name]
            out_file.write(
                f"{rng.integers(99,1111)},{cell_name},{rng.integers(88,10000)},"
                f"{alias},{rng.random()},{dataset_label}\n")
    return tmp_path


@pytest.fixture(scope='module')
def incomplete_cell_metadata_fixture(
        tmp_dir_fixture,
        cell_to_cluster_fixture,
        cell_to_dataset_fixture,
        alias_fixture,
        missing_subclass_fixture,
        cluster_to_supertype_fixture,
        supertype_to_subclass_fixture):
    """
    Simulates CSV that associates cell_name with cluster alias.
    Omits all cells assigned to missing_subclass_fixture
    """
    tmp_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='incomplete_cell_metadata_',
        suffix='.csv')

    rng = np.random.default_rng(5443388)
    with open(tmp_path, 'w') as out_file:
        out_file.write(
            'nonsense,cell_label,more_nonsense,cluster_alias,woah,dataset_label\n')
        for cell_name in cell_to_cluster_fixture:

            cluster_name = cell_to_cluster_fixture[cell_name]
            supertype = cluster_to_supertype_fixture[cluster_name]
            subclass = supertype_to_subclass_fixture[supertype]

            if subclass == missing_subclass_fixture:
                continue

            dataset_label = cell_to_dataset_fixture[cell_name]
            alias = alias_fixture[cluster_name]
            out_file.write(
                f"{rng.integers(99,1111)},{cell_name},{rng.integers(88,10000)},"
                f"{alias},{rng.random()},{dataset_label}\n")

    return tmp_path


@pytest.fixture(scope='module')
def term_label_to_name_fixture(
        cluster_to_supertype_fixture,
        supertype_to_subclass_fixture,
        subclass_to_class_fixture):
    """
    return a dict mapping (level, label) to a human readable name
    """
    result = dict()
    class_lookup = {n:None
                    for n in set(subclass_to_class_fixture.values())}

    for lookup, class_name in [(cluster_to_supertype_fixture, 'cluster'),
                               (supertype_to_subclass_fixture, 'supertype'),
                               (subclass_to_class_fixture, 'subclass'),
                               (class_lookup, 'class')]:
        for child in lookup:
            this_key = (class_name, child)
            assert this_key not in result
            result[this_key] = f'{class_name}_{child}_readable'
    return result


@pytest.fixture(scope='module')
def cluster_membership_fixture(
        alias_fixture,
        tmp_dir_fixture,
        cluster_to_supertype_fixture,
        supertype_to_subclass_fixture,
        subclass_to_class_fixture,
        term_label_to_name_fixture):
    """
    Simulates cluster_to_cluster_annotation_membership.csv
    """
    rng = np.random.default_rng(853211)
    tmp_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.csv')
    rng = np.random.default_rng(76123)
    columns = [
        'garbage0',
        'cluster_annotation_term_set_name',
        'garbage1',
        'cluster_alias',
        'garbage2',
        'garbage3',
        'cluster_annotation_term_set_label',
        'cluster_annotation_term_name',
        'cluster_annotation_term_label',
        'garbage4']

    class_lookup = {n:None
                    for n in set(subclass_to_class_fixture.values())}

    lines = []
    for lookup, class_name in [(cluster_to_supertype_fixture, 'cluster'),
                               (supertype_to_subclass_fixture, 'supertype'),
                               (subclass_to_class_fixture, 'subclass'),
                               (class_lookup, 'class')]:
        for child in lookup:
            this = ''
            for col in columns:
                if 'garbage' in col:
                    this += f'{_create_word(rng)},'
                elif col == 'cluster_alias':
                    this += f'{alias_fixture[child]},'
                elif col == 'cluster_annotation_term_set_label':
                    this += f'{class_name},'
                elif col == 'cluster_annotation_term_set_name':
                    this += f'{class_name}_readable,'
                elif col == 'cluster_annotation_term_label':
                    this += f'{child},'
                elif col == 'cluster_annotation_term_name':
                    this += f'{term_label_to_name_fixture[(class_name, child)]},'
                else:
                    raise RuntimeError(f'cannot parse column {col}')
            this = this[:-1]+'\n'
            lines.append(this)
    rng.shuffle(lines)

    with open(tmp_path, 'w') as dst:
        dst.write(','.join(columns))
        dst.write('\n')
        for line in lines:
            dst.write(line)

    return tmp_path


@pytest.fixture(scope='module')
def cluster_annotation_term_fixture(
        cluster_to_supertype_fixture,
        supertype_to_subclass_fixture,
        subclass_to_class_fixture,
        alias_fixture,
        tmp_dir_fixture):
    """
    Simulates the CSV that has the parent-child
    relationship of taxonomic levels in it
    """
    tmp_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.csv')

    #label is the label of this node
    #cluster_annotation_term_set_label is somethign like 'subclass' or 'supertype'
    #parent_term_label is the parent of this
    #parent_term_set_label is what kind of thing parent is

    columns = [
        'garbage0',
        'garbage1',
        'label',
        'parent_term_label',
        'cluster_annotation_term_set_label',
        'garbage2',
        'parent_term_set_label']

    rng = np.random.default_rng(8123)

    with open(tmp_path, 'w') as dst:
        dst.write(','.join(columns))
        dst.write('\n')

        line_list = []
        for child_lookup, child_class, parent_class in [
                    (supertype_to_subclass_fixture, 'supertype', 'subclass'),
                    (subclass_to_class_fixture, 'subclass', 'class'),
                    (cluster_to_supertype_fixture, 'cluster', 'supertype')]:
            for child in child_lookup:
                this = ''
                for column_name in columns:
                    if 'garbage' in column_name:
                        this += f'{_create_word(rng)},'
                    elif column_name == 'cluster_annotation_term_set_label':
                        this += f'{child_class},'
                    elif column_name == 'parent_term_set_label':
                        this += f'{parent_class},'
                    elif column_name == 'parent_term_label':
                        this += f'{child_lookup[child]},'
                    elif column_name == 'label':
                        this += f'{child},'
                    else:
                        raise RuntimeError(
                            f"unknown column {column_name}")
                this = this[:-1]+"\n"
                line_list.append(this)
        for ii in range(20):
            junk_line = ",".join([_create_word(rng) for ii in range(len(columns))])
            junk_line += "\n"
            line_list.append(junk_line)
        rng.shuffle(line_list)
        for line in line_list:
            dst.write(line)
    return tmp_path


@pytest.fixture(scope='module')
def x_fixture(
        cell_to_cluster_fixture):
    rng = np.random.default_rng(5678)
    n_genes= 239
    n_cells = len(cell_to_cluster_fixture)
    data = np.zeros(n_cells*n_genes, dtype=float)
    chosen = rng.choice(np.arange(n_cells*n_genes),
                        n_cells*n_genes//3,
                        replace=False)
    data[chosen] = rng.random(len(chosen))
    data = data.reshape(n_cells, n_genes)
    return data


@pytest.fixture(scope='module')
def h5ad_path_list_fixture(
        cell_to_cluster_fixture,
        x_fixture,
        tmp_dir_fixture):
    rng = np.random.default_rng(7612221)
    path_list = []
    n_cells = x_fixture.shape[0]
    n_genes = x_fixture.shape[1]
    var_data = [
        {'gene_id': f'gene_{ii}', 'garbage': _create_word(rng)}
        for ii in range(n_genes)]
    var = pd.DataFrame(var_data).set_index('gene_id')

    cell_name_list = list(cell_to_cluster_fixture.keys())
    cell_name_list.sort()
    rng.shuffle(cell_name_list)
    n_per = len(cell_name_list) // 4
    assert n_per > 2
    for i0 in range(0, n_cells, n_per):
        i1 = i0+n_per
        cell_names = cell_name_list[i0:i1]
        obs_data = [
            {"cell_id": c,
            "huh": _create_word(rng)}
            for c in cell_names]

        obs = pd.DataFrame(obs_data).set_index('cell_id')
        this_x = scipy_sparse.csr_matrix(x_fixture[i0:i1, :])
        this_a = anndata.AnnData(
            X=this_x,
            obs=obs,
            var=var,
            dtype=x_fixture.dtype)

        this_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5ad')

        this_a.write_h5ad(this_path)

        path_list.append(this_path)
    assert len(path_list) > 3
    return path_list


@pytest.mark.parametrize(
    "downsample_h5ad_list,split_by_dataset",
    itertools.product([True, False], [True, False]))
def test_precompute_cli(
        cell_metadata_fixture,
        cluster_membership_fixture,
        cluster_annotation_term_fixture,
        h5ad_path_list_fixture,
        x_fixture,
        cell_to_cluster_fixture,
        cell_to_dataset_fixture,
        dataset_list_fixture,
        cluster_to_supertype_fixture,
        tmp_dir_fixture,
        downsample_h5ad_list,
        split_by_dataset):
    """
    So far, this is only tests the contents of

    n_cells
    sum
    sumsq
    ge1
    """
    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5')

    if downsample_h5ad_list:
        h5ad_list = [h5ad_path_list_fixture[0],
                     h5ad_path_list_fixture[1]]
    else:
        h5ad_list = h5ad_path_list_fixture

    config = {
        'output_path': output_path,
        'clobber': True,
        'h5ad_path_list': h5ad_list,
        'normalization': 'raw',
        'cell_metadata_path': cell_metadata_fixture,
        'cluster_annotation_path': cluster_annotation_term_fixture,
        'cluster_membership_path': cluster_membership_fixture,
        'hierarchy': ['class', 'subclass', 'supertype', 'cluster'],
        'split_by_dataset': split_by_dataset}

    runner = PrecomputationABCRunner(
        args=[],
        input_data=config)

    runner.run()

    n_genes = x_fixture.shape[1]
    expected_gene_names = [f"gene_{ii}" for ii in range(n_genes)]

    dataset_to_output = dict()
    if split_by_dataset:
        for dataset in dataset_list_fixture:
            new_path = output_path[:-2] + f'{dataset}.h5'
            dataset_to_output[dataset] = new_path
    else:
        dataset_to_output['None'] = output_path
    combined_path = pathlib.Path(output_path[:-2] + 'combined.h5')

    if split_by_dataset:
        assert combined_path.is_file()
    else:
        assert not combined_path.is_file()

    for dataset in dataset_to_output:
        actual_output = dataset_to_output[dataset]

        # expected statistics per cluster
        cluster_to_n_cells = dict()
        cluster_to_sum = dict()
        cluster_to_sumsq = dict()
        cluster_to_ge1 = dict()
        for cluster_name in cluster_to_supertype_fixture:
            cluster_to_n_cells[cluster_name] = 0
            cluster_to_sum[cluster_name] = np.zeros(n_genes, dtype=float)
            cluster_to_sumsq[cluster_name] = np.zeros(n_genes, dtype=float)
            cluster_to_ge1[cluster_name] = np.zeros(n_genes, dtype=int)

        for h5ad_path in h5ad_list:
            a_data = anndata.read_h5ad(h5ad_path)
            obs = a_data.obs

            cell_by_gene = CellByGeneMatrix(
                data = a_data.X.toarray(),
                gene_identifiers=a_data.var.index.values,
                normalization='raw')

            cell_by_gene.to_log2CPM_in_place()

            for i_row, cell_id in enumerate(obs.index.values):
                if dataset == 'None' or cell_to_dataset_fixture[cell_id] == dataset:
                    cluster_name = cell_to_cluster_fixture[cell_id]
                    cluster_to_n_cells[cluster_name] += 1
                    cluster_to_sum[cluster_name] += cell_by_gene.data[i_row, :]
                    cluster_to_sumsq[cluster_name] += cell_by_gene.data[i_row,:]**2
                    ge1 = (cell_by_gene.data[i_row, :] >= 1)
                    cluster_to_ge1[cluster_name][ge1] += 1

        with h5py.File(actual_output, 'r') as src:
            src_keys = src.keys()
            for k in ('taxonomy_tree', 'metadata', 'col_names', 'cluster_to_row',
                      'n_cells', 'sum', 'sumsq', 'gt0', 'gt1', 'ge1'):
                assert k in src_keys

            actual_gene_names = json.loads(src['col_names'][()].decode('utf-8'))
            assert actual_gene_names == expected_gene_names

            # only test cluster stats at this point
            cluster_to_row = json.loads(
                src['cluster_to_row'][()].decode('utf-8'))

            n_cells = src['n_cells'][()]
            sum_arr = src['sum'][()]
            sumsq_arr = src['sumsq'][()]
            ge1_arr = src['ge1'][()]
            for cluster_name in cluster_to_n_cells:
                i_row = cluster_to_row[cluster_name]
                assert n_cells[i_row] == cluster_to_n_cells[cluster_name]

                np.testing.assert_allclose(
                    sum_arr[i_row, :],
                    cluster_to_sum[cluster_name],
                    atol=0.0,
                    rtol=1.0e-6)

                np.testing.assert_allclose(
                    sumsq_arr[i_row, :],
                    cluster_to_sumsq[cluster_name],
                    atol=0.0,
                    rtol=1.0e-6)

                np.testing.assert_array_equal(
                    ge1_arr[i_row, :],
                    cluster_to_ge1[cluster_name])

            metadata = json.loads(src['metadata'][()].decode('utf-8'))
            assert 'timestamp' in metadata
            assert 'dataset' in metadata
            assert 'config' in metadata
            for k in config:
                assert metadata['config'][k] == config[k]

    if split_by_dataset:
        # check contents of merged files
        expected_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.h5')
        path_list = list(dataset_to_output.values())
        merge_precompute_files(
            precompute_path_list=path_list,
            output_path=expected_path)
        with h5py.File(expected_path, 'r') as expected:
            with h5py.File(combined_path, 'r') as actual:
                for k in ('n_cells', 'sum', 'sumsq', 'ge1', 'gt0', 'gt1'):
                    np.testing.assert_allclose(
                        expected[k][()],
                        actual[k][()],
                        atol=0.0,
                        rtol=1.0e-6)


@pytest.fixture(scope='module')
def precomputed_stats_path_fixture(
        h5ad_path_list_fixture,
        cell_metadata_fixture,
        cluster_annotation_term_fixture,
        cluster_membership_fixture,
        dataset_list_fixture,
        tmp_dir_fixture):
    """
    List of properly produced precomputed stats files
    (exclude the 'combined' file)
    """

    output_dir = tempfile.mkdtemp(dir=tmp_dir_fixture)
    output_path = f'{output_dir}/precomputed_stats.h5'

    config = {
        'output_path': output_path,
        'clobber': True,
        'h5ad_path_list': h5ad_path_list_fixture,
        'normalization': 'raw',
        'cell_metadata_path': cell_metadata_fixture,
        'cluster_annotation_path': cluster_annotation_term_fixture,
        'cluster_membership_path': cluster_membership_fixture,
        'hierarchy': ['class', 'subclass', 'supertype', 'cluster'],
        'split_by_dataset': True}

    runner = PrecomputationABCRunner(
        args=[],
        input_data=config)

    runner.run()

    precomputed_stats_path_list = [
        str(n) for n in pathlib.Path(output_dir).iterdir()
        if n.is_file and 'combined' not in n.name]

    assert len(precomputed_stats_path_list) == len(dataset_list_fixture)
    return precomputed_stats_path_list


@pytest.mark.parametrize(
    "files_exist",[True, False])
def test_reference_cli_config(
        precomputed_stats_path_fixture,
        dataset_list_fixture,
        tmp_dir_fixture,
        files_exist):
    """
    Test that the reference marker CLI tool creates expected
    files and throws expected errors.

    This test is here because we are using the pre-established
    multi dataset infrastructure to create the precomputed data
    paths.
    """
    valid_output_dir = tempfile.mkdtemp(dir=tmp_dir_fixture)

    # paths that we expect the ReferenceMarker
    # CLI tool will produce
    default_output_paths = []
    for pth in precomputed_stats_path_fixture:
        pth = pathlib.Path(pth)
        old_name = pth.name
        old_stem = old_name.split('.')[0]
        new_name = old_name.replace(old_stem, 'reference_markers', 1)
        default_output_paths.append(f'{valid_output_dir}/{new_name}')

    if files_exist:
        # create paths at teh default locations, forcing the CLI
        # tool to salt its output
        for pth in default_output_paths:
            with open(pth, 'wb') as out_file:
                out_file.write(b'junk')

    config = {
        'precomputed_path_list': precomputed_stats_path_fixture,
        'output_dir': valid_output_dir,
        'clobber': False,
        'drop_level': None,
        'tmp_dir': str(tmp_dir_fixture),
        'n_processors': 4,
        'exact_penetrance': False,
        'p_th': 0.5,
        'q1_th': 0.5,
        'q1_min_th': 0.01,
        'qdiff_th': 0.5,
        'qdiff_min_th': 0.01,
        'log2_fold_th': 1.0,
        'log2_fold_min_th': 0.01,
        'n_valid': 5
    }

    if files_exist:
        with pytest.raises(RuntimeError, match="already exists; to overwrite"):
            runner = ReferenceMarkerRunner(
                args=[],
                input_data=config)
            runner.run()
        for pth in default_output_paths:
            pth = pathlib.Path(pth)
            assert pth.is_file()

    else:
        runner = ReferenceMarkerRunner(
            args=[],
            input_data=config)
        runner.run()

        output_list = [str(n) for n in pathlib.Path(valid_output_dir).iterdir()]
        assert set(output_list) == set(default_output_paths)

        found_precompute_paths = []
        for pth in default_output_paths:
            with h5py.File(pth, 'r') as src:
                metadata = json.loads(src['metadata'][()].decode('utf-8'))
                found_precompute_paths.append(metadata['precomputed_path'])

                # assert that there are some markers in this file
                ntot = np.diff(src['sparse_by_pair/up_pair_idx'][()])
                ntot += np.diff(src['sparse_by_pair/down_pair_idx'][()])
                assert ntot.sum() > 0

                # re-run the analysis, to make sure results align
                expected_path = mkstemp_clean(
                    dir=tmp_dir_fixture,
                    suffix='.h5')

                find_markers_for_all_taxonomy_pairs(
                    precomputed_stats_path=metadata['precomputed_path'],
                    taxonomy_tree=TaxonomyTree.from_precomputed_stats(
                            metadata['precomputed_path']),
                    output_path=expected_path,
                    tmp_dir=tmp_dir_fixture,
                    n_processors=config['n_processors'],
                    p_th=config['p_th'],
                    q1_th=config['q1_th'],
                    q1_min_th=config['q1_min_th'],
                    qdiff_th=config['qdiff_th'],
                    qdiff_min_th=config['qdiff_min_th'],
                    log2_fold_th=config['log2_fold_th'],
                    log2_fold_min_th=config['log2_fold_min_th'],
                    n_valid=config['n_valid'])

                with h5py.File(expected_path, 'r') as expected:
                    for grp in ('sparse_by_pair', 'sparse_by_gene'):
                        for dataset in expected[grp].keys():
                            np.testing.assert_array_equal(
                                src[f'{grp}/{dataset}'][()],
                                expected[f'{grp}/{dataset}'][()])

        # make sure every precomputed_stats file got a corresponding
        # marker file
        assert set(found_precompute_paths) == set(
                        precomputed_stats_path_fixture)



@pytest.mark.parametrize(
    "exact_penetrance,drop_level",
    itertools.product(
        [True, False],
        [None, 'subclass']
    )
)
def test_roundtrip_reference_cli_config(
        precomputed_stats_path_fixture,
        dataset_list_fixture,
        tmp_dir_fixture,
        exact_penetrance,
        drop_level):
    """
    Test that the reference marker CLI tool correctly records
    the config dict needed to recreate its results.

    This test is here because we are using the pre-established
    multi dataset infrastructure to create the precomputed data
    paths.
    """

    baseline_output_dir = tempfile.mkdtemp(
        dir=tmp_dir_fixture,
        prefix='reference_roundtrip_baseline_')

    config = {
        'precomputed_path_list': precomputed_stats_path_fixture,
        'output_dir': baseline_output_dir,
        'clobber': False,
        'drop_level': drop_level,
        'tmp_dir': str(tmp_dir_fixture),
        'n_processors': 4,
        'exact_penetrance': exact_penetrance,
        'p_th': 0.5,
        'q1_th': 0.5,
        'q1_min_th': 0.01,
        'qdiff_th': 0.5,
        'qdiff_min_th': 0.01,
        'log2_fold_th': 1.0,
        'log2_fold_min_th': 0.01,
        'n_valid': 5
    }

    runner = ReferenceMarkerRunner(
        args=[],
        input_data=config)
    runner.run()

    result_files = [
        n for n in pathlib.Path(baseline_output_dir).iterdir()
    ]

    new_config = None
    for pth in result_files:
        with h5py.File(pth, 'r') as src:
            metadata = json.loads(src['metadata'][()].decode('utf-8'))
        if new_config is None:
            new_config = metadata['config']
        else:
            assert new_config == metadata['config']

    new_config.pop('output_dir')
    test_output_dir = tempfile.mkdtemp(
        dir=tmp_dir_fixture,
        prefix='reference_roundtrip_test_'
    )
    new_config['output_dir'] = test_output_dir
    new_runner = ReferenceMarkerRunner(
        args=[],
        input_data=new_config)
    new_runner.run()

    test_files = [
        n for n in pathlib.Path(test_output_dir).iterdir()
    ]

    def _h5_match(obj0, obj1):
        if isinstance(obj0, h5py.Dataset):
            d0 = obj0[()]
            d1 = obj1[()]
            if isinstance(d0, np.ndarray):
                np.testing.assert_allclose(
                    d0,
                    d1,
                    atol=0.0,
                    rtol=1.0e-7
                )
            else:
                assert d0 == d1
        else:
            for k in obj0.keys():
                if k == 'metadata':
                    continue
                _h5_match(obj0[k], obj1[k])

    assert len(test_files) == len(result_files)
    for pth in result_files:
        found_it = False
        test_pth = None
        for test_pth_candidate in test_files:
            if test_pth_candidate.name == pth.name:
                found_it = True
                test_pth = test_pth_candidate
                break
        assert found_it
        with h5py.File(pth, 'r') as baseline:
            with h5py.File(test_pth, 'r') as test:
                _h5_match(baseline, test)


@pytest.mark.parametrize(
    "downsample_h5ad_list,split_by_dataset,do_pruning",
    itertools.product([True, False], [True, False], [True, False]))
def test_precompute_cli_incomplete_cell_metadata(
        incomplete_cell_metadata_fixture,
        cluster_membership_fixture,
        cluster_annotation_term_fixture,
        h5ad_path_list_fixture,
        x_fixture,
        cell_to_cluster_fixture,
        cell_to_dataset_fixture,
        dataset_list_fixture,
        cluster_to_supertype_fixture,
        tmp_dir_fixture,
        downsample_h5ad_list,
        split_by_dataset,
        do_pruning):
    """
    A smoketest to make sure that the correct error is raised (or not)
    when cell_metadata.csv does not assign cells to all of the cell types
    in the taxonomy. (Error should not be raised if do_pruning is True)
    """
    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5')

    if downsample_h5ad_list:
        h5ad_list = [h5ad_path_list_fixture[0],
                     h5ad_path_list_fixture[1]]
    else:
        h5ad_list = h5ad_path_list_fixture

    config = {
        'output_path': output_path,
        'clobber': True,
        'h5ad_path_list': h5ad_list,
        'normalization': 'raw',
        'cell_metadata_path': incomplete_cell_metadata_fixture,
        'cluster_annotation_path': cluster_annotation_term_fixture,
        'cluster_membership_path': cluster_membership_fixture,
        'hierarchy': ['class', 'subclass', 'supertype', 'cluster'],
        'split_by_dataset': split_by_dataset,
        'do_pruning': do_pruning}

    runner = PrecomputationABCRunner(
        args=[],
        input_data=config)

    if not do_pruning:
        msg = "is not present in the keys at level cluster"
        with pytest.raises(RuntimeError, match=msg):
            runner.run()
    else:
        runner.run()



@pytest.fixture
def h5ad_path_list_alt_layer_fixture(
        request,
        h5ad_path_list_fixture,
        tmp_dir_fixture):
    """
    Write out alternate set of h5ad files where data is
    recorded in different layers.
    """
    layer = request.param
    result_path_list = []
    for src_path in h5ad_path_list_fixture:
        src = anndata.read_h5ad(src_path, backed='r')
        dst_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix=f'h5ad_alt_layer_',
            suffix='.h5ad'
        )
        xx = np.zeros(src.X.shape, dtype=int)
        if layer == 'dummy':
            layers = {'dummy': src.X[()]}
            raw = None
        elif layer == 'raw':
            layers = None
            raw = {'X': src.X[()]}
        else:
            raise RuntimeError(
                f"Test cannot parse layer '{layer}'"
            )
        dst = anndata.AnnData(
            X=xx,
            obs=src.obs,
            var=src.var,
            layers=layers,
            raw=raw
        )
        dst.write_h5ad(dst_path)
        result_path_list.append(dst_path)
    return {'path': result_path_list, 'layer': layer}




@pytest.mark.parametrize(
    "downsample_h5ad_list,split_by_dataset,h5ad_path_list_alt_layer_fixture",
    itertools.product(
        [True, False],
        [True, False],
        ['dummy', 'raw']),
    indirect=['h5ad_path_list_alt_layer_fixture'])
def test_precompute_cli_from_layers(
        cell_metadata_fixture,
        cluster_membership_fixture,
        cluster_annotation_term_fixture,
        h5ad_path_list_fixture,
        h5ad_path_list_alt_layer_fixture,
        cell_to_cluster_fixture,
        cell_to_dataset_fixture,
        dataset_list_fixture,
        cluster_to_supertype_fixture,
        tmp_dir_fixture,
        downsample_h5ad_list,
        split_by_dataset):
    """
    Run the precomputation CLI twice, once on data stored in X;
    once on identical data stored in another layer. Verify that
    the two results are equal.
    """

    baseline_output_dir = tempfile.mkdtemp(
        dir=tmp_dir_fixture,
        prefix='baseline_')
    test_output_dir = tempfile.mkdtemp(
        dir=tmp_dir_fixture,
        prefix='test_')

    baseline_output_path = mkstemp_clean(
        dir=baseline_output_dir,
        suffix='.h5')

    test_output_path = mkstemp_clean(
        dir=test_output_dir,
        suffix='.h5')

    full_alt_list = h5ad_path_list_alt_layer_fixture['path']
    layer = h5ad_path_list_alt_layer_fixture['layer']
    if layer == 'raw':
        layer = 'raw/X'

    if downsample_h5ad_list:
        h5ad_list = [h5ad_path_list_fixture[0],
                     h5ad_path_list_fixture[1]]
        alt_h5ad_list = [full_alt_list[0],
                         full_alt_list[1]]
    else:
        h5ad_list = h5ad_path_list_fixture
        alt_h5ad_list = full_alt_list

    baseline_config = {
        'output_path': baseline_output_path,
        'clobber': True,
        'h5ad_path_list': h5ad_list,
        'normalization': 'raw',
        'cell_metadata_path': cell_metadata_fixture,
        'cluster_annotation_path': cluster_annotation_term_fixture,
        'cluster_membership_path': cluster_membership_fixture,
        'hierarchy': ['class', 'subclass', 'supertype', 'cluster'],
        'split_by_dataset': split_by_dataset,
        'layer': 'X'}

    baseline_runner = PrecomputationABCRunner(
        args=[],
        input_data=baseline_config
    )

    baseline_runner.run()

    test_config = copy.deepcopy(baseline_config)
    test_config.pop('output_path')
    test_config.pop('h5ad_path_list')
    test_config.pop('layer')

    test_config['output_path'] = test_output_path
    test_config['h5ad_path_list'] = alt_h5ad_list
    test_config['layer'] = layer

    test_runner = PrecomputationABCRunner(
        args=[],
        input_data=test_config
    )

    test_runner.run()

    if not split_by_dataset:
        with h5py.File(baseline_output_path, 'r') as base:
            with h5py.File(test_output_path, 'r') as test:
                for k in ('n_cells', 'sum', 'sumsq', 'ge1', 'gt0', 'gt1'):
                    np.testing.assert_allclose(
                        base[k][()],
                        test[k][()],
                        atol=0.0,
                        rtol=1.0e-7
                    )
                for k in ('cluster_to_row', 'col_names'):
                    assert base[k][()] == test[k][()]
    else:
        # test on dataset 1, 2, 3 (suffix before .h5)
        # have to blow away specified file, first.

        # remove specified output paths, which would never
        # have been correctly populated in the
        # split_by_dataset case.
        pathlib.Path(baseline_output_path).unlink()
        pathlib.Path(test_output_path).unlink()

        # create lookups linking dataset suffix to file path
        # for output files
        baseline_file_lookup = {
            n.name.split('.')[-2]: n
            for n in pathlib.Path(baseline_output_dir).iterdir()
        }
        test_file_lookup = {
            n.name.split('.')[-2]: n
            for n in pathlib.Path(test_output_dir).iterdir()
        }

        assert set(baseline_file_lookup.keys()) == set(test_file_lookup.keys())

        for suffix in baseline_file_lookup:
            baseline_path = baseline_file_lookup[suffix]
            test_path = test_file_lookup[suffix]
            with h5py.File(baseline_path, 'r') as base:
                with h5py.File(test_path, 'r') as test:
                    for k in ('n_cells', 'sum', 'sumsq', 'ge1', 'gt0', 'gt1'):
                        np.testing.assert_allclose(
                            base[k][()],
                            test[k][()],
                            atol=0.0,
                            rtol=1.0e-7
                        )
                    for k in ('cluster_to_row', 'col_names'):
                        assert base[k][()] == test[k][()]


@pytest.mark.parametrize(
    "downsample_h5ad_list,split_by_dataset,h5ad_path_list_alt_layer_fixture",
    itertools.product(
        [True, False],
        [True, False],
        ['dummy', 'raw']),
    indirect=['h5ad_path_list_alt_layer_fixture'])
def test_roundtrip_precomputed_abc_config(
        cell_metadata_fixture,
        cluster_membership_fixture,
        cluster_annotation_term_fixture,
        h5ad_path_list_alt_layer_fixture,
        cell_to_cluster_fixture,
        cell_to_dataset_fixture,
        dataset_list_fixture,
        cluster_to_supertype_fixture,
        tmp_dir_fixture,
        downsample_h5ad_list,
        split_by_dataset):
    """
    Test that the config dict recorded in the metadata of the output
    precomputed_stats files 'just works' when passed back into the runner
    """

    baseline_output_dir = tempfile.mkdtemp(
        dir=tmp_dir_fixture,
        prefix='baseline_')
    test_output_dir = tempfile.mkdtemp(
        dir=tmp_dir_fixture,
        prefix='test_')

    baseline_output_path = mkstemp_clean(
        dir=baseline_output_dir,
        suffix='.h5')

    test_output_path = mkstemp_clean(
        dir=test_output_dir,
        suffix='.h5')

    full_alt_list = h5ad_path_list_alt_layer_fixture['path']
    layer = h5ad_path_list_alt_layer_fixture['layer']
    if layer == 'raw':
        layer = 'raw/X'

    if downsample_h5ad_list:
        h5ad_list = [full_alt_list[0],
                     full_alt_list[1]]
    else:
        h5ad_list = full_alt_list

    baseline_config = {
        'output_path': baseline_output_path,
        'clobber': True,
        'h5ad_path_list': h5ad_list,
        'normalization': 'raw',
        'cell_metadata_path': cell_metadata_fixture,
        'cluster_annotation_path': cluster_annotation_term_fixture,
        'cluster_membership_path': cluster_membership_fixture,
        'hierarchy': ['class', 'subclass', 'supertype', 'cluster'],
        'split_by_dataset': split_by_dataset,
        'layer': layer}

    baseline_runner = PrecomputationABCRunner(
        args=[],
        input_data=baseline_config
    )

    baseline_runner.run()

    if split_by_dataset:
        # remove dummy file
        pathlib.Path(baseline_output_path).unlink()
        output_files = [n for n in pathlib.Path(baseline_output_dir).iterdir()]
    else:
        output_files = [baseline_output_path]

    # make sure the same config was written to each output file
    output_config = None
    for pth in output_files:
        with h5py.File(pth, 'r') as src:
            assert src['sumsq'][()].sum() > 0.0
            metadata = json.loads(src['metadata'][()].decode('utf-8'))
            baseline_output_map = metadata['dataset_to_output_map']
        config = metadata['config']
        if output_config is None:
            output_config = config
        else:
            assert output_config == config

    config.pop('output_path')
    config['output_path'] = test_output_path

    test_runner = PrecomputationABCRunner(
        args=[],
        input_data=config)
    test_runner.run()

    if split_by_dataset:
        pathlib.Path(test_output_path).unlink()
        test_output_files = [
            n for n in pathlib.Path(test_output_dir).iterdir()]
    else:
        test_output_files = [test_output_path]

    with h5py.File(test_output_files[0], 'r') as src:
        metadata = json.loads(src['metadata'][()].decode('utf-8'))
        test_output_map = metadata['dataset_to_output_map']

    assert set(baseline_output_map.values()) != set(test_output_map.values())
    assert set(baseline_output_map.keys()) == set(test_output_map.keys())

    for dataset in baseline_output_map:
        b_path = baseline_output_map[dataset]
        t_path = test_output_map[dataset]
        with h5py.File(b_path, 'r') as baseline_src:
            with h5py.File(t_path, 'r') as test_src:
                assert test_src['cluster_to_row'][()] == baseline_src['cluster_to_row'][()]
                test_tree = json.loads(test_src['taxonomy_tree'][()].decode('utf-8'))
                base_tree = json.loads(baseline_src['taxonomy_tree'][()].decode('utf-8'))
                test_tree.pop('metadata')
                base_tree.pop('metadata')
                assert test_tree == base_tree
                for k in ('sum', 'sumsq', 'ge1', 'gt0', 'gt1', 'n_cells'):
                    np.testing.assert_allclose(
                        test_src[k][()],
                        baseline_src[k][()],
                        atol=0.0,
                        rtol=1.0e-7
                    )
