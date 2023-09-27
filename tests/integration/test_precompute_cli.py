"""
This module will test the CLI tool to compute a taxonomy tree
and precomputed stats file from a set of files that looks like
the June 2023 ABC Atlas data release
"""
import pytest

import anndata
import h5py
import itertools
import json
import numpy as np
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)

from cell_type_mapper.cli.precompute_stats import (
    PrecomputationRunner)


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
    for ii in range(1234):
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
    tmp_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.csv')
    """
    Simulates CSV that associates cell_name with cluster alias
    """
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

    runner = PrecomputationRunner(
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
