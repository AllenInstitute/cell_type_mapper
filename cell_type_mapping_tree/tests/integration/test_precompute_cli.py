"""
This module will test the CLI tool to compute a taxonomy tree
and precomputed stats file from a set of files that looks like
the June 2023 ABC Atlas data release
"""
import pytest

import anndata
import h5py
import numpy as np
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.cli.precompute_stats import (
    PrecomputationRunner)


def _create_word(rng):
    alphabet=[
        n for n in 'abcdefghijklmnopqrstuvwxyz']
    return ''.join(rng.choice(alphabet, 5))


@pytest.fixture
def tmp_dir_fixture(
        tmp_path_factory):

    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('abc_taxonomy_'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def cluster_names_fixture():
    result = []
    for ii in range(1234):
        result.append(f'cluster_{ii}')
    return result

@pytest.fixture
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

@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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

@pytest.fixture
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

@pytest.fixture
def cell_metadata_fixture(
        tmp_dir_fixture,
        cell_to_cluster_fixture,
        alias_fixture):
    tmp_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.csv')
    """
    Simulates CSV that associates cell_name with cluster alias
    """
    rng = np.random.default_rng(5443388)
    with open(tmp_path, 'w') as out_file:
        out_file.write('nonsense,cell_label,more_nonsense,cluster_alias,woah\n')
        for cell_name in cell_to_cluster_fixture:
            cluster_name = cell_to_cluster_fixture[cell_name]
            alias = alias_fixture[cluster_name]
            out_file.write(
                f"{rng.integers(99,1111)},{cell_name},{rng.integers(88,10000)},"
                f"{alias},{rng.random()}\n")
    return tmp_path

@pytest.fixture
def cluster_membership_fixture(
        alias_fixture,
        tmp_dir_fixture,
        cluster_to_supertype_fixture,
        supertype_to_subclass_fixture,
        subclass_to_class_fixture):
    """
    Simulates cluster_to_cluster_annotation_membership.csv
    """
    rng = np.random.default_rng(853211)
    tmp_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.csv')
    rng = np.random.default_rng(76123)
    columns = [
        'garbage0',
        'cluster_annotation_term_set_label',
        'garbage1',
        'cluster_alias',
        'garbage2',
        'garbage3',
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
                elif col == 'cluster_annotation_term_label':
                    this += f'{child},'
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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
    return path_list

def test_precompute_cli(
        cell_metadata_fixture,
        cluster_membership_fixture,
        cluster_annotation_term_fixture,
        h5ad_path_list_fixture,
        tmp_dir_fixture):
    """
    So far, this is just a smoke test that makes sure the
    resulting file has the expected datasets
    """
    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5')

    config = {
        'output_path': output_path,
        'h5ad_path_list': h5ad_path_list_fixture,
        'normalization': 'raw',
        'cell_metadata_path': cell_metadata_fixture,
        'cluster_annotation_path': cluster_annotation_term_fixture,
        'cluster_membership_path': cluster_membership_fixture,
        'hierarchy': ['class', 'subclass', 'supertype', 'cluster']}

    runner = PrecomputationRunner(
        args=[],
        input_data=config)

    runner.run()

    with h5py.File(output_path, 'r') as src:
        src_keys = src.keys()
        for k in ('taxonomy_tree', 'metadata', 'col_names', 'cluster_to_row',
                  'n_cells', 'sum', 'sumsq', 'gt0', 'gt1', 'ge1'):
            assert k in src_keys
