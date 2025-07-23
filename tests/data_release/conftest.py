import pytest

import copy
import numpy as np
import pandas as pd
import pathlib
import shutil
import tempfile
import warnings

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

import cell_type_mapper.test_utils.gene_mapping.mappers as gene_mappers


def _create_word(rng):
    alphabet = [
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
def cell_metadata_fixture(
        tmp_dir_fixture,
        cell_to_cluster_fixture,
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
        out_file.write('nonsense,cell_label,more_nonsense,'
                       'cluster_alias,woah\n')
        for cell_name in cell_to_cluster_fixture:
            cluster_name = cell_to_cluster_fixture[cell_name]
            alias = alias_fixture[cluster_name]
            out_file.write(
                f"{rng.integers(99, 1111)},{cell_name},"
                f"{rng.integers(88, 10000)},"
                f"{alias},{rng.random()}\n")
    return pathlib.Path(tmp_path)


@pytest.fixture(scope='module')
def cell_to_cluster_membership_fixture(
        tmp_dir_fixture,
        cell_metadata_fixture):
    """
    Break the cell_metadata_fixture file up so that cluster_alias
    is actually recorded in a cell_to_cluster_membership file
    (along with some extra cells that aren't in cell_metadata)

    Return the path to the new cell_metadata.csv and the
    cell_to_cluster_membership.csv
    """
    new_cell_metadata_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='cell_metadata_',
        suffix='.csv'
    )
    cell_to_cluster_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='cell_to_cluster_membership_',
        suffix='.csv'
    )
    src = pd.read_csv(cell_metadata_fixture)
    cell_to_cluster = src[
        ['cell_label', 'cluster_alias']
    ].to_dict(orient='records')
    cell_to_cluster.append(
            {'cell_label': 'not_really_a_cell',
             'cluster_alias': 'bogus_alias'}
    )
    cell_to_cluster = pd.DataFrame(cell_to_cluster)
    cell_to_cluster.to_csv(cell_to_cluster_path)
    new_cell_metadata = src[['cell_label', 'nonsense', 'more_nonsense']]
    new_cell_metadata.to_csv(new_cell_metadata_path)
    return {
        'cell_metadata': new_cell_metadata_path,
        'cell_to_cluster': cell_to_cluster_path
    }


@pytest.fixture(scope='module')
def missing_subclass_fixture():
    """
    subclass to trim from incomplete cell_metadata
    """
    return 'subclass_1'


@pytest.fixture(scope='module')
def incomplete_cell_metadata_fixture(
        supertype_to_subclass_fixture,
        cluster_to_supertype_fixture,
        cell_to_cluster_fixture,
        alias_fixture,
        missing_subclass_fixture,
        tmp_dir_fixture):
    """
    A cell_metadata.csv file that leaves out all of the
    cells in the subclass indicated by missing_subclass_fixture
    """
    tmp_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='incomplete_cell_metadata_',
        suffix='.csv')

    rng = np.random.default_rng(5443388)
    with open(tmp_path, 'w') as out_file:
        out_file.write('nonsense,cell_label,more_nonsense,'
                       'cluster_alias,woah\n')
        for cell_name in cell_to_cluster_fixture:
            cluster_name = cell_to_cluster_fixture[cell_name]
            supertype = cluster_to_supertype_fixture[cluster_name]
            subclass = supertype_to_subclass_fixture[supertype]
            if subclass == missing_subclass_fixture:
                continue
            alias = alias_fixture[cluster_name]
            out_file.write(
                f"{rng.integers(99, 1111)},{cell_name},"
                f"{rng.integers(88, 10000)},"
                f"{alias},{rng.random()}\n")
    return pathlib.Path(tmp_path)


@pytest.fixture(scope='module')
def term_label_to_name_fixture(
        cluster_to_supertype_fixture,
        supertype_to_subclass_fixture,
        subclass_to_class_fixture):
    """
    return a dict mapping (level, label) to a human readable name
    """
    result = dict()
    class_lookup = {n: None
                    for n in set(subclass_to_class_fixture.values())}

    for lookup, class_name in [(cluster_to_supertype_fixture, 'cluster'),
                               (supertype_to_subclass_fixture, 'supertype'),
                               (subclass_to_class_fixture, 'subclass'),
                               (class_lookup, 'class')]:
        for child in lookup:
            this_key = (class_name, child)
            assert this_key not in result

            # add the ' ' and '/' so that we can test the munging
            # of node names into csv files the way they come out
            # of CK's R code
            result[this_key] = f'{class_name} {child}/readable'
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
        'cluster_annotation_term_set_name',
        'garbage0',
        'cluster_annotation_term_set_label',
        'garbage1',
        'cluster_alias',
        'cluster_annotation_term_name',
        'garbage2',
        'garbage3',
        'cluster_annotation_term_label',
        'garbage4']

    class_lookup = {n: None
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
                    this += (
                        f'{term_label_to_name_fixture[(class_name, child)]},'
                    )
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

    return pathlib.Path(tmp_path)


@pytest.fixture(scope='module')
def cluster_annotation_term_fixture(
        cluster_to_supertype_fixture,
        supertype_to_subclass_fixture,
        subclass_to_class_fixture,
        tmp_dir_fixture):
    """
    Simulates the CSV that has the parent-child
    relationship of taxonomic levels in it
    """
    tmp_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.csv')

    # label is the label of this node
    # cluster_annotation_term_set_label is
    # something like 'subclass' or 'supertype'
    # parent_term_label is the parent of this
    # parent_term_set_label is what kind of thing parent is

    columns = [
        'cluster_annotation_term_set_name',
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
                    elif column_name == 'cluster_annotation_term_set_name':
                        this += f'{child_class}_readable,'
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
            junk_line = ",".join(
                [_create_word(rng) for ii in range(len(columns))])
            junk_line += "\n"
            line_list.append(junk_line)
        rng.shuffle(line_list)
        for line in line_list:
            dst.write(line)
    return tmp_path


@pytest.fixture(scope='module')
def baseline_tree_data_fixture(
        cell_to_cluster_fixture,
        cluster_to_supertype_fixture,
        supertype_to_subclass_fixture,
        subclass_to_class_fixture):
    data = dict()
    data['hierarchy'] = ['class',
                         'subclass',
                         'supertype',
                         'cluster']

    for lookup, parent_level in [(subclass_to_class_fixture, 'class'),
                                 (supertype_to_subclass_fixture, 'subclass'),
                                 (cluster_to_supertype_fixture, 'supertype'),
                                 (cell_to_cluster_fixture, 'cluster')]:
        this = dict()
        for child_label in lookup:
            parent = lookup[child_label]
            child = child_label

            if parent not in this:
                this[parent] = []
            this[parent].append(child)
        for parent in this:
            this[parent].sort()
        data[parent_level] = this

    return data


@pytest.fixture(scope='module')
def baseline_tree_fixture(
        baseline_tree_data_fixture):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return TaxonomyTree(data=baseline_tree_data_fixture)


@pytest.fixture(scope='module')
def baseline_tree_without_cells_fixture(
        baseline_tree_data_fixture):
    data = copy.deepcopy(baseline_tree_data_fixture)
    for k in data['cluster']:
        data['cluster'][k] = []

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return TaxonomyTree(data=data)


@pytest.fixture(scope='module')
def baseline_incomplete_tree_fixture(
        baseline_tree_data_fixture,
        missing_subclass_fixture):
    """
    Construct a TaxonomyTree in which the subclass indicated by
    missing_subclass_fixture does not exist
    """
    tree_data = copy.deepcopy(baseline_tree_data_fixture)
    for class_name in list(tree_data['class'].keys()):
        if missing_subclass_fixture in tree_data['class'][class_name]:
            tree_data['class'][class_name].remove(missing_subclass_fixture)

    supertype_list = tree_data['subclass'].pop(missing_subclass_fixture)
    cluster_list = []
    for supertype in supertype_list:
        cluster_list += list(tree_data['supertype'].pop(supertype))

    for cluster in cluster_list:
        tree_data['cluster'].pop(cluster)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return TaxonomyTree(data=tree_data)


@pytest.fixture(scope='module')
def expected_marker_lookup_fixture(
        baseline_tree_fixture):
    """
    Return dict of expected marker lookup
    """
    cellranger_6_lookup = gene_mappers.get_cellranger_gene_id_mapping()
    parent_list = baseline_tree_fixture.all_parents
    rng = np.random.default_rng(88122312)

    gene_symbol_list = []
    used_ens = set()
    for gene_id in cellranger_6_lookup:
        if len(cellranger_6_lookup[gene_id]) > 1:
            continue
        if cellranger_6_lookup[gene_id][0] in used_ens:
            continue
        used_ens.add(cellranger_6_lookup[gene_id][0])
        gene_symbol_list.append(gene_id)

    true_lookup = dict()
    for parent in parent_list:
        if parent is None:
            parent_key = 'None'
        else:
            parent_key = f'{parent[0]}/{parent[1]}'
            if len(baseline_tree_fixture.children(parent[0], parent[1])) < 2:
                continue
        n_genes = rng.integers(10, 30)
        chosen_genes = list(rng.choice(gene_symbol_list,
                                       n_genes,
                                       replace=False))
        chosen_genes.sort()
        true_lookup[parent_key] = chosen_genes

    return true_lookup


@pytest.fixture(scope='module')
def marker_gene_csv_dir(
        expected_marker_lookup_fixture,
        cluster_membership_fixture,
        cell_metadata_fixture,
        cluster_annotation_term_fixture,
        tmp_dir_fixture):
    """
    Populate a directory with the marker gene files.
    Return the path to the dir
    """

    taxonomy_tree = TaxonomyTree.from_data_release(
        cell_metadata_path=cell_metadata_fixture,
        cluster_membership_path=cluster_membership_fixture,
        cluster_annotation_path=cluster_annotation_term_fixture,
        hierarchy=['class', 'subclass', 'supertype', 'cluster'])

    parent_list = taxonomy_tree.all_parents

    marker_dir = pathlib.Path(
            tempfile.mkdtemp(
                dir=tmp_dir_fixture,
                prefix='marker_gene_lists_'))

    hierarchy = taxonomy_tree.hierarchy
    hierarchy_to_idx = {None: 1}
    for idx, h in enumerate(hierarchy[:-1]):
        hierarchy_to_idx[h] = idx+2

    for parent in parent_list:
        if parent is None:
            parent_key = 'None'
            idx = hierarchy_to_idx[parent]
            munged = 'root'
        else:
            parent_key = f'{parent[0]}/{parent[1]}'
            idx = hierarchy_to_idx[parent[0]]
            readable_name = taxonomy_tree.label_to_name(
                level=parent[0],
                label=parent[1],
                name_key='name')
            munged = readable_name.replace(' ', '+').replace('/', '__')
        if parent_key not in expected_marker_lookup_fixture:
            continue

        gene_list = expected_marker_lookup_fixture[parent_key]

        file_name = f'marker.{idx}.{munged}.csv'
        file_path = marker_dir / file_name
        with open(file_path, 'w') as out_file:
            out_file.write("a header\n")
            for gene in gene_list:
                out_file.write(f'"{gene}"\n')
    return marker_dir


@pytest.fixture(scope='module')
def bad_marker_gene_csv_dir(
        marker_gene_csv_dir,
        tmp_dir_fixture):
    """
    Populate a directory with the marker gene files.
    Intentionally leave out one of the expected files.
    Return the path to the dir
    """

    marker_dir = pathlib.Path(
            tempfile.mkdtemp(
                dir=tmp_dir_fixture,
                prefix='marker_gene_lists_'))

    pth_list = [n for n in marker_gene_csv_dir.iterdir()]
    ct = 0
    for pth in pth_list:
        ct += 1
        if ct == 5:
            continue
        new_pth = marker_dir / pth.name
        shutil.copy(src=pth, dst=new_pth)

    return marker_dir


@pytest.fixture(scope='module')
def bad_marker_gene_csv_dir_2(
        marker_gene_csv_dir,
        tmp_dir_fixture):
    """
    Populate a directory with the marker gene files.
    Intentionally mangle contets of one of the files.
    Return the path to the dir
    """

    marker_dir = pathlib.Path(
            tempfile.mkdtemp(
                dir=tmp_dir_fixture,
                prefix='marker_gene_lists_'))

    pth_list = [n for n in marker_gene_csv_dir.iterdir()]
    ct = 0
    for pth in pth_list:
        new_pth = marker_dir / pth.name
        shutil.copy(src=pth, dst=new_pth)
        ct += 1
        if ct == 10:
            with open(new_pth, 'a') as out_file:
                out_file.write('"blah"\n')

    return marker_dir
