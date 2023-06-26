import pytest

import numpy as np
import pathlib
import tempfile

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.taxonomy.data_release_utils import (
    get_tree_above_leaves,
    get_label_to_name,
    get_cell_to_cluster_alias)

from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from hierarchical_mapping.data.gene_id_lookup import (
    gene_id_lookup)


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
    class_lookup = {n:None
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

    #label is the label of this node
    #cluster_annotation_term_set_label is somethign like 'subclass' or 'supertype'
    #parent_term_label is the parent of this
    #parent_term_set_label is what kind of thing parent is

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
            junk_line = ",".join([_create_word(rng) for ii in range(len(columns))])
            junk_line += "\n"
            line_list.append(junk_line)
        rng.shuffle(line_list)
        for line in line_list:
            dst.write(line)
    return tmp_path


@pytest.fixture(scope='module')
def baseline_tree_fixture(
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

    return TaxonomyTree(data=data)


@pytest.fixture(scope='module')
def expected_marker_lookup_fixture(
        baseline_tree_fixture):
    """
    Return dict of expected marker lookup
    """
    parent_list = baseline_tree_fixture.all_parents
    rng = np.random.default_rng(88122312)

    gene_symbol_list = []
    for gene_id in gene_id_lookup:
        if not gene_id.startswith('ENS'):
            continue
        gene_symbol_list.append(gene_id_lookup[gene_id]['gene_symbol'])

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
        expected_marker_lookup_fixture,
        cluster_membership_fixture,
        cell_metadata_fixture,
        cluster_annotation_term_fixture,
        tmp_dir_fixture):
    """
    Populate a directory with the marker gene files.
    Intentionally leave out one of the expected files.
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

    ct = 0
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
        ct += 1
        if ct == 5:
            continue
        with open(file_path, 'w') as out_file:
            out_file.write("a header\n")
            for gene in gene_list:
                out_file.write(f'"{gene}"\n')
    return marker_dir
