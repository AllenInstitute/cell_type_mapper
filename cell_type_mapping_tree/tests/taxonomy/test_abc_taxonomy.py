"""
This module will test TaxonomyTree against the serialization scheme
adopted for the June 2023 ABC Atlas data release
"""
import pytest

import pathlib
import numpy as np

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)


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
def cluster_alias_fixture(cluster_names_fixture):
    n_clusters = len(cluster_names_fixture)
    values = np.arange(n_clusters)
    rng = np.random.default_rng(664433)
    rng.shuffle(values)
    result = {n:ii for n, ii in zip(cluster_names_fixture, values)}
    return result

@pytest.fixture
def cluster_to_supertype_fixture(cluster_names_fixture):
    result = dict()
    n_super = len(cluster_names_fixture)//3
    assert n_super > 2
    super_type_list = [f'super_type_{ii}'
                       for ii in n_super]
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
    n_class = len(subclasses) // 2
    assert n_class > 2
    classes = [f"class_{ii}" for ii in n_classes]
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
    for ii in range(5556):
        cell_name = f"cell_{ii}"
        chosen_cluster = rng.choice(cluster_names_fixture)
        result[cell_name] = chosen_cluster
    return result

@pytest.fixture
def cell_metadata_fixture(
        tmp_dir_fixture,
        cell_to_cluster_fixture,
        cluster_alias_fixture):
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
            alias = cluster_alias_fixture[cluster_name]
            out_file.write(
                f"{rng.integers(99,1111)},{cell_name},{rng.integers(88,10000)},"
                f"{alias},{rng.random()}\n")
    return pathlib.Path(tmp_path)

@pytest.fixture
def cluster_fixture(cluster_alias_fixture, tmp_dir_fixture):
    """
    Simulates CSV that associates cluster_alias with cluster_name
    """
    tmp_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.csv')
    rng = np.random.default_rng(76123)
    with open(tmp_path, 'w') as out_file:
        out_file.write('junk,cluster_alias,more_junk,label,other_junk\n')
        for cluster in cluster_alias_fixture:
            out_file.write(
                f"{rng.random()},{cluster_alias_fixture[cluster]},"
                f"{rng.integers(8,299)},{cluster},{rng.random()}\n")
    return pathlib.Path(tmp_path)


@pytest.fixture
def cluster_annotation_term_fixture(
        cell_to_cluster_fixture,
        cluster_to_supertype_fixture,
        supertype_to_subclass_fixture,
        subclass_to_class_fixture,
        cluster_alias_fixture,
        tmp_dir_fixture):
    """
    Simulates the CSV that has the parent-child
    relationship of taxonomic levels in it
    """
    tmp_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.csv')

   """
   this is the file we need to mimic
   
   label,name,cluster_annotation_term_set_label,parent_term_label,parent_term_set_l
abel,term_set_order,term_order,cluster_annotation_term_set_name,color_hex_triple
t
CCN20230504_NEUR_Glut,Glut,CCN20230504_NEUR,,,0,0,neurotransmitter,#2B93DF
CCN20230504_NEUR_GABA,GABA,CCN20230504_NEUR,,,0,1,neurotransmitter,#FF3358
CCN20230504_NEUR_Glut-GABA,Glut-GABA,CCN20230504_NEUR,,,0,2,neurotransmitter,#0a
9964
CCN20230504_NEUR_Chol,Chol,CCN20230504_NEUR,,,0,3,neurotransmitter,#73E785
CCN20230504_NEUR_Dopa,Dopa,CCN20230504_NEUR,,,0,4,neurotransmitter,#fcf04b
CCN20230504_NEUR_Hist,Hist,CCN20230504_NEUR,,,0,5,neurotransmitter,#ff7621
CCN20230504_NEUR_Sero,Sero,CCN20230504_NEUR,,,0,6,neurotransmitter,#533691
CCN20230504_NEUR_GABA-Glyc,GABA-Glyc,CCN20230504_NEUR,,,0,7,neurotransmitter,#82
0e57
    """
    label is the label of this node
    cluster_annotation_term_set_label is somethign like 'subclass' or 'supertype'
    parent_term_label is the parent of this
    parent_term_set_label is what kind of thing parent is

    with open(tmp_path, 'w') as dst:


def test_all_this(
        cell_metadata_fixture,
        cluster_fixture):
    pass
