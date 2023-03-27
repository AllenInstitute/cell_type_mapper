# script to construct the taxonomy tree from the cl.df.3levels.csv
# file specified for the test

import anndata
import h5py
import json
import numpy as np
import pathlib

from hierarchical_mapping.utils.taxonomy_utils import (
    validate_taxonomy_tree)

from hierarchical_mapping.corr.utils import (
    match_genes)

from hierarchical_mapping.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad_and_tree)

def get_gene_name_lookup(h5ad_path):
    a_data = anndata.read_h5ad(h5ad_path, backed='r')
    gene_name_list = list(a_data.var_names)
    gene_name_lookup = {n:ii for ii, n in enumerate(gene_name_list)}
    return gene_name_lookup

def find_marker_files(
        marker_dir,
        taxonomy_tree,
        level,
        level_number):
    """
    return dict mapping (level, parent) to path to marker genes
    """
    good = 0
    bad = 0
    to_pop = []

    result = dict()
    for k in taxonomy_tree[level]:
        pth = k.replace(' ', '+').replace('/','__')
        pth = marker_dir / f"marker.{level_number}.{pth}.csv"
        if not pth.is_file():
            #print(f"could not find {pth}")
            bad += 1
        else:
            good += 1
            this_key = (level,k)
            assert this_key not in result
            result[this_key] = pth
    print(f"good {good} bad {bad}")
    return result

def read_taxonomy_tree(tree_src_file):
    tree = dict()
    tree['hierarchy'] = ['level_1', 'level_2', 'cluster']
    for level in tree['hierarchy']:
        tree[level] = dict()
    with open(tree_src_file, 'r') as in_file:
        in_file.readline()
        for line in in_file:
            params = line.strip().split('","')
            cluster = params[0].replace('"','')
            level_1 = params[1].replace('"','')
            level_2 = params[2].replace('"', '')
            if cluster not in tree['cluster']:
                tree['cluster'][cluster] = []
            if level_1 not in tree['level_1']:
                tree['level_1'][level_1] = set()
            if level_2 not in tree['level_2']:
                tree['level_2'][level_2] = set()
            tree['level_1'][level_1].add(level_2)
            tree['level_2'][level_2].add(cluster)
    for k in tree['level_1']:
        tree['level_1'][k] = list(tree['level_1'][k])
    for k in tree['level_2']:
        tree['level_2'][k] = list(tree['level_2'][k])

    validate_taxonomy_tree(tree)
    return tree


def format_markers(
        marker_path,
        reference_gene_lookup,
        query_gene_lookup):
    marker_names = []
    with open(marker_path, 'r') as in_file:
        in_file.readline()
        for line in in_file:
            marker_names.append(line.strip().replace('"',''))

    reference_idx = []
    query_idx = []
    for n in marker_names:
        if n not in reference_gene_lookup or n not in query_gene_lookup:
            continue
        reference_idx.append(reference_gene_lookup[n])
        query_idx.append(query_gene_lookup[n])
    assert len(reference_idx) > 10
    reference_idx = np.array(reference_idx)
    query_idx = np.array(query_idx)
    sorted_dex = np.argsort(reference_idx)
    reference_idx = reference_idx[sorted_dex]
    query_idx = query_idx[sorted_dex]
    return {'reference': reference_idx, 'query': query_idx}

def create_marker_cache(
        reference_path,
        query_path,
        marker_path_lookup,
        output_path):

    #reference_gene_lookup = get_gene_name_lookup(reference_path)
    #query_gene_lookup = get_gene_name_lookup(query_path)

    #with open('reference_gene_lookup.json', 'w') as out_file:
    #    out_file.write(json.dumps(reference_gene_lookup))
    #with open('query_gene_lookup.json', 'w') as out_file:
    #    out_file.write(json.dumps(query_gene_lookup))
    #exit()

    with open('reference_gene_lookup.json', 'rb') as in_file:
        reference_gene_lookup = json.load(in_file)
    with open('query_gene_lookup.json', 'rb') as in_file:
        query_gene_lookup = json.load(in_file)

    with h5py.File(output_path, 'w') as out_file:
        for grp in marker_path_lookup:
            if 'root' not in grp:
                grp_key = f'{grp[0]}/{grp[1]}'
            else:
                print(f'making key "None" for -- {grp}')
                grp_key = 'None'
            out_file.create_group(grp_key)
            this_grp = format_markers(
                reference_gene_lookup=reference_gene_lookup,
                query_gene_lookup=query_gene_lookup,
                marker_path=marker_path_lookup[grp])
            for k in ('reference', 'query'):
                out_file[grp_key].create_dataset(
                    k, data = this_grp[k])

def assign_rows_to_tree(
        taxonomy_tree,
        reference_path):
    a_data = anndata.read_h5ad(reference_path, backed='r')
    obs = a_data.obs
    cl_values = list(obs.cl.values)
    for i_row, cl in enumerate(cl_values):
        cl = str(cl)
        taxonomy_tree['cluster'][cl].append(i_row)
    return taxonomy_tree


def main():
    reference_path = pathlib.Path(
        '/allen/programs/celltypes/workgroups/rnaseqanalysis/changkyul/CIRRO/U19_CR6/processed.U19_all.postQC.AIT17.0.20230226.sync.h5ad')
    assert reference_path.is_file()

    query_path = pathlib.Path(
        '/allen/programs/celltypes/workgroups/rnaseqanalysis/changkyul/CIRRO/MFISH/atlas_brain_638850.remap.4334174.updated.imputed.h5ad')
    assert query_path.is_file()

    tree_src_file = pathlib.Path(
       "/allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/Taxonomies/AIT17.0_mouse/Templates/cl.df.3levels.csv")
    assert tree_src_file.is_file()

    marker_dir = pathlib.Path(
        '/allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/Taxonomies/AIT17.0_mouse/Templates/marker_list_on_nodes')
    assert marker_dir.is_dir()

    tree = read_taxonomy_tree(tree_src_file)

    print('=======root========')
    root_markers = find_marker_files(
        marker_dir=marker_dir,
        taxonomy_tree = {None: ['root']},
        level=None,
        level_number=1)

    print('=====level_1=====')
    level_1_markers = find_marker_files(
        marker_dir=marker_dir,
        taxonomy_tree=tree,
        level='level_1',
        level_number=2)

    print('=====level_2=====')
    level_2_markers = find_marker_files(
        marker_dir=marker_dir,
        taxonomy_tree=tree,
        level='level_2',
        level_number=3)

    markers = dict()
    markers.update(root_markers)
    print(len(markers))

    markers.update(level_1_markers)
    print(len(markers))

    markers.update(level_2_markers)
    print(len(markers))

    create_marker_cache(
        reference_path=reference_path,
        query_path=query_path,
        marker_path_lookup=markers,
        output_path='validation_marker_cache.h5')

    tree = assign_rows_to_tree(
            taxonomy_tree=tree,
            reference_path=reference_path)
    with open('taxonomy_tree.json', 'w') as out_file:
        out_file.write(json.dumps(tree,indent=2))

    print("precomputing summary stats")
    precompute_summary_stats_from_h5ad_and_tree(
        data_path=reference_path,
        taxonomy_tree=tree,
        output_path='validation_test_precompute.h5')


if __name__ == "__main__":
    main()
