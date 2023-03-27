# script to construct the taxonomy tree from the cl.df.3levels.csv
# file specified for the test

import json
import pathlib
from hierarchical_mapping.utils.taxonomy_utils import (
    validate_taxonomy_tree)

def find_marker_files(
        marker_dir,
        taxonomy_tree,
        level,
        level_number):
    good = 0
    bad = 0
    to_pop = []
    for k in taxonomy_tree[level]:
        pth = k.replace(' ', '+').replace('/','__')
        pth = marker_dir / f"marker.{level_number}.{pth}.csv"
        if not pth.is_file():
            print(f"could not find {pth}")
            bad += 1
            to_pop.append(k)
        else:
            good += 1
    print(f"good {good} bad {bad}")


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


def main():
    tree_src_file = pathlib.Path(
       "/allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/Taxonomies/AIT17.0_mouse/Templates/cl.df.3levels.csv")
    assert tree_src_file.is_file()

    marker_dir = pathlib.Path(
        '/allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/Taxonomies/AIT17.0_mouse/Templates/marker_list_on_nodes')
    assert marker_dir.is_dir()

    tree = read_taxonomy_tree(tree_src_file)

    with open('taxonomy_tree.json', 'w') as out_file:
        out_file.write(json.dumps(tree,indent=2))

    print('=====level_1=====')
    find_marker_files(
        marker_dir=marker_dir,
        taxonomy_tree=tree,
        level='level_1',
        level_number=2)

    print('=====level_2=====')
    find_marker_files(
        marker_dir=marker_dir,
        taxonomy_tree=tree,
        level='level_2',
        level_number=3)


if __name__ == "__main__":
    main()
