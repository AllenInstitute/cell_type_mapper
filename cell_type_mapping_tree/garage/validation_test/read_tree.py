# script to construct the taxonomy tree from the cl.df.3levels.csv
# file specified for the test

import json
import pathlib
from hierarchical_mapping.utils.taxonomy_utils import (
    validate_taxonomy_tree)

def main():
    src_file = pathlib.Path(
       "/allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/Taxonomies/AIT17.0_mouse/Templates/cl.df.3levels.csv")
    assert src_file.is_file()

    tree = dict()
    tree['hierarchy'] = ['level_1', 'level_2', 'cluster']
    for level in tree['hierarchy']:
        tree[level] = dict()
    with open(src_file, 'r') as in_file:
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
    with open('taxonomy_tree.json', 'w') as out_file:
        out_file.write(json.dumps(tree,indent=2))

if __name__ == "__main__":
    main()
