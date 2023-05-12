import json
import pathlib

from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)


def main():

    out_path = (
        "/allen/aibs/technology/danielsf/knowledge_base/"
        "benchmarking/mouse_cookbook/markers/markers.json")

    taxonomy_path = (
       "/allen/aibs/technology/danielsf/knowledge_base/"
       "benchmarking/mouse/taxonomy/taxonomy_tree_cross_platform.json")

    taxonomy_tree = TaxonomyTree.from_json_file(taxonomy_path)

    marker_dir = pathlib.Path(
        "/allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/"
        "Taxonomies/AIT17.0_mouse/Templates/marker_list_on_nodes")
    character_map = {' ':'+', '/':'__'}
    marker_lookup = dict()
    root_path = marker_dir / 'marker.1.root.csv'
    with open(root_path, 'r') as in_file:
        these = []
        in_file.readline()
        for line in in_file:
            these.append(line.strip().replace('"', ''))
        marker_lookup['None'] = these

    for parent in taxonomy_tree.all_parents:
        if parent is None:
            continue
        if parent[0] == 'level_1':
            level = 2
        elif parent[0] == 'level_2':
            level = 3
        name = parent[1]
        for orig in character_map:
            name = name.replace(orig, character_map[orig])
        marker_path = marker_dir / f"marker.{level}.{name}.csv"
        these = []
        print(f"reading {marker_path}")
        if marker_path.is_file():
            with open(marker_path, 'r') as in_file:
                in_file.readline()
                for line in in_file:
                    these.append(line.strip().replace('"', ''))
        marker_lookup[f'{parent[0]}/{parent[1]}'] = these

    with open(out_path, 'w') as out_file:
        out_file.write(json.dumps(marker_lookup,indent=2))

if __name__ == "__main__":
    main()
