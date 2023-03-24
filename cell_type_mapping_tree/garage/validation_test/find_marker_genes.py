import json
import pathlib

def find_marker_files(
        marker_dir,
        taxonomy_tree,
        level,
        level_number):
    good = 0
    bad = 0
    for k in taxonomy_tree[level]:
        pth = k.replace(' ', '+').replace('/','__')
        pth = marker_dir / f"marker.{level_number}.{pth}.csv"
        if not pth.is_file():
            print(f"could not find {pth}")
            bad += 1
        else:
            good += 1
    print(f"good {good} bad {bad}")



def main():
    taxonomy_tree = json.load(open('taxonomy_tree.json', 'rb'))

    marker_dir = pathlib.Path(
        '/allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/Taxonomies/AIT17.0_mouse/Templates/marker_list_on_nodes')
    assert marker_dir.is_dir()

    find_marker_files(
        marker_dir=marker_dir,
        taxonomy_tree=taxonomy_tree,
        level='level_1',
        level_number=2)

if __name__ == "__main__":
    main()
