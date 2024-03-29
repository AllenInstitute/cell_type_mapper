import json
import pathlib

from hierarchical_mapping.utils.utils import (
    get_timestamp)

from hierarchical_mapping.gene_id.gene_id_mapper import (
    GeneIdMapper)

from hierarchical_mapping.taxonomy.data_release_utils import (
    get_label_to_name)

from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)

def main():

    marker_dir = pathlib.Path(
        '/allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/Taxonomies/AIT17_knowledge/nlevel4_marker_index/marker_list_on_nodes')

    assert marker_dir.is_dir()

    data_dir = pathlib.Path(
        "/allen/programs/celltypes/workgroups/rnaseqanalysis/lydian/ABC_handoff/metadata/")
    assert data_dir.is_dir()

    cluster_membership = data_dir / "WMB-taxonomy/20230630/cluster_to_cluster_annotation_membership.csv"
    assert cluster_membership.is_file()

    cluster_annotation = data_dir / "WMB-taxonomy/20230630/cluster_annotation_term.csv"
    assert cluster_annotation.is_file()

    cell_metadata = data_dir / "WMB-10X/20230630/cell_metadata.csv"
    assert cell_metadata.is_file()

    taxonomy_tree = TaxonomyTree.from_data_release(
        cell_metadata_path=cell_metadata,
        cluster_annotation_path=cluster_annotation,
        cluster_membership_path=cluster_membership,
        hierarchy=[
        "CCN20230504_CLAS",
        "CCN20230504_SUBC",
        "CCN20230504_SUPT",
        "CCN20230504_CLUS"])

    taxonomy_tree = taxonomy_tree.drop_level("CCN20230504_SUPT")
    print(taxonomy_tree.hierarchy)

    alias_mapper = get_label_to_name(
        csv_path=cluster_membership,
        valid_term_set_labels=taxonomy_tree.hierarchy,
        name_column='cluster_annotation_term_name',
        strict_alias=True)

    level_to_idx = {n:ii+2 for ii, n in enumerate(taxonomy_tree.hierarchy)}
    print('level to idx')
    print(level_to_idx)
    good_ct = 0
    bad_ct = 0
    parent_list = taxonomy_tree.all_parents

    print('iterating over marker files')
    parent_to_path = dict()
    for parent_node in parent_list:
        if parent_node is None:
            fname = 'marker.1.root.csv'
            parent_key = 'None'
        else:
            parent_key = f'{parent_node[0]}/{parent_node[1]}'
            children = taxonomy_tree.children(parent_node[0], parent_node[1])
            if len(children) < 2:
                continue
            level_idx = level_to_idx[parent_node[0]]
            readable_name = alias_mapper[(parent_node[0], parent_node[1])]
            prefix = readable_name.split()[0]
            readable_name = readable_name.replace(f'{prefix} ','')
            munged = readable_name.replace(' ','+').replace('/','__')
            fname = f'marker.{level_idx}.{munged}.csv'
        fpath = marker_dir / fname
        if fpath.is_file():
            print(f'found {fname}')
        else:
            raise RuntimeError(f"{fname} does not exist")
        parent_to_path[parent_key] = fpath

    marker_lookup = dict()
    gene_id_mapper = GeneIdMapper.from_default()
    bad_genes = set()
    for parent_key in parent_to_path:
        fpath = parent_to_path[parent_key]
        gene_symbols = []
        with open(fpath, 'r') as src:
            src.readline()
            for line in src:
                symbol = line.strip().replace('"','')
                if ' ' in symbol and symbol.split()[1].startswith('ENS'):
                    symbol = symbol.split()[0]
                gene_symbols.append(symbol)
        gene_id = gene_id_mapper.map_gene_identifiers(
            gene_id_list=gene_symbols)
        marker_lookup[parent_key] = gene_id
        for orig, new in zip(gene_symbols, gene_id):
            if 'nonsense' in new:
                bad_genes.add(orig)
    bad_genes = list(bad_genes)
    bad_genes.sort()
    if len(bad_genes) > 0:
        msg = "Unmappable genes\n"
        for g in bad_genes:
            msg += f"{g}\n"
        raise RuntimeError(msg)

    marker_lookup['metadata'] = {
        'src': str(marker_dir.resolve().absolute()),
         'accessed_on': get_timestamp()}

    with open('marker_lookup_mouse_230626.json', 'w') as out_file:
        out_file.write(json.dumps(marker_lookup,indent=2))

    print("all done")

if __name__ == "__main__":
    main()
