import h5py
import json
import numpy as np
import pathlib

import pandas as pd

from cell_type_mapper.marker_lookup.marker_lookup import (
    map_aibs_gene_names)
from cell_type_mapper.taxonomy.taxonomy_tree import TaxonomyTree

def main():
    data_dir = pathlib.Path('/allen/programs/celltypes/workgroups/rnaseqanalysis/lydian/ABC_handoff/metadata/WMB-taxonomy/20230830')
    assert data_dir.is_dir()

    mean_path = pathlib.Path(
        "/allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/Taxonomies/AIT21.0.root_mouse/cl.means.v9_locked.csv")
    assert mean_path.is_file()

    tree = TaxonomyTree.from_data_release(
        cell_metadata_path=None,
        cluster_annotation_path=data_dir/"cluster_annotation_term.csv",
        cluster_membership_path=data_dir/"cluster_to_cluster_annotation_membership.csv",
        hierarchy=["CCN20230722_CLAS", "CCN20230722_SUBC", "CCN20230722_SUPT", "CCN20230722_CLUS"])

    mean_df = pd.read_csv(mean_path)

    aibs_gene_names= list(mean_df['Unnamed: 0'].values)
    gene_name_map = map_aibs_gene_names(aibs_gene_names)
    gene_names = [gene_name_map[g] for g in aibs_gene_names]

    alias_to_label = dict()
    for leaf in tree.all_leaves:
        alias = tree.label_to_name(
                    level=tree.leaf_level,
                    label=leaf,
                    name_key='alias')
        alias_to_label[alias] = leaf

    data = np.zeros((len(mean_df.columns)-1, len(mean_df)), dtype=float)
    cluster_to_row = dict()
    for i_col, col in enumerate(mean_df.columns[1:]):
        data[i_col, :] = mean_df[col].values
        label = alias_to_label[col]
        assert label not in cluster_to_row
        cluster_to_row[label] = i_col

    out_path = '/allen/aibs/technology/danielsf/knowledge_base/scratch/abc_revision_230830/abc_stats_230807.h5'
    with h5py.File(out_path, 'w') as dst:
        dst.create_dataset('sum', data=data)
        dst.create_dataset('n_cells', data=np.ones(data.shape[0], dtype=int))
        dst.create_dataset(
            'cluster_to_row', data=json.dumps(cluster_to_row).encode('utf-8'))
        dst.create_dataset(
            'col_names', data=json.dumps(gene_names).encode('utf-8'))
        dst.create_dataset('taxonomy_tree',
            data=tree.to_str().encode('utf-8'))

if __name__ == "__main__":
    main()
