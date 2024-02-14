from cell_type_mapper.taxonomy.taxonomy_tree import TaxonomyTree
from cell_type_mapper.diff_exp.score_utils import (
    read_precomputed_stats)
from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad)
from cell_type_mapper.visualization.confusion_matrix import (
    plot_confusion_matrix)

import copy
import json
import numpy as np
import pandas as pd
import pathlib

import matplotlib.figure as mfig
from matplotlib.backends.backend_pdf import PdfPages

def main():

    bakeoff_dir = pathlib.Path(
        "/allen/aibs/technology/danielsf/knowledge_base/siletti/bakeoff")
    assert bakeoff_dir.is_dir()

    mapping_path_list = [
        bakeoff_dir / "human_f0.25.json",
        bakeoff_dir / "mouse_f0.25.json",
    ]

    for mapping_path in mapping_path_list:
        plot_confusion(mapping_path)


def plot_confusion(
        mapping_path):
    """
    Return a dict mapping level to mfig.Figure
    """
    if 'human' in mapping_path.name:
        species = 'human'
    else:
        species = 'mouse'
    with open(mapping_path, 'rb') as src:
        mapping = json.load(src)

    config = mapping['config']
    precomputed_stats_path = config['precomputed_stats']['path']

    taxonomy_tree = TaxonomyTree.from_precomputed_stats(
        precomputed_stats_path)


    truth_df = read_df_from_h5ad(
        mapping['config']['query_path'],
        df_name='obs')

    mapping = {
        c['cell_id']: c for c in mapping['results']}

    result = dict()
    confusion_fig = get_confusion_fig(
        taxonomy_tree=taxonomy_tree,
        mapping=mapping,
        truth_df=truth_df,
        level=taxonomy_tree.hierarchy[1],
        species=species)

    with PdfPages(f'bakeoff/{species}_{taxonomy_tree.hierarchy[1]}.pdf') as dst:
        dst.savefig(confusion_fig)

    return result


def get_confusion_fig(
        taxonomy_tree,
        mapping,
        truth_df,
        level,
        fontsize=20,
        species='mouse'):

    node_list = taxonomy_tree.nodes_at_level(level)
    node_list.sort()
    node_to_idx = {
        n:ii for ii, n in enumerate(node_list)}

    truth_lookup = {
        cell_id: node
        for cell_id, node in zip(truth_df.index.values, truth_df[level].values)}

    conf_fig = mfig.Figure(figsize=(200, 200))
    fontsize= 200

    conf_axis = conf_fig.add_subplot(1,1,1)
    cell_id_list = list(truth_lookup.keys())
    true_labels = [
        taxonomy_tree.label_to_name(
            level=level,
            label=truth_lookup[cell_id])
        for cell_id in cell_id_list
    ]
    exp_labels = [
        taxonomy_tree.label_to_name(
            level=level,
            label=mapping[cell_id][level]['assignment'])
        for cell_id in cell_id_list
    ]

    if species == 'human':
        membership_df = pd.read_csv(
            "/allen/programs/celltypes/workgroups/rnaseqanalysis/lydian/ABC_handoff/metadata/WHB-taxonomy/20240330/cluster_annotation_term.csv")
        term_to_set_order = {
            t:int(o) for t,o in zip(membership_df['label'].values, membership_df.term_order.values)}

        label_to_int = dict()
        for l in taxonomy_tree.hierarchy:
            this = dict()
            for node in taxonomy_tree.nodes_at_level(l):
                this[node] = term_to_set_order[node]
            label_to_int[l] = this

        label_names = []
        label_int = []

        reverse_h = copy.deepcopy(taxonomy_tree.hierarchy)
        reverse_h.reverse()
        level_to_prefix = dict()
        prefix = 1
        for h in reverse_h:
            level_to_prefix[h] = prefix
            prefix *= len(taxonomy_tree.all_leaves)

        for node in taxonomy_tree.nodes_at_level(level):
            parentage = taxonomy_tree.parents(level=level, node=node)
            val = 0
            for p in parentage:
                val += level_to_prefix[p]*label_to_int[p][parentage[p]]
            val += label_to_int[level][node]
            name = taxonomy_tree.label_to_name(
                level=level, label=node)
            label_int.append(val)
            label_names.append(name)

        label_int = np.array(label_int)
        label_names = np.array(label_names)
        sorted_dex = np.argsort(label_int)
        label_order = label_names[sorted_dex]

        assert len(label_order) == len(taxonomy_tree.nodes_at_level(level))
    else:
        alias_list = []
        name_list = []
        for leaf in taxonomy_tree.all_leaves:
            alias = int(taxonomy_tree.label_to_name(
                level=taxonomy_tree.leaf_level,
                label=leaf,
                name_key='alias'))
            parentage = taxonomy_tree.parents(
                level=taxonomy_tree.leaf_level,
                node=leaf)
            this = parentage[level]
            name = taxonomy_tree.label_to_name(
                level=level, label=this)
            if name not in name_list:
                name_list.append(name)
                alias_list.append(alias)
        alias_list = np.array(alias_list)
        name_list = np.array(name_list)
        sorted_dex = np.argsort(alias_list)
        label_order = name_list[sorted_dex]

    plot_confusion_matrix(
        figure=conf_fig,
        axis=conf_axis,
        true_labels=true_labels,
        experimental_labels=exp_labels,
        label_order=label_order,
        normalize_by=None,
        is_log=True,
        label_x_axis=True,
        label_y_axis=True,
        fontsize=fontsize)

    return conf_fig

if __name__ == "__main__":
    main()
