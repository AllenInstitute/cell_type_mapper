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

    siletti_dir = pathlib.Path(
        "/allen/aibs/technology/danielsf/knowledge_base/siletti")
    training_dir = siletti_dir / "training"
    testing_dir = siletti_dir / "test"
    mapping_dir = testing_dir / "mappings"
    assert training_dir.is_dir()
    assert testing_dir.is_dir()
    assert mapping_dir.is_dir()

    precomputed_stats_path = training_dir / "precomputed_stats.siletti.training.h5"
    assert precomputed_stats_path.is_file()

    mapping_path = mapping_dir / "mapping.202402071400.json" 

    plot_lookup = plot_rates(
        precomputed_stats_path=precomputed_stats_path,
        mapping_path=mapping_path)

    output_path = 'figures/siletti_rates.pdf'
    with PdfPages(output_path) as pdf_handle:
        for level in plot_lookup:
            pdf_handle.savefig(plot_lookup[level]) 

    print(f'wrote {output_path}')


def plot_rates(
        precomputed_stats_path,
        mapping_path):
    """
    Return a dict mapping level to mfig.Figure
    """

    taxonomy_tree = TaxonomyTree.from_precomputed_stats(
        precomputed_stats_path)

    cluster_stats = read_precomputed_stats(
        precomputed_stats_path,
        taxonomy_tree=taxonomy_tree,
        for_marker_selection=True)

    with open(mapping_path, 'rb') as src:
        mapping = json.load(src)

    truth_df = read_df_from_h5ad(
        mapping['config']['query_path'],
        df_name='obs')

    mapping = {
        c['cell_id']: c for c in mapping['results']}

    result = dict()
    for level in taxonomy_tree.hierarchy:
        (fig, confusion_fig) = rates_per_level(
            taxonomy_tree=taxonomy_tree,
            cluster_stats=cluster_stats['cluster_stats'],
            mapping=mapping,
            truth_df=truth_df,
            level=level)

        result[level] = fig
        with PdfPages(f'figures/{level}.pdf') as dst:
            dst.savefig(confusion_fig)

    return result


def rates_per_level(
        taxonomy_tree,
        cluster_stats,
        mapping,
        truth_df,
        level,
        fontsize=20):

    node_list = taxonomy_tree.nodes_at_level(level)
    node_list.sort()
    node_to_idx = {
        n:ii for ii, n in enumerate(node_list)}

    truth_lookup = {
        cell_id: node
        for cell_id, node in zip(truth_df.index.values, truth_df[level].values)}

    true_pos = np.zeros(len(node_list), dtype=np.float32)
    false_neg = np.zeros(len(node_list), dtype=np.float32)
    false_pos = np.zeros(len(node_list), dtype=np.float32)
    n_test_cells = np.zeros(len(node_list), dtype=np.float32)
    n_assn_cells = np.zeros(len(node_list), dtype=np.float32)
    n_training_cells = np.array(
        [cluster_stats[f'{level}/{node}']['n_cells']
        for node in node_list])

    for cell_id in truth_lookup:
        truth = truth_lookup[cell_id]
        true_idx = node_to_idx[truth]
        n_test_cells[true_idx] += 1.0
        assn = mapping[cell_id][level]['assignment']
        assn_idx = node_to_idx[assn]
        n_assn_cells[assn_idx] += 1.0
        if truth == assn:
            true_pos[true_idx] += 1.0
        else:
            false_pos[assn_idx] += 1.0
            false_neg[true_idx] += 1.0

    fig = mfig.Figure(figsize=(20, 20))
    axis_list = [fig.add_subplot(2,2,ii+1) for ii in range(3)]
    tp_axis = axis_list[0]
    fp_axis = axis_list[1]
    fn_axis = axis_list[2]

    x_arr = n_test_cells
    x_label = 'n_test_cells'

    tp_axis.set_title(level, fontsize=fontsize)

    tp_axis.set_xlabel(x_label, fontsize=fontsize)
    tp_axis.set_ylabel('n_true_pos/(n_true_pos+n_false_neg)',
                       fontsize=fontsize)
    tp_axis.scatter(x_arr,
        true_pos/np.where(n_test_cells>0, n_test_cells, 1))

    fp_axis.set_xlabel(x_label, fontsize=fontsize)
    fp_axis.set_ylabel('n_false_pos/(n_true_pos+n_false_pos)',
                       fontsize=fontsize)
    fp_axis.scatter(x_arr,
        false_pos/np.where(n_assn_cells>0, n_assn_cells, 1))

    fn_axis.set_xlabel(x_label, fontsize=fontsize)
    fn_axis.set_ylabel('n_false_neg/(n_true_pos+n_false_neg)')
    denom = np.where(n_test_cells > 0, n_test_cells, 1)
    is_zero = (n_test_cells < 1)
    if is_zero.sum() > 0:
        assert false_neg[is_zero].max() < 1
    fn_axis.scatter(x_arr, false_neg/denom)

    for axis in (tp_axis, fp_axis, fn_axis):
        axis.set_xscale('log')

    n_train_max = np.argmax(n_training_cells)
    bad_name = taxonomy_tree.label_to_name(
        level=level,
        label=node_list[n_train_max],
        name_key='name')
    print(f'    max n_train {bad_name} {n_training_cells[n_train_max]}')

    if level == taxonomy_tree.hierarchy[0]:
        conf_fig = mfig.Figure(figsize=(100,100))
        fontsize = 100
    elif level == taxonomy_tree.hierarchy[1]:
        conf_fig = mfig.Figure(figsize=(200, 200))
        fontsize= 200
    else:
        conf_fig = mfig.Figure(figsize=(400, 400))
        fontsize = 400

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

    all_l = set(label_order)
    assert len(all_l) == len(label_order)
    all_t = set(true_labels)
    for t in all_t:
        if t not in all_l:
            raise RuntimeError(f"missing {t} from label order")
    all_e = set(exp_labels)
    for e in all_e:
        assert e in all_l
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

    #with PdfPages(f'figures/{level}.pdf') as dst:
    #    dst.savefig(conf_fig)

    print(f'plotted {level}')
    return fig, conf_fig

if __name__ == "__main__":
    main()
