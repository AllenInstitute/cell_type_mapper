from cell_type_mapper.taxonomy.taxonomy_tree import TaxonomyTree
from cell_type_mapper.diff_exp.score_utils import (
    read_precomputed_stats)
from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad)

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

    #precomputed_stats_path = training_dir / "precomputed_stats.siletti.training.h5"
    #assert precomputed_stats_path.is_file()

    #mapping_path_list = [
    #    mapping_dir / "mapping.202402071100.json",
    #    mapping_dir / "mapping.202402091130.json",
    #    mapping_dir / "mapping.202402091300.json",
    #    mapping_dir / "mapping.202402091500.json"
    #]

    bakeoff_dir = pathlib.Path(
        "/allen/aibs/technology/danielsf/knowledge_base/siletti/bakeoff")
    assert bakeoff_dir.is_dir()

    mapping_path_list = [
        bakeoff_dir / "mouse_f0.9.json",
        bakeoff_dir / "human_f0.9.json",
        bakeoff_dir / "mouse_f0.25.json",
        bakeoff_dir / "human_f0.25.json"
    ]

    plot = confidence_plots(mapping_path_list, binsize=0.01)

    output_path = f"figures/bakeoff_confidence_plot.pdf"
    with PdfPages(output_path) as pdf_handle:
        pdf_handle.savefig(plot) 

    print(f'wrote {output_path}')


def confidence_plots(
        mapping_path_list,
        binsize=0.1,
        fontsize=20):

    fig = mfig.Figure(figsize=(20,20))
    axis_list = [fig.add_subplot(2,2,ii+1) for ii in range(3)]

    color_list = ['c', 'b', 'r', 'g']

    for mapping_path, color in zip(mapping_path_list, color_list):
        one_plot(
            mapping_path=mapping_path,
            fig=fig,
            axis_list=axis_list,
            fontsize=fontsize,
            binsize=binsize,
            color=color)

    for axis in axis_list:
        axis.set_xlabel('probability', fontsize=fontsize)
        axis.set_ylabel('accuracy', fontsize=fontsize)
        axis.legend(loc=0, fontsize=fontsize)

    fig.tight_layout() 

    return fig

def one_plot(mapping_path, fig, axis_list, fontsize, binsize=0.1, color='b'): 
    with open(mapping_path, 'rb') as src:
        raw_mapping = json.load(src)

    config = raw_mapping['config']
    output_name = pathlib.Path(config['extended_result_path']).name
    markers = pathlib.Path(
        config['query_markers']['serialized_lookup']).name
    config = config['type_assignment']
    legend_label = (f"{config['bootstrap_iteration']} iter; "
                    f"{config['bootstrap_factor']:.2e}; "
                    f"{output_name}")

    taxonomy_tree = TaxonomyTree.from_precomputed_stats(
        raw_mapping['config']['precomputed_stats']['path'])

    supt = 'CCN20230722_SUPT'
    if supt in taxonomy_tree.hierarchy:
        taxonomy_tree = taxonomy_tree.drop_level(supt)

    truth_path= raw_mapping['config']['query_path']
    truth_obs = read_df_from_h5ad(truth_path, df_name='obs')

    truth = dict()
    for level in taxonomy_tree.hierarchy:
        for cell, val in zip(truth_obs.index.values, truth_obs[level].values):
            if cell not in truth:
                truth[cell] = dict()
            truth[cell][level] = val

    mapping = {c['cell_id']: c for c in raw_mapping['results']}
    del raw_mapping

    for i_level, level in enumerate(taxonomy_tree.hierarchy):
        (bins, expected, actual) =  get_rate_lookup(
            taxonomy_tree=taxonomy_tree,
            mapping=mapping,
            truth=truth,
            level=level,
            binsize=binsize)

        axis = axis_list[i_level]
        axis.set_title(level, fontsize=fontsize)
        axis.plot(bins, actual, label=legend_label, c=color)
        axis.plot(bins, expected, c=color, linestyle='--')

    return fig 


def get_rate_lookup(
        taxonomy_tree,
        mapping,
        truth,
        level,
        binsize=0.1):

    bins = np.arange(binsize, 1.0+binsize, binsize)


    true_prob = []
    false_prob = []
    all_prob = []

    for cell_id in mapping:
        is_true = (mapping[cell_id][level]['assignment']
                   == truth[cell_id][level])

        prob = 1.0
        for l in taxonomy_tree.hierarchy:
            prob *= mapping[cell_id][l]['bootstrapping_probability']
            if l == level:
                break
        if is_true:
            true_prob.append(prob)
        else:
            false_prob.append(prob)
        all_prob.append(prob)

    true_prob = np.array(true_prob)
    false_prob = np.array(false_prob)
    all_prob = np.array(all_prob)

    expected = []
    actual = []
    used_bins = []
    for b in bins:
        expected_mask = (all_prob<=b)
        this_true = (true_prob<=b).sum()
        this_false = (false_prob<=b).sum()

        if len(expected_mask) == 0:
            continue
        if this_true+this_false == 0:
            continue

        used_bins.append(b)
        expected.append(all_prob[expected_mask].sum()/expected_mask.sum())   
        denom = max(1, this_true+this_false)
        actual.append(this_true/denom)
    return np.array(used_bins), np.array(expected), np.array(actual) 


if __name__ == "__main__":
    main()
