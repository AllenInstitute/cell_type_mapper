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

    precomputed_stats_path = training_dir / "precomputed_stats.siletti.training.h5"
    assert precomputed_stats_path.is_file()

    mapping_path_list = [
        mapping_dir / "mapping.202402071400.json" ,
        mapping_dir / "mapping.202402071100.json",
        mapping_dir / "mapping.202402091130.json",
        mapping_dir / "mapping.202402091200.json"
    ]

    plot = confidence_plots(mapping_path_list, binsize=0.05)

    output_path = f"figures/confidence_plot.pdf"
    with PdfPages(output_path) as pdf_handle:
        pdf_handle.savefig(plot) 

    print(f'wrote {output_path}')


def confidence_plots(
        mapping_path_list,
        binsize=0.1,
        fontsize=20):

    fig = mfig.Figure(figsize=(20,20))
    axis_list = [fig.add_subplot(2,2,ii+1) for ii in range(3)]

    for mapping_path in mapping_path_list:
        one_plot(
            mapping_path=mapping_path,
            fig=fig,
            axis_list=axis_list,
            fontsize=fontsize,
            binsize=binsize)

    for axis in axis_list:
        axis.set_xlabel('probability', fontsize=fontsize)
        axis.set_ylabel('accuracy', fontsize=fontsize)
        axis.plot([0.0, 1.0], [0.0, 1.0], color='r', linestyle='--')
        axis.legend(loc=0, fontsize=fontsize)

    fig.tight_layout() 

    return fig

def one_plot(mapping_path, fig, axis_list, fontsize, binsize=0.1): 
    with open(mapping_path, 'rb') as src:
        raw_mapping = json.load(src)

    config = raw_mapping['config']
    config = config['type_assignment']
    legend_label = (f"{config['bootstrap_iteration']} iter; "
                    f"{config['bootstrap_factor']:.2e}")

    taxonomy_tree = TaxonomyTree.from_precomputed_stats(
        raw_mapping['config']['precomputed_stats']['path'])

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
        (bins, rate) =  get_rate_lookup(
            taxonomy_tree=taxonomy_tree,
            mapping=mapping,
            truth=truth,
            level=level,
            binsize=binsize)

        axis = axis_list[i_level]
        axis.set_title(level, fontsize=fontsize)
        axis.plot(bins, rate, label=legend_label)


    return fig 


def get_rate_lookup(
        taxonomy_tree,
        mapping,
        truth,
        level,
        binsize=0.1):

    bins = np.arange(binsize, 1.0+binsize, binsize)

    true_assn = np.zeros(bins.shape, dtype=float)
    false_assn = np.zeros(bins.shape, dtype=float)

    for cell_id in mapping:
        is_true = (mapping[cell_id][level]['assignment']
                   == truth[cell_id][level])

        prob = 1.0
        for l in taxonomy_tree.hierarchy:
            prob *= mapping[cell_id][l]['bootstrapping_probability']
            if l == level:
                break
        prob_idx = np.searchsorted(bins, prob)
        if is_true:
            true_assn[prob_idx] += 1
        else:
            false_assn[prob_idx] += 1

    denom = np.where(true_assn+false_assn >0, true_assn+false_assn, 1)
    rate = true_assn.astype(float)/denom
    valid = (true_assn+false_assn > 0)
    bins = bins[valid]
    rate = rate[valid]
    return bins, rate
    

if __name__ == "__main__":
    main()
