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


    bakeoff_dir = pathlib.Path(
        "/allen/aibs/technology/danielsf/knowledge_base/siletti/bakeoff")
    assert bakeoff_dir.is_dir()

    sea_ad_dir = bakeoff_dir.parent / 'sea_ad_genes'
    assert sea_ad_dir.is_dir()

    small_marker_dir = bakeoff_dir.parent / 'small_markers'
    assert small_marker_dir.is_dir()

    frac_grid_dir = bakeoff_dir.parent / 'frac_grid'
    assert frac_grid_dir.is_dir()

    """
    mouse_path_list = [
        bakeoff_dir / "mouse_f0.9.json",
        bakeoff_dir / "mouse_f0.25.json",
    ]

    handoff_path_list = [
        bakeoff_dir / "mouse_f0.9_handoff.json",
        bakeoff_dir / "mouse_f0.25_handoff.json"
    ]

    by_hand_path_list = [
        bakeoff_dir / "mouse_f0.9_by_hand.json",
        bakeoff_dir / "mouse_f0.25_by_hand.json"
    ]

    human_path_list = [
        bakeoff_dir / "human_f0.9.json",
        bakeoff_dir / "human_f0.25.json"
    ]

    human_path_list = [
        sea_ad_dir / "human_f0.9_full_markers.json",
        sea_ad_dir / "human_f0.8_full_markers.json",
        sea_ad_dir / "human_f0.75_full_markers.json",
        sea_ad_dir / "human_f0.5_full_markers.json",
        sea_ad_dir / "human_f0.25_full_markers.json"]
    """

    sea_ad_path_list = [
        sea_ad_dir / "human_f0.9_n300.json",
        sea_ad_dir / "human_f0.8_n300.json",
        sea_ad_dir / "human_f0.75_n300.json",
        sea_ad_dir / "human_f0.5_n300.json",
        sea_ad_dir / "human_f0.25_n300.json"]

 

    handoff_path_list = [
        frac_grid_dir / "mouse_f0.9_handoff.json",
        frac_grid_dir / "mouse_f0.8_handoff.json",
        frac_grid_dir / "mouse_f0.75_handoff.json",
        frac_grid_dir / "mouse_f0.5_handoff.json",
        bakeoff_dir / "mouse_f0.25_handoff.json"]

    output_path = "bakeoff/f1_plots_handoff.pdf"
    with PdfPages(output_path) as pdf_handle:
        plot_species_comparison(
            mapping_path_list=sea_ad_path_list,
            pdf_handle=pdf_handle,
            drop_level=None,
            species='human')

        """
        plot_species_comparison(
            mapping_path_list=mouse_path_list,
            pdf_handle=pdf_handle,
            drop_level='CCN20230722_SUPT',
            species='mouse')

        plot_species_comparison(
            mapping_path_list=by_hand_path_list,
            pdf_handle=pdf_handle,
            drop_level='CCN20230722_SUPT',
            species='mouse_by_hand')

        plot_species_comparison(
            mapping_path_list=handoff_path_list,
            pdf_handle=pdf_handle,
            drop_level='CCN20230722_SUPT',
            species='mouse_handoff')
        """

def plot_species_comparison(
        mapping_path_list,
        pdf_handle,
        drop_level=None,
        fontsize=30,
        species='mouse'):
    """
    Assumes every mapping in mapping_path_list has the same taxonomy
    tree and query path
    """
    config = json.load(open(mapping_path_list[0], 'rb'))['config']
    taxonomy_tree = TaxonomyTree.from_precomputed_stats(
        config['precomputed_stats']['path'])

    if drop_level is not None:
        taxonomy_tree = taxonomy_tree.drop_level(drop_level)

    truth_path= config['query_path']
    truth_obs = read_df_from_h5ad(truth_path, df_name='obs')

    truth = dict()
    for level in taxonomy_tree.hierarchy:
        for cell, val in zip(truth_obs.index.values, truth_obs[level].values):
            if cell not in truth:
                truth[cell] = dict()
            truth[cell][level] = val

    n_col = len(mapping_path_list)
    n_row = len(taxonomy_tree.hierarchy)
    fig = mfig.Figure(
        figsize=(n_col*15, len(taxonomy_tree.hierarchy)*10))
    axis_lookup = dict()
    for i_level, level in enumerate(taxonomy_tree.hierarchy):
        axis_lookup[level] = dict()
        for i_path, path in enumerate(mapping_path_list):
            axis = fig.add_subplot(n_row, n_col, i_path+i_level*n_col+1)
            axis_lookup[level][path] = axis

    print('')
    for mapping_path in mapping_path_list:
        with open(mapping_path, 'rb') as src:
            mapping = json.load(src)
 
        config = mapping['config']
        out_name = pathlib.Path(config['extended_result_path'])
        out_name = str(out_name.relative_to(out_name.parent.parent))
        print(f'========{out_name}==========')

        mapping = {c['cell_id']: c for c in mapping['results']}

        assert set(mapping.keys()) == set(truth.keys())

        for level in taxonomy_tree.hierarchy:
            axis = axis_lookup[level][mapping_path]
            plot_f1(mapping, truth, axis, level, taxonomy_tree)
            f = config['type_assignment']['bootstrap_factor']
            axis.set_title(
                f'{level}: {out_name}', fontsize=20)

    fig.tight_layout()

    pdf_handle.savefig(fig)


def plot_f1(mapping, truth, axis, level, taxonomy_tree, fontsize=20):

    data_lookup = dict()
    for node in taxonomy_tree.nodes_at_level(level):
        data_lookup[node] = {
            'tp': 0, 'fp': 0, 'fn': 0
        }

    n_correct = 0
    n_incorrect = 0
    for cell_id in mapping:
        true_label = truth[cell_id][level]
        mapped_label = mapping[cell_id][level]['assignment']
        if true_label == mapped_label:
            data_lookup[true_label]['tp'] += 1
            n_correct += 1
        else:
            data_lookup[true_label]['fn'] += 1
            data_lookup[mapped_label]['fp'] += 1
            n_incorrect += 1

    n_cells = []
    f1 = []
    name = []
    full_tp = 0.0
    full_fn = 0.0
    full_fp =0.0
    for node in data_lookup:
        n = data_lookup[node]['fn'] + data_lookup[node]['tp']
        tp = data_lookup[node]['tp']
        fn = data_lookup[node]['fn']
        fp = data_lookup[node]['fp']
        full_tp += tp
        full_fn += fn
        full_fp += fp
        if tp ==0 and fp == 0 and fn == 0:
            print(f'skipping {level} {node}')
            continue

        ff = 2*tp/(2*tp+fp+fn)
        n_cells.append(n)
        f1.append(ff)
        name.append(
            taxonomy_tree.label_to_name(level=level, label=node))

    n_cells = np.array(n_cells)
    name = np.array(name)
    f1 = np.array(f1)
    sorted_dex = np.argsort(n_cells)
    name = name[sorted_dex]
    f1 = f1[sorted_dex]
    n_cells = n_cells[sorted_dex]
    #for ii in range(5):
    #    jj = len(n_cells)-1-ii
    #    print(name[jj], n_cells[jj], f1[jj])
    wgt = 0.0
    avg = 0.0
    flat = 0.0
    flat_n = 0.0
    for n, ff in zip(n_cells, f1):
        wgt += n
        avg += n*ff
        if n > 0:
            flat += ff
            flat_n += 1.0
    micro = 2*full_tp/(2*full_tp+full_fp+full_fn)
    print(f'{level} avg f1: {avg/wgt:.2e} -- micro {micro:.2e} '
          f'-- flat {flat/flat_n:.2e} -- corr {n_correct:.2e} incorr {n_incorrect:.2e}')
    axis.scatter(n_cells, f1)
    axis.set_xlabel('Ncells (TP + FN)', fontsize=fontsize)
    axis.set_ylabel('F1', fontsize=fontsize)


if __name__ == "__main__":
    main()
