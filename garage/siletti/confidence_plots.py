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

    output_path = "bakeoff/confidence_distribution_handoff.pdf"
    with PdfPages(output_path) as pdf_handle:
        plot_species_comparison(
            mapping_path_list=human_path_list,
            pdf_handle=pdf_handle,
            drop_level=None,
            species='human')

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

    n_col = 4
    fig = mfig.Figure(
        figsize=(n_col*15, len(taxonomy_tree.hierarchy)*10))
    axis_lookup = dict()
    for i_level, level in enumerate(taxonomy_tree.hierarchy):
        axis_lookup[level] = dict()
        for i_k, k in enumerate(('cdf', 'pdf', 'pdf_rate', 'roc')):
            axis_lookup[level][k] = fig.add_subplot(
                len(taxonomy_tree.hierarchy), n_col,
                1+i_level*n_col + i_k)

    for mapping_path, color in zip(mapping_path_list, ('r', 'b')):
        mapping = json.load(open(mapping_path,'rb'))
        this_config = mapping['config']
        assert this_config['query_path'] == config['query_path']
        assert this_config[
            'precomputed_stats']['path'] == config['precomputed_stats']['path']
        mapping = {c['cell_id']: c for c in mapping['results']}

        legend_label = (
            f'fraction: {this_config["type_assignment"]["bootstrap_factor"]:.2e}'
        )

        for i_level, level in enumerate(taxonomy_tree.hierarchy):
            axis = axis_lookup[level]['cdf']
            plot_cdf_comparison(
                axis=axis,
                taxonomy_tree=taxonomy_tree,
                mapping=mapping,
                truth=truth,
                level=level,
                binsize=0.01,
                color=color,
                legend_label=legend_label)

            ct_axis = axis_lookup[level]['pdf']
            rate_axis = axis_lookup[level]['pdf_rate']
            roc_axis = axis_lookup[level]['roc']
            plot_pdf_comparison(
                ct_axis=ct_axis,
                rate_axis=rate_axis,
                roc_axis=roc_axis,
                taxonomy_tree=taxonomy_tree,
                mapping=mapping,
                truth=truth,
                level=level,
                binsize=0.02,
                color=color,
                legend_label=legend_label)


    for level in taxonomy_tree.hierarchy:
        axis = axis_lookup[level]['cdf']
        if level == taxonomy_tree.hierarchy[0]:
            title = f'{species}: {level}'
        else:
            title = level
        axis.set_title(title, fontsize=fontsize)
        axis.set_xlabel('bootstrapping probability', fontsize=fontsize)
        axis.set_ylabel('cumulative distribution', fontsize=fontsize)
        axis.legend(loc=0, fontsize=fontsize)
        axis.tick_params(which='both', axis='both', labelsize=fontsize)

        axis = axis_lookup[level]['pdf']
        axis.set_xlabel('bootstrapping probability', fontsize=fontsize)
        axis.set_ylabel('number correct', fontsize=fontsize)
        axis.legend(loc=0, fontsize=fontsize)
        axis.tick_params(which='both', axis='both', labelsize=fontsize)

        axis = axis_lookup[level]['pdf_rate']
        axis.set_xlabel('bootstrapping probability', fontsize=fontsize)
        axis.set_ylabel('fraction correct', fontsize=fontsize)
        axis.legend(loc=0, fontsize=fontsize)
        axis.tick_params(which='both', axis='both', labelsize=fontsize)

        axis = axis_lookup[level]['roc']
        axis.set_xlabel('N false labels', fontsize=fontsize)
        axis.set_ylabel('N true labels', fontsize=fontsize)
        axis.legend(loc=0, fontsize=fontsize)
        axis.tick_params(which='both', axis='both', labelsize=fontsize)

    fig.tight_layout()

    pdf_handle.savefig(fig)


def plot_cdf_comparison(
        axis,
        taxonomy_tree,
        mapping,
        truth,
        level,
        binsize=0.1,
        color='b',
        legend_label=None):

    (bins,
     expected,
     actual) = get_rate_lookup_cdf(
        taxonomy_tree=taxonomy_tree,
        mapping=mapping,
        truth=truth,
        level=level,
        binsize=binsize)

    axis.plot(bins, actual, label=legend_label, c=color, linewidth=5)
    axis.plot(bins, expected, c=color, linestyle='--', linewidth=5)


def plot_pdf_comparison(
        ct_axis,
        rate_axis,
        roc_axis,
        taxonomy_tree,
        mapping,
        truth,
        level,
        binsize=0.1,
        color='b',
        legend_label=None):

    (bins,
     expected,
     actual,
     total,
     n_true_pos,
     n_false_pos) = get_rate_lookup_pdf(
        taxonomy_tree=taxonomy_tree,
        mapping=mapping,
        truth=truth,
        level=level,
        binsize=binsize)

    ct_axis.stairs(actual, bins, label=legend_label, color=color, linewidth=5)
    x_bins = 0.5*(bins[1:]+bins[:-1])
    ct_axis.plot(x_bins, expected, c=color, linestyle='--', linewidth=5)
    ct_axis.set_yscale('log')

    total[total<1] = 1
    rate_axis.stairs(actual/total, bins,
                     label=legend_label, color=color, linewidth=5)
    if color == 'b':
        rate_axis.plot(
             x_bins,
             expected/total,
             c='c',
             linestyle='--',
             label='expected',
             linewidth=5)

    sorted_dex = np.argsort(n_false_pos)
    if color == 'r':
        linestyle = '--'
        zorder = 0
        marker = 'x'
        markersize = 20
    else:
        linestyle = '-'
        zorder = 1
        marker = None
        markersize = None
    roc_axis.plot(n_false_pos[sorted_dex],
                  n_true_pos[sorted_dex],
                  color=color,
                  label=legend_label,
                  linewidth=5,
                  linestyle=linestyle,
                  zorder=zorder,
                  marker=marker,
                  markersize=markersize)


def get_rate_lookup_cdf(
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



def get_rate_lookup_pdf(
        taxonomy_tree,
        mapping,
        truth,
        level,
        binsize=0.1):

    bins = np.arange(0.0, 1.0+binsize, binsize)

    true_assn = np.zeros(bins.shape[0]-1, dtype=float)
    total = np.zeros(bins.shape[0]-1, dtype=float)

    n_cells = len(mapping)
    true_pos = np.zeros(n_cells, dtype=int)
    false_pos = np.zeros(n_cells, dtype=int)
    all_prob = np.zeros(n_cells, dtype=float)

    assert len(truth) == len(mapping)
    for i_cell, cell_id in enumerate(mapping):
        is_true = (mapping[cell_id][level]['assignment']
                   == truth[cell_id][level])

        prob = 1.0
        for l in taxonomy_tree.hierarchy:
            prob *= mapping[cell_id][l]['bootstrapping_probability']
            if l == level:
                break

        prob_idx = np.floor(prob/binsize).astype(int)-1
        if prob_idx < 0:
            prob_idx = 0

        if is_true:
            true_assn[prob_idx] += 1
            true_pos[i_cell] = 1
        else:
            false_pos[i_cell] = 1
        all_prob[i_cell] = prob
        total[prob_idx] += 1


    unq_prob = np.unique(all_prob)
    final_tp = []
    final_fp = []
    ct = 0
    for val in unq_prob[::-1]:
        idx = np.where(all_prob == val)
        ct += len(idx[0])
        final_tp.append(true_pos[idx].sum())
        final_fp.append(false_pos[idx].sum())
    assert ct == len(all_prob)
    true_pos = np.cumsum(np.array(final_tp))
    false_pos = np.cumsum(np.array(final_fp))

    expected = np.zeros(total.shape, dtype=float)
    for ii in range(len(expected)):
        v = 0.5*(bins[ii]+bins[ii+1])
        expected[ii] = v*total[ii]

    return (bins,
            expected,
            true_assn,
            total,
            true_pos,
            false_pos)



if __name__ == "__main__":
    main()
