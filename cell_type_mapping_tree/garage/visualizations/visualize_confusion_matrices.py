import matplotlib.figure as mfig
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

from anndata._io.specs import read_elem
import argparse
import h5py
import json
import numpy as np
import pathlib
import re

from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)


def invert_tree(taxonomy_tree_path):
    tree = TaxonomyTree.from_json_file(taxonomy_tree_path)
    as_leaves = tree.as_leaves
    inverse_lookup = dict()
    for level in as_leaves:
        inverse_lookup[level] = dict()
        for parent in as_leaves[level]:
            for leaf in as_leaves[level][parent]:
                inverse_lookup[level][leaf] = parent
    return tree, inverse_lookup


def thin_img(img, label_list):
    n_el = img.shape[0]
    to_keep = []
    for ii in range(n_el):
        if img[ii, :].sum() > 0 or img[:, ii].sum() > 0:
            to_keep.append(ii)
    to_keep = np.array(to_keep)
    img = img[to_keep, :]
    img = img[:, to_keep]
    return img, np.array(label_list)[to_keep]


def plot_confusion_matrix(
        figure,
        axis,
        true_labels,
        experimental_labels,
        label_order,
        normalize_by='truth',
        fontsize=20,
        title=None,
        is_log=False):

    img = np.zeros((len(label_order), len(label_order)), dtype=int)
    label_to_idx = {
        l:ii for ii,l in enumerate(label_order)}

    for truth, experiment in zip(true_labels, experimental_labels):
        true_idx = label_to_idx[truth]
        experiment_idx = label_to_idx[experiment]
        img[true_idx, experiment_idx] += 1

    s0 = img.sum()
    img, thinned_labels = thin_img(img, label_list=label_order)
    assert img.sum() == s0

    img = np.ma.masked_array(
        img, mask=(img==0))

    img = img.astype(float)

    if normalize_by == 'truth':
        for ii in range(img.shape[0]):
            denom = img[ii, :].sum()
            img[ii, :] /= max(1, denom)
    elif normalize_by == 'experiment':
        for ii in range(img.shape[1]):
            denom = img[:, ii].sum()
            img[:, ii] /= max(1, denom)
    else:
        raise RuntimeError(
            f"normalize_by {normalize_by} makes no sense")

    if is_log:
        cax_title = 'log10(normalized count)'
        with np.errstate(divide='ignore'):
            valid = (img>0.0)
            min_val = np.log10(np.min(img[valid]))
            img = np.where(img>0.0, np.log10(img), min_val-2)
    else:
        cax_title = 'normalized count'

    display_img = axis.imshow(img, cmap='cool')
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = figure.colorbar(display_img, ax=axis, cax=cax,
        label=cax_title)

    for s in ('top', 'right', 'left', 'bottom'):
        axis.spines[s].set_visible(False)

    axis.set_xlabel('mapped label', fontsize=fontsize)
    axis.set_ylabel('true label', fontsize=fontsize)
    if title is not None:
        axis.set_title(title, fontsize=fontsize)

    tick_values = [ii for ii in range(len(thinned_labels))]
    axis.set_xticks(tick_values)
    axis.set_xticklabels(thinned_labels, fontsize=7, rotation='vertical')
    axis.set_yticks(tick_values)
    axis.set_yticklabels(thinned_labels, fontsize=7, rotation='horizontal')



def summary_plots(
        classification_path,
        ground_truth_column_list,
        pdf_handle,
        is_log10):

    classification_path = pathlib.Path(classification_path)
    print(classification_path.name)
    results = json.load(open(classification_path, 'rb'))

    results_lookup = {
        cell['cell_id']: cell for cell in results["results"]}

    tree_path = results['config']['precomputed_stats']['taxonomy_tree']
    query_path = pathlib.Path(results['config']['query_path'])

    with h5py.File(query_path, 'r') as src:
        query_obs = read_elem(src['obs'])
    for g in ground_truth_column_list:
        if g in query_obs.columns:
            ground_truth_column = g
            break
    print(f"using ground truth {ground_truth_column}")
    assert ground_truth_column in query_obs.columns

    (taxonomy_tree,
     inverted_tree) = invert_tree(tree_path)

    leaf_list = taxonomy_tree.all_leaves
    leaf_names = []
    leaf_idx = []
    int_pattern = re.compile('[0-9]+')
    for n in leaf_list:
        leaf_names.append(n)
        ii = int(int_pattern.findall(n)[0])
        assert ii not in leaf_idx
        leaf_idx.append(ii)
    leaf_idx = np.array(leaf_idx)
    leaf_names = np.array(leaf_names)
    sorted_dex = np.argsort(leaf_idx)
    leaf_order = leaf_names[sorted_dex]

    obs = query_obs

    for level in taxonomy_tree.hierarchy:
        if level == taxonomy_tree.leaf_level:
            label_order = leaf_order
        else:
            label_order = []
            for leaf in leaf_order:
                this = inverted_tree[level][leaf]
                if this not in label_order:
                    label_order.append(this)

        these_experiments = []
        these_truth = []
        for cell_id, ground_truth in zip(obs.index.values,
                                     obs[ground_truth_column].values):
            gt_key = f"cl.{ground_truth}"
            if gt_key in inverted_tree[level]:
                these_experiments.append(
                    results_lookup[cell_id][level]['assignment'])
                these_truth.append(
                    inverted_tree[level][f"cl.{ground_truth}"])

        fig = mfig.Figure(figsize=(25, 10), dpi=300)
        grid = gridspec.GridSpec(nrows=20, ncols=60)
        grid.update(bottom=0.1, top=0.99, left=0.01, right=0.99,
            wspace=0.1, hspace=0.01)
        axis_list = [
            fig.add_subplot(grid[0:20, 0:10]),
            fig.add_subplot(grid[0:20, 14:34]),
            fig.add_subplot(grid[0:20, 38:58])]

        good = 0
        bad = 0
        for truth, experiment in zip(these_truth, these_experiments):
            if truth == experiment:
                good += 1
            else:
                bad += 1

        #msg = f"{classification_path.name}\n"
        summary_name = f"{query_path.parent.name}/{query_path.name}"
        msg = f"{summary_name}\n"
        msg += f"{level}\n"
        msg += f"=========\n"
        msg += f"correctly mapped: {good}\n"
        msg += f"incorrectly mapped: {bad}\n"
        msg += f"fraction correct: {good/float(good+bad):.3e}"

        axis_list[0].text(
            5,
            50,
            msg,
            fontsize=15)
        axis_list[0].set_xlim((0, 100))
        axis_list[0].set_ylim((0, 100))
        for s in ('top', 'left', 'bottom', 'right'):
            axis_list[0].spines[s].set_visible(False)
        axis_list[0].tick_params(
            axis='both', which='both', size=0, labelsize=0)

        plot_confusion_matrix(
            figure=fig,
            axis=axis_list[1],
            true_labels=these_truth,
            experimental_labels=these_experiments,
            label_order=label_order,
            normalize_by='truth',
            fontsize=20,
            title=f"{level} normalized by true label",
            is_log=is_log10)

        plot_confusion_matrix(
            figure=fig,
            axis=axis_list[2],
            true_labels=these_truth,
            experimental_labels=these_experiments,
            label_order=label_order,
            normalize_by='experiment',
            fontsize=20,
            title=f"{level} normalized by mapped label",
            is_log=is_log10)

        pdf_handle.savefig(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classification_dir', type=str, default=None)
    parser.add_argument('--ground_truth_column', type=str, default=None, nargs='+')
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--log10', default=False, action='store_true')
    args = parser.parse_args()

    input_dir = pathlib.Path(args.classification_dir)
    input_path_list = [n for n in input_dir.iterdir() if n.name.endswith('result.json')]
    input_path_list.sort()

    if not isinstance(args.ground_truth_column, list):
        ground_truth_column_list = [args.ground_truth_column]
    else:
        ground_truth_column_list = args.ground_truth_column

    with PdfPages(args.output_path) as pdf_handle:
        for pth in input_path_list:
            if 'in_platform' in pth.name:
                ground_truth_column = 'cl'
            else:
                ground_truth_column = 'gt_cl'
            summary_plots(
                classification_path=pth,
                ground_truth_column_list=ground_truth_column_list,
                pdf_handle=pdf_handle,
                is_log10=args.log10)

if __name__ == "__main__":
    main()
