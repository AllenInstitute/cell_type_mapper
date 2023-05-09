import matplotlib.figure as mfig
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

import anndata
import argparse
import json
import numpy as np
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


def thin_img(img):
    n_el = img.shape[0]
    to_keep = []
    for ii in range(n_el):
        if img[ii, :].sum() > 0 or img[:, ii].sum() > 0:
            to_keep.append(ii)
    to_keep = np.array(to_keep)
    img = img[to_keep, :]
    img = img[:, to_keep]
    return img


def plot_confusion_matrix(
        figure,
        axis,
        true_labels,
        experimental_labels,
        label_order,
        normalize_by='truth',
        fontsize=20,
        title=None):

    img = np.zeros((len(label_order), len(label_order)), dtype=int)
    label_to_idx = {
        l:ii for ii,l in enumerate(label_order)}

    for truth, experiment in zip(true_labels, experimental_labels):
        true_idx = label_to_idx[truth]
        experiment_idx = label_to_idx[experiment]
        img[true_idx, experiment_idx] += 1

    s0 = img.sum()
    img = thin_img(img)
    assert img.sum() == s0
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

    with np.errstate(divide='ignore'):
        valid = (img>0.0)
        min_val = np.log10(np.min(img[valid]))
        img = np.where(img>0.0, np.log10(img), min_val-2)

    display_img = axis.imshow(img)
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = figure.colorbar(display_img, ax=axis, cax=cax,
        label="log10(normalized count)")

    for s in ('top', 'right', 'left', 'bottom'):
        axis.spines[s].set_visible(False)

    axis.set_xlabel('test label', fontsize=fontsize)
    axis.set_ylabel('true label', fontsize=fontsize)
    if title is not None:
        axis.set_title(title, fontsize=fontsize)

    axis.tick_params(axis='both', which='both', size=0,
                     labelsize=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classification_path', type=str, default=None)
    parser.add_argument('--ground_truth_column', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()

    results = json.load(open(args.classification_path, 'rb'))

    results_lookup = {
        cell['cell_id']: cell for cell in results["results"]}

    tree_path = results['config']['precomputed_stats']['taxonomy_tree']
    query_path = results['config']['query_path']

    query_data = anndata.read_h5ad(query_path, backed='r')
    assert args.ground_truth_column in query_data.obs.columns

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

    obs = query_data.obs

    with PdfPages(args.output_path, 'w') as pdf:
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
                                         obs[args.ground_truth_column].values):
                these_experiments.append(
                    results_lookup[cell_id][level]['assignment'])
                these_truth.append(
                    inverted_tree[level][f"cl.{ground_truth}"])

            fig = mfig.Figure(figsize=(20, 10), dpi=500)
            axis_list = [fig.add_subplot(1,2,ii+1) for ii in range(2)]
            plot_confusion_matrix(
                figure=fig,
                axis=axis_list[0],
                true_labels=these_truth,
                experimental_labels=these_experiments,
                label_order=label_order,
                normalize_by='truth',
                fontsize=20,
                title=f"{level} normalized by true label "
               "(thinned)")

            plot_confusion_matrix(
                figure=fig,
                axis=axis_list[1],
                true_labels=these_truth,
                experimental_labels=these_experiments,
                label_order=label_order,
                normalize_by='experiment',
                fontsize=20,
                title=f"{level} normalized by test label "
               "(thinned)")

 

            fig.tight_layout()
            pdf.savefig(fig)


if __name__ == "__main__":
    main()
