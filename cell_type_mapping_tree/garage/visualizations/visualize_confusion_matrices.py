import matplotlib.figure as mfig
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

import anndata
import argparse
import json
import numpy as np
import re

from hieararchical_mapping.taxonomy.taxonomy_tree import (
    taxonomy_tree)


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


def plot_confusion_matrix(
        figure,
        axis,
        true_labels,
        experimental_labels,
        label_order,
        normalize_by='truth',
        fontsize=20,
        title=None):

    img = np.zeros((len(label_order), len(label_order)), dtype=float)
    label_to_idx = {
        l:ii for ii,l in enumerate(label_order)}

    for truth, experiment in zip(true_labels, experimental_labels):
        true_idx = label_to_idx[truth]
        experiment_idx = label_to_idx[experiment]
        img[true_idx, experiment_idx] += 1

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

    axis = axis.imshow(img)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(img, ax=ax, cax=cax)

    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(False)

    ax.set_xlabel('found label', fontsize=fontsize)
    ax.set_ylabel('true label', fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classification_path', type=str, default=None)
    parser.add_argument('--ground_truth_column', type=str, default=None)
    parser.add_argumetn('--output_path', type=str, default=None)
    args = parser.parse_args()

    results = json.load(open(args.classification_path, 'rb'))

    result_lookup = {
        cell['cell_id']: cell for cell in results["results"]}

    tree_path = results['config']['precomputed_stats']['taxonomy_tree']
    query_path = results['query_path']

    query_data = anndata.read_h5ad(query_path, backed='r')
    assert args.ground_truth_column in query_data.obs.columns

    (taxonomy_tree,
     inverted_tree) = invert_tree(tree_path)

    leaf_list = taxonomy_tree.nodes_at_level(taxonomy_tree.leaf)
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
            if level == taxonomy_tree.leaf:
                label_order = leaf_order
            else:
                label_order = []
                for leaf in label_order:
                    this = inverse_tree[level][leaf]
                    if this not in label_order:
                        label_order.append(this)


        these_experiments = []
        these_truth = []
        for cell_id, ground_truth in zip(obs.index.values,
                                         obs[args.ground_truth_column].values):
            these_experiments.append(
                results_lookup[cell_id][level]['assignment'])
            these_truth.append(
                inverse_tree[level][ground_truth])

        fig = mfig.Figure(figsize=(10, 10), dpi=500)
        axis = fig.add_subplot(1,1,1)
        plot_confusion_matrix(
            figure=fig,
            axis=axis,
            true_labels=these_truth,
            experimental_labels=these_experiments,
            label_order=label_order,
            normalize_by='truth',
            fontsize=20,
            title=f"{level} normalized by true label")

        fig.tight_layout()
        pdf.savefig(fig)


if __name__ == "__main__":
    main()
