import anndata
import argparse
import json
import matplotlib.figure as mfig
import numpy as np
import pathlib

from hierarchical_mapping.utils.taxonomy_utils import (
    convert_tree_to_leaves)

# read the taxonomy in from the json file you saved
# see how well it does at level_1 and level_2

def invert_tree(taxonomy_tree):
    tree_to_leaves = convert_tree_to_leaves(taxonomy_tree)
    inverted_tree = dict()
    for level in taxonomy_tree["hierarchy"][:-1]:
        this = dict()
        for node in tree_to_leaves[level]:
            for child in tree_to_leaves[level][node]:
                assert child not in this
                this[child] = node
        inverted_tree[level] = this
    return inverted_tree


def assess_results(
        cell_to_truth,
        cell_to_assignment,
        hierarchy_level,
        taxonomy_tree,
        inverted_tree,
        axis_list):
    #fig_path = f"figs/{fig_prefix}_{hierarchy_level}.png"

    hierarchy = taxonomy_tree['hierarchy']
    #nrows = len(hierarchy)
    #ncols = 1
    #fig = mfig.Figure(figsize=(ncols*10, nrows*10))
    #axis_list = [fig.add_subplot(nrows, ncols, ii+1)
    #             for ii in range(nrows*ncols)]

    good_cells = []
    bad_cells = []
    for cell in cell_to_assignment:
        true_assignment = cell_to_truth[cell['cell_id']]
        if hierarchy_level != taxonomy_tree['hierarchy'][-1]:
            true_assignment = inverted_tree[hierarchy_level][true_assignment]
        assignment = cell[hierarchy_level]['assignment']
        if assignment == true_assignment:
            good_cells.append(cell)
        else:
            bad_cells.append(cell)

    print(f"{hierarchy_level} {len(good_cells):.2e} good {len(bad_cells):.2e} bad")
    n_good = len(good_cells)
    n_bad = len(bad_cells)

    nbins = 100
    fontsize = 20
    for i_level, level in enumerate([hierarchy_level]):
        axis = axis_list[i_level]
        good_confidence = np.array([c[level]['confidence'] for c in good_cells])
        bad_confidence = np.array([c[level]['confidence'] for c in bad_cells])
        axis.hist(
            good_confidence,
            bins=nbins,
            color='b',
            zorder=0,
            alpha=1.0,
            label=f"{n_good:.2e} correctly assigned at '{hierarchy_level}'",
            density=True)

        axis.hist(
            bad_confidence,
            bins=nbins,
            color='r',
            zorder=1,
            alpha=0.7,
            label=f"{n_bad:.2e} incorrectly assigned at '{hierarchy_level}'",
            density=True)

        #axis.set_yscale('log')

        axis.legend(loc='upper left', fontsize=fontsize)
        axis.set_xlabel('confidence level', fontsize=fontsize)
        axis.set_ylabel('normalized histogram', fontsize=fontsize)
        axis.set_title(f"confidence at level={level}", fontsize=fontsize)

    #fig.tight_layout()
    #print(f"writing {fig_path}")
    #fig.savefig(fig_path)


def do_full_fig(
        cell_to_truth,
        cell_to_assignment,
        taxonomy_tree,
        inverted_tree,
        fig_path):

    nrows = len(taxonomy_tree['hierarchy'])
    ncols = 1
    fig = mfig.Figure(figsize=(ncols*10, nrows*10))
    axis_list = [fig.add_subplot(nrows, ncols, ii+1)
                 for ii in range(nrows*ncols)]

    for i_level, level in enumerate(taxonomy_tree['hierarchy']):
        this_axis_list = [axis_list[i_level]]
        assess_results(
            cell_to_truth=cell_to_truth,
            cell_to_assignment=cell_to_assignment,
            taxonomy_tree=taxonomy_tree,
            inverted_tree=inverted_tree,
            hierarchy_level=level,
            axis_list=this_axis_list)
    fig.tight_layout()
    fig.savefig(fig_path)

def main():
    parser = argparse.argument_parser()
    parser.add_argument('--result_path', type=str, default=None)
    parser.add_argument('--fig_path', type=str, default=None)
    args = parser.parse_args()


    query_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/changkyul/CIRRO/MFISH/atlas_brain_638850.remap.4334174.updated.imputed.h5ad'

    #result_path = 'data/assignment_230406_full_election.json'

    #data_dir = pathlib.Path(
    #    '/allen/aibs/technology/danielsf/knowledge_base/validation')
    #assert data_dir.is_dir()
    #taxonomy_tree = json.load(open(data_dir / 'taxonomy_tree.json', 'rb'))

    result_path = args.result_path

    raw_results = json.load(open(result_path, 'rb'))
    results = raw_results['result']
    taxonomy_tree = raw_results['taxonomy_tree']
    del raw_results

    inverted_tree = invert_tree(taxonomy_tree)

    # get the truth
    truth_cache_path = pathlib.Path("data/truth_cache.json")
    if not truth_cache_path.is_file():
        a_data = anndata.read_h5ad(query_path, backed='r')
        obs = a_data.obs
        cell_to_truth = dict()
        for cell_id, cluster_value in zip(
                a_data.obs_names.values, obs['best.cl'].values):
           cell_to_truth[cell_id] = str(cluster_value)
        with open(truth_cache_path, "w") as out_file:
            out_file.write(json.dumps(cell_to_truth))

    cell_to_truth = json.load(open(truth_cache_path, "rb"))

    do_full_fig(
        cell_to_truth=cell_to_truth,
        cell_to_assignment=results,
        taxonomy_tree=taxonomy_tree,
        inverted_tree=inverted_tree,
        fig_path=args.fig_path)

if __name__ == "__main__":
    main()
