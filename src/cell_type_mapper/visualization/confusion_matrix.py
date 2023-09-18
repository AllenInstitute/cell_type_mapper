import matplotlib.figure as mfig
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

from anndata._io.specs import read_elem
import h5py
import json
import numpy as np
import pathlib
import re

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)


def many_summary_plots_pdf(
        classification_path_list,
        ground_truth_column_list,
        plot_path,
        is_log10=False,
        munge_ints=False,
        is_flat=False,
        confidence_key_list=['bootstrapping_probability'],
        leaf_order=None,
        drop_level=None):

    with PdfPages(plot_path) as pdf_handle:
        for classification_path in classification_path_list:
            summary_plots_for_one_file(
                classification_path=classification_path,
                ground_truth_column_list=ground_truth_column_list,
                pdf_handle=pdf_handle,
                is_log10=is_log10,
                munge_ints=munge_ints,
                is_flat=is_flat,
                confidence_key_list=confidence_key_list,
                leaf_order=leaf_order,
                drop_level=drop_level)


def single_summary_plot_pdf(
        classification_path,
        ground_truth_column_list,
        plot_path,
        is_log10=False,
        munge_ints=False,
        is_flat=False,
        confidence_key_list=['bootstrapping_probability'],
        leaf_order=None,
        drop_level=None):

    with PdfPages(plot_path) as pdf_handle:
        summary_plots_for_one_file(
            classification_path=classification_path,
            ground_truth_column_list=ground_truth_column_list,
            pdf_handle=pdf_handle,
            is_log10=is_log10,
            munge_ints=munge_ints,
            is_flat=is_flat,
            confidence_key_list=confidence_key_list,
            leaf_order=leaf_order,
            drop_level=drop_level)


def summary_plots_for_one_file(
        classification_path,
        ground_truth_column_list,
        pdf_handle,
        is_log10,
        munge_ints,
        is_flat=False,
        confidence_key_list=['bootstrapping_probability'],
        leaf_order=None,
        drop_level=None):

    classification_path = pathlib.Path(classification_path)
    print(classification_path.name)
    results = json.load(open(classification_path, 'rb'))

    n_cells = len(results['results'])
    query_path = pathlib.Path(results['config']['query_path'])
    query_path_str = f"{query_path.parent.name}/\n    {query_path.name}"
    marker_path = pathlib.Path(
        results['config']['query_markers']['serialized_lookup'])
    marker_lookup_str = f"{marker_path.parent.name}/\n    {marker_path.name}"
    precompute_path = pathlib.Path(
        results['config']['precomputed_stats']['path'])
    precompute_str = (
        f"{precompute_path.parent.name}/\n    {precompute_path.name}")

    log = results['log']
    timing_statements = []
    for line in log:
        if "RAN" in line or "BENCHMARK" in line:
            timing_statements.append(line)

    precomputed_path = results['config']['precomputed_stats']['path']
    with h5py.File(precomputed_path, 'r') as src:
        taxonomy_tree = TaxonomyTree(
            data=json.loads(src['taxonomy_tree'][()].decode('utf-8')))

    if drop_level is not None:
        taxonomy_tree = taxonomy_tree.drop_level(drop_level)

    inverted_tree = invert_tree(taxonomy_tree)
    raw_results = results.pop('results')

    result_levels = set()
    for cell in raw_results:
        for level in taxonomy_tree.hierarchy:
            if level not in cell:
                continue
            result_levels.add(level)
            old = cell[level].pop('assignment')
            if level == taxonomy_tree.leaf_level:
                name_key = 'alias'
            else:
                name_key = 'name'
            new_name = taxonomy_tree.label_to_name(
               level=level,
               label=old,
               name_key=name_key)
            cell[level]['assignment'] = new_name

    if not is_flat:
        results_lookup = {
            cell['cell_id']: cell for cell in raw_results}
    else:
        results_lookup = dict()
        for raw_cell in raw_results:
            cell = {'cell_id': raw_cell['cell_id']}
            for level in taxonomy_tree.hierarchy:
                if level == taxonomy_tree.leaf_level:
                    cell[level] = {'assignment': raw_cell['assignment']}
                    for confidence_key in confidence_key_list:
                        cell[level][confidence_key] = raw_cell[confidence_key]
                else:
                    cell[level] = {
                        'assignment':
                        inverted_tree[level][raw_cell['assignment']]}
                    for confidence_key in confidence_key_list:
                        cell[level][confidence_key] = 1.0
            results_lookup[raw_cell['cell_id']] = cell

    query_path = pathlib.Path(results['config']['query_path'])

    with h5py.File(query_path, 'r') as src:
        query_obs = read_elem(src['obs'])
    for g in ground_truth_column_list:
        if g in query_obs.columns:
            ground_truth_column = g
            break
    print(f"using ground truth {ground_truth_column}")
    assert ground_truth_column in query_obs.columns

    if leaf_order is None:
        leaf_list = taxonomy_tree.all_leaves
        if munge_ints:
            leaf_names = []
            leaf_idx = []
            int_pattern = re.compile('[0-9]+')
            for n in leaf_list:
                leaf_names.append(
                    taxonomy_tree.label_to_name(
                        level=taxonomy_tree.leaf_level,
                        label=n,
                        name_key='alias'))
                ii = int(int_pattern.findall(n)[0])
                assert ii not in leaf_idx
                leaf_idx.append(ii)
            leaf_idx = np.array(leaf_idx)
            leaf_names = np.array(leaf_names)
            sorted_dex = np.argsort(leaf_idx)
            leaf_order = leaf_names[sorted_dex]
        else:
            leaf_order = []
            for n in leaf_list:
                leaf_order.append(
                    taxonomy_tree.label_to_name(
                        level=taxonomy_tree.leaf_level,
                        label=n,
                        name_key='alias'))
            leaf_order.sort()

    obs = query_obs

    n_levels = len(taxonomy_tree.hierarchy)-1

    grid_gap = 5
    grid_height = 20
    grid_width = 20
    msg_width = 20

    n_fig_cols = 3 + len(confidence_key_list)

    full_width = msg_width+n_fig_cols*(grid_height+grid_gap)+1
    full_height = n_levels*grid_height + (n_levels-1)*grid_gap+1+grid_gap
    grid = gridspec.GridSpec(nrows=full_height, ncols=full_width)

    grid.update(
        bottom=0.1,
        top=0.99,
        left=0.01,
        right=0.99,
        wspace=0.1,
        hspace=0.01)

    fig_width = np.ceil(full_width*0.5).astype(int)
    fig_height = np.ceil(full_height*0.5).astype(int)
    fig = mfig.Figure(figsize=(fig_width, fig_height), dpi=300)

    axis_list = [
        fig.add_subplot(grid[0:full_height, 0:msg_width])]

    sub_axis_lists = []
    for i_row in range(n_levels):
        this_sub_list = []
        r0 = i_row*grid_gap+i_row*grid_height + grid_gap
        r1 = r0 + grid_height
        assert r1 < full_height
        for i_col in range(n_fig_cols-len(confidence_key_list)):
            c0 = msg_width+(i_col+1)*grid_gap+i_col*grid_width
            c1 = c0 + grid_width
            assert c1 < full_width
            this_axis = fig.add_subplot(grid[r0:r1, c0:c1])
            this_sub_list.append(this_axis)
            axis_list.append(this_axis)
        sub_axis_lists.append(this_sub_list)

    histogram_axis_lookup = dict()
    for i_conf, confidence_key in enumerate(confidence_key_list):

        c0 = msg_width+(3+i_conf+1)*grid_gap+(3+i_conf)*grid_width
        c1 = c0 + grid_width
        histogram_axis_lookup[confidence_key] = fig.add_subplot(
            grid[grid_gap:grid_height, c0:c1])

    good_confidence_lookup = {n: [] for n in confidence_key_list}
    bad_confidence_lookup = {n: [] for n in confidence_key_list}

    accuracy_statements = []
    for i_level, level in enumerate(taxonomy_tree.hierarchy):
        if level not in result_levels:
            continue

        if i_level < len(sub_axis_lists):
            this_axis_list = sub_axis_lists[i_level]
        else:
            this_axis_list = None

        if level == taxonomy_tree.leaf_level:
            label_order = leaf_order
        else:
            label_order = []
            for leaf in leaf_order:
                if leaf in inverted_tree[level]:
                    this = inverted_tree[level][leaf]
                    if this not in label_order:
                        label_order.append(this)

        these_experiments = []
        these_truth = []
        these_cells = []
        for cell_id, ground_truth in zip(obs.index.values,
                                         obs[ground_truth_column].values):
            if munge_ints:
                gt_key = f"cl.{ground_truth}"
            else:
                gt_key = f"{ground_truth}"

            if gt_key in inverted_tree[level]:
                these_cells.append(cell_id)
                these_experiments.append(
                    results_lookup[cell_id][level]['assignment'])
                these_truth.append(
                    inverted_tree[level][gt_key])

        good = 0
        bad = 0

        for cell_id, truth, experiment in zip(these_cells,
                                              these_truth,
                                              these_experiments):
            if level == taxonomy_tree.leaf_level:
                cell = results_lookup[cell_id]
                confidence = {n: 1.0 for n in confidence_key_list}
                for ll in taxonomy_tree.hierarchy:
                    if ll in cell:
                        for confidence_key in confidence_key_list:
                            confidence[confidence_key] *= \
                                cell[ll][confidence_key]
            if truth == experiment:
                good += 1
                if level == taxonomy_tree.leaf_level:
                    for confidence_key in confidence_key_list:
                        good_confidence_lookup[confidence_key].append(
                            confidence[confidence_key])
            else:
                bad += 1
                if level == taxonomy_tree.leaf_level:
                    for confidence_key in confidence_key_list:
                        bad_confidence_lookup[confidence_key].append(
                            confidence[confidence_key])
        print(f"{level} {good} {bad}")
        msg = f"{level} correctly mapped: {good} -- {good/float(good+bad):.3e}"
        accuracy_statements.append(f"{msg}\n")
        print(msg)

        if i_level == (n_levels-1):
            label_x_axis = True
        else:
            label_x_axis = False

        if this_axis_list is not None:
            plot_confusion_matrix(
                figure=fig,
                axis=this_axis_list[0],
                true_labels=these_truth,
                experimental_labels=these_experiments,
                label_order=label_order,
                normalize_by='truth',
                fontsize=20,
                title=f"{level} normalized by true label",
                is_log=is_log10,
                label_x_axis=label_x_axis,
                label_y_axis=True)

            plot_confusion_matrix(
                figure=fig,
                axis=this_axis_list[1],
                true_labels=these_truth,
                experimental_labels=these_experiments,
                label_order=label_order,
                normalize_by='experiment',
                fontsize=20,
                title=f"{level} normalized by mapped label",
                is_log=is_log10,
                label_x_axis=label_x_axis,
                label_y_axis=False,)

            plot_confusion_matrix(
                figure=fig,
                axis=this_axis_list[2],
                true_labels=these_truth,
                experimental_labels=these_experiments,
                label_order=label_order,
                normalize_by=None,
                fontsize=20,
                title=f"{level} raw count",
                is_log=True,
                label_x_axis=label_x_axis,
                label_y_axis=False,)

    output_path = pathlib.Path(results['config']['extended_result_path'])
    msg = f"{output_path.name}\n"
    msg += f"query set: {query_path_str}\n"
    msg += f"precomputed stats: {precompute_str}\n"
    msg += f"marker genes: {marker_lookup_str}\n"
    msg += f"{n_cells} query cells\n"
    msg += "\naccuracy\n=========\n"
    for line in accuracy_statements:
        msg += line
    msg += "\ntiming\n=========\n"
    bmark_pattern = re.compile("BENCHMARK")
    for line in timing_statements:
        if "BENCHMARK" in line:
            position = bmark_pattern.search(line)
            line = line[position.start()+11:]
        line_list = split_sentence(line)
        for sub in line_list:
            msg += sub+"\n"

    print(msg)
    axis_list[0].text(
       5, 100, msg, fontsize=20,
       verticalalignment='top')
    axis_list[0].set_ylim(0, 110)
    axis_list[0].set_xlim(0, 100)
    for s in ('top', 'left', 'bottom', 'right'):
        axis_list[0].spines[s].set_visible(False)
    axis_list[0].tick_params(
        axis='both', which='both', size=0, labelsize=0)

    for confidence_key in confidence_key_list:
        good_confidence = good_confidence_lookup[confidence_key]
        bad_confidence = bad_confidence_lookup[confidence_key]
        histogram_axis = histogram_axis_lookup[confidence_key]

        print(
            f"good {confidence_key} {np.mean(good_confidence)} "
            f"+/- {np.std(good_confidence)}")
        print(
            f"bad {confidence_key} {np.mean(bad_confidence)} "
            f"+/- {np.std(bad_confidence)}")
        histogram_axis.hist(good_confidence, bins=100, density=True,
                            zorder=0, color='b', label='correct cells')
        histogram_axis.hist(
            bad_confidence,
            bins=100,
            density=True,
            zorder=1,
            alpha=0.7,
            color='r',
            label='incorrect cells')

        histogram_axis.legend(loc=0, fontsize=20)
        histogram_axis.set_xlabel(confidence_key, fontsize=20)
        histogram_axis.set_ylabel('density', fontsize=20)

    pdf_handle.savefig(fig)


def plot_confusion_matrix(
        figure,
        axis,
        true_labels,
        experimental_labels,
        label_order,
        normalize_by='truth',
        fontsize=20,
        title=None,
        is_log=False,
        munge_ints=True,
        label_x_axis=True,
        label_y_axis=True):

    img = np.zeros((len(label_order), len(label_order)), dtype=int)
    label_to_idx = {
        l: ii for ii, l in enumerate(label_order)}

    for truth, experiment in zip(true_labels, experimental_labels):
        true_idx = label_to_idx[truth]
        experiment_idx = label_to_idx[experiment]
        img[true_idx, experiment_idx] += 1

    s0 = img.sum()
    img, thinned_labels = thin_img(img, label_list=label_order)
    assert img.sum() == s0

    img = np.ma.masked_array(
        img, mask=(img == 0))

    if normalize_by is not None:
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
        if normalize_by is not None:
            raise RuntimeError(
                f"normalize_by {normalize_by} makes no sense")

    if normalize_by is None:
        cax_title = 'raw count'
    else:
        cax_title = 'normalized count'

    if is_log:
        cax_title = f'log10({cax_title})'
        with np.errstate(divide='ignore'):
            log10_img = np.log10(img.astype(float))
            img = np.ma.masked_array(
                log10_img, mask=(img == 0))

    display_img = axis.imshow(img, cmap='cool')
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = figure.colorbar(
        display_img,
        ax=axis,
        cax=cax,
        label=cax_title)

    cbar.ax.tick_params(axis="both", which="both", labelsize=fontsize)
    cbar.ax.set_ylabel(ylabel=cax_title, fontdict={'fontsize': fontsize})

    for s in ('top', 'right', 'left', 'bottom'):
        axis.spines[s].set_visible(False)

    if label_x_axis:
        axis.set_xlabel('mapped label', fontsize=fontsize)
    if label_y_axis:
        axis.set_ylabel('true label', fontsize=fontsize)

    if title is not None:
        axis.set_title(title, fontsize=fontsize)

    tick_values = [ii for ii in range(len(thinned_labels))]
    axis.set_xticks(tick_values)
    axis.set_xticklabels(thinned_labels, fontsize=15, rotation='vertical')
    axis.set_yticks(tick_values)
    axis.set_yticklabels(thinned_labels, fontsize=15, rotation='horizontal')


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


def split_sentence(sentence, char_lim=60):
    line_list = []
    for i0 in range(0, len(sentence), char_lim):
        this_line = sentence[i0:i0+char_lim]
        if i0 > 0:
            this_line = f"    {this_line}"
        this_line += "-"
        line_list.append(this_line)
    line_list[-1] = line_list[-1][:-1]
    return line_list


def invert_tree(tree):
    as_leaves = tree.as_leaves
    inverse_lookup = dict()
    for level in as_leaves:
        inverse_lookup[level] = dict()
        for parent in as_leaves[level]:
            if level == tree.leaf_level:
                name_key = 'alias'
            else:
                name_key = 'name'
            parent_name = tree.label_to_name(
                level=level,
                label=parent,
                name_key=name_key)
            for leaf in as_leaves[level][parent]:
                alias = tree.label_to_name(
                    level=tree.leaf_level,
                    label=leaf,
                    name_key='alias')
                inverse_lookup[level][alias] = parent_name
    return inverse_lookup
