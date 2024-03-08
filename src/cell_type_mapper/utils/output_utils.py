import h5py
import json
import numpy as np
import pandas as pd
import pathlib
import time

import cell_type_mapper

from cell_type_mapper.utils.utils import (
    get_timestamp)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)


def re_order_blob(
        results_blob,
        query_path):
    """
    Re-order the cells in results_blob to match the order in which they appear
    in the h5ad file pointed to by query_path
    Parameters
    ----------
    results_blob:
        The blob-form results. A list of dicts. Each dict is a cell
        and looks like
            {
                'cell_id': my_cell,
                'level1': {'assignment':...
                           'confidence':...},
                'level2': {'assignment':...,
                           'confidence':...}}

    query_path:
        Path to the query h5ad file. The contents of results_blob will be
        re-ordered to match the order of query_path.obs.index.values

    Returns
    -------
    results_blob:
        re-ordered accordingly
    """

    cell_order = read_df_from_h5ad(
            h5ad_path=query_path,
            df_name='obs').index.values

    results_blob = {
        c['cell_id']: c for c in results_blob}

    results_blob = list([
        results_blob[c] for c in cell_order])

    return results_blob


def blob_to_csv(
        results_blob,
        taxonomy_tree,
        output_path,
        confidence_key='confidence',
        confidence_label='confidence',
        metadata_path=None,
        config=None):
    """
    Write a set of results originally formatted for a JSON blob
    out to the CSV our users will expect.

    Parameters
    ----------
    results_blob:
        The blob-form results. A list of dicts. Each dict is a cell
        and looks like
            {
                'cell_id': my_cell,
                'level1': {'assignment':...
                           'confidence':...},
                'level2': {'assignment':...,
                           'confidence':...}}
    taxonomy_tree:
        TaxonomyTree that went into this run.
        Hierarchy will be listed as a comment at the top of this file.
    output_path:
        Path to the CSV file that will be written
    confidence_key:
        The key with in the blob dict pointing to the confidence stat
    confidence_label:
        The name used for the confidence stat in the CSV header
    metadata_path:
        Path to the metadta path going with this output (if any;
        file name will be recorded as a comment at the top of
        the file)
    config:
        Optional dict containing the parameters with which the code
        was run.
    """
    str_hierarchy = json.dumps(taxonomy_tree.hierarchy)

    with open(output_path, 'w') as dst:
        if metadata_path is not None:
            metadata_path = pathlib.Path(metadata_path)
            dst.write(f'# metadata = {metadata_path.name}\n')
        dst.write(f'# taxonomy hierarchy = {str_hierarchy}\n')
        readable_hierarchy = [
            taxonomy_tree.level_to_name(level_label=level_label)
            for level_label in taxonomy_tree.hierarchy]
        if readable_hierarchy != taxonomy_tree.hierarchy:
            str_readable_hierarchy = json.dumps(readable_hierarchy)
            dst.write(f'# readable taxonomy hierarchy = '
                      f'{str_readable_hierarchy}\n')

        version_str = '#'
        if config is not None:
            if config['flatten']:
                version_str += " algorithm: 'correlation';"
            else:
                version_str += " algorithm: 'hierarchical';"
        version_str += f" codebase: {cell_type_mapper.__repository__};"
        version_str += f" version: {cell_type_mapper.__version__}\n"
        dst.write(version_str)

        csv_df = blob_to_df(
            results_blob=results_blob,
            taxonomy_tree=taxonomy_tree)

        column_rename = dict()
        for level in taxonomy_tree.hierarchy:
            readable_level = taxonomy_tree.level_to_name(level_label=level)
            src_key = f"{readable_level}_{confidence_key}"
            dst_key = f"{readable_level}_{confidence_label}"
            column_rename[src_key] = dst_key
        csv_df.rename(mapper=column_rename, axis=1, inplace=True)

        columns_to_drop = []
        for col in csv_df.columns:
            if col == 'cell_id':
                continue
            if 'name' in col or 'label' in col or 'alias' in col:
                continue
            if confidence_label in col:
                continue
            columns_to_drop.append(col)

        if len(columns_to_drop) > 0:
            csv_df.drop(columns_to_drop, axis=1, inplace=True)

        csv_df.to_csv(dst, index=False, float_format='%.4f')


def blob_to_df(
        results_blob,
        taxonomy_tree):
    """
    Convert a JSON blob of results into a pandas dataframe
    """
    records = []
    for cell in results_blob:
        this_record = {'cell_id': cell['cell_id']}
        for level in taxonomy_tree.hierarchy:
            readable_level = taxonomy_tree.level_to_name(level_label=level)
            label = cell[level]['assignment']
            name = taxonomy_tree.label_to_name(
                        level=level,
                        label=label,
                        name_key='name')
            this_record[f'{readable_level}_label'] = label
            this_record[f'{readable_level}_name'] = name
            if level == taxonomy_tree.leaf_level:
                alias = taxonomy_tree.label_to_name(
                            level=level,
                            label=label,
                            name_key='alias')
                this_record[f'{readable_level}_alias'] = alias

            for element in cell[level]:
                if element == 'assignment':
                    continue
                value = cell[level][element]
                if isinstance(value, list):
                    for idx in range(len(value)):
                        key = f'{readable_level}_{element}_{idx}'
                        this_record[key] = value[idx]
                else:
                    key = f'{readable_level}_{element}'
                    this_record[key] = value

        records.append(this_record)

    df = pd.DataFrame(records)

    for col in df.columns:
        convert_to_category = False
        if 'label' in col:
            convert_to_category = True
        elif 'name' in col:
            convert_to_category = True
        elif 'alias' in col:
            convert_to_category = True
        elif 'assignment' in col:
            convert_to_category = True

        if convert_to_category:
            df[col] = df[col].astype('category')

    return df


def get_execution_metadata(
        module_file,
        t0):
    """
    Parameters
    ----------
    module_file:
        Result of __file__ in whatever piece of code
        is calling this function
    t0:
       The start time for duration calculation.
       If None, ignore duration calculation

    Return a dict containing
        timestamp  -- the time at which this function was called
        duration -- how long the code took to run in seconds
        module -- what module was run
        codebase -- the code repository
        version -- version of cell_type_mapper that was run

    """
    metadata = dict()
    metadata['timestamp'] = get_timestamp()
    if t0 is not None:
        metadata['duration'] = time.time()-t0

    ctm_parent = pathlib.Path(cell_type_mapper.__file__).parent.parent
    module = pathlib.Path(module_file).relative_to(ctm_parent)
    metadata['module'] = str(module)
    metadata['version'] = cell_type_mapper.__version__
    metadata['codebase'] = cell_type_mapper.__repository__
    return metadata


def blob_to_hdf5(
        output_blob,
        dst_path):

    metadata = dict()
    for k in output_blob:
        if k == 'results':
            continue
        metadata[k] = output_blob[k]

    taxonomy_tree = TaxonomyTree(
        data=metadata['taxonomy_tree'])

    n_levels = len(taxonomy_tree.hierarchy)

    results = output_blob['results']
    n_cells = len(results)

    cell_id = np.array([r['cell_id'].encode('utf-8') for r in results])

    corr = np.zeros(
        (n_cells, n_levels), dtype=float)
    prob = np.zeros(
        (n_cells, n_levels), dtype=float)
    assignments = np.zeros(
        (n_cells, n_levels), dtype=int)

    bad_val = -1
    n_runners_up = metadata['config']['type_assignment']['n_runners_up']
    if n_runners_up > 0:
        r_assignments = bad_val*np.ones(
            (n_cells, n_levels, n_runners_up), dtype=int)
        r_prob = np.zeros(
            (n_cells, n_levels, n_runners_up), dtype=float)
        r_corr = np.zeros(
            (n_cells, n_levels, n_runners_up), dtype=float)
    else:
        r_assignments = None
        r_prob = None
        r_corr = None

    node_to_int = dict()
    int_to_node = dict()
    for level in taxonomy_tree.hierarchy:
        node_to_int[level] = dict()
        these_nodes = taxonomy_tree.nodes_at_level(level)
        int_to_node[level] = [None]*len(these_nodes)
        for i_node, node in enumerate(these_nodes):
            node_to_int[level][node] = i_node
            int_to_node[level][i_node] = node

    for i_cell in range(len(results)):
        cell = results[i_cell]
        for i_level, level in enumerate(taxonomy_tree.hierarchy):
            idx = node_to_int[level][cell[level]['assignment']]
            assignments[i_cell, i_level] = idx
            corr[i_cell, i_level] = cell[level]['avg_correlation']
            prob[i_cell, i_level] = cell[level]['bootstrapping_probability']
            if 'runner_up_assignment' in cell[level]:
                this_n = len(cell[level]['runner_up_assignment'])
                for i_r in range(this_n):
                    idx = node_to_int[level][
                        cell[level]['runner_up_assignment'][i_r]]
                    r_assignments[i_cell, i_level, i_r] = idx
                    r_prob[i_cell, i_level, i_r] = cell[level][
                            'runner_up_probability'][i_r]
                    r_corr[i_cell, i_level, i_r] = cell[level][
                            'runner_up_correlation'][i_r]

    with h5py.File(dst_path, 'w') as dst:
        dst.create_dataset(
            'metadata',
            data=json.dumps(metadata).encode('utf-8'))

        dst.create_dataset(
            'int_to_node',
            data=json.dumps(int_to_node).encode('utf-8'))

        dst.create_dataset(
            'cell_id',
            data=cell_id)

        for name, data in zip(('assignment',
                               'bootstrapping_probability',
                               'average_correlation',
                               'runner_up_assignment',
                               'runner_up_probability',
                               'runner_up_correlation'),
                              (assignments,
                               prob,
                               corr,
                               r_assignments,
                               r_prob,
                               r_corr)):
            if data is not None:
                dst.create_dataset(
                    name,
                    data=data,
                    compression='gzip',
                    compression_opts=4,
                    chunks=True)


def hdf5_to_blob(
        src_path):

    with h5py.File(src_path, 'r') as src:
        blob = json.loads(
            src['metadata'][()].decode('utf-8'))
        cell_id_arr = src['cell_id'][()]
        int_to_node = json.loads(
            src['int_to_node'][()].decode('utf-8'))

        assignment = src['assignment'][()]
        prob = src['bootstrapping_probability'][()]
        corr = src['average_correlation'][()]
        if 'runner_up_assignment' in src:
            r_assignment = src['runner_up_assignment'][()]
            r_prob = src['runner_up_probability'][()]
            r_corr = src['runner_up_correlation'][()]
        else:
            r_assignment = None
            r_prob = None
            r_corr = None

    taxonomy_tree = TaxonomyTree(data=blob['taxonomy_tree'])

    results = []
    for i_cell, cell_id in enumerate(cell_id_arr):
        cell = {'cell_id': cell_id.decode('utf-8')}
        for i_level, level in enumerate(taxonomy_tree.hierarchy):
            this = {
                'assignment': int_to_node[level][assignment[i_cell, i_level]],
                'bootstrapping_probability': prob[i_cell, i_level],
                'avg_correlation': corr[i_cell, i_level],
                'runner_up_assignment': [],
                'runner_up_probability': [],
                'runner_up_correlation': []
            }
            if r_assignment is not None:
                n_r = r_assignment.shape[-1]
                for i_r in range(n_r):
                    if r_assignment[i_cell, i_level, i_r] < 0:
                        continue
                    node = int_to_node[level][
                        r_assignment[i_cell, i_level, i_r]]
                    this['runner_up_assignment'].append(node)
                    this['runner_up_probability'].append(
                        r_prob[i_cell, i_level, i_r])
                    this['runner_up_correlation'].append(
                        r_corr[i_cell, i_level, i_r])
            cell[level] = this
        results.append(cell)
    blob['results'] = results
    return blob
