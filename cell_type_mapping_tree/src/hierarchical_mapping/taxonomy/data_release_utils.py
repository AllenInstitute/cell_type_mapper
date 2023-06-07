"""
This module will contain utility functions to help read the CSV files that
are part of an official data release taxonomy.
"""


def get_header_map(
        csv_path,
        desired_columns):
    """
    Return a dict mapping the name of columns to the
    index of the columns indicating where they appear in
    a csv

    Parameters
    ----------
    csv_path:
        Path to the csv
    desired_columns:
        List of the column_names whose locations you want
    """
    error_msg = ''
    with open(csv_path, 'r') as src:
        header_line = src.readline()
    header_line = header_line.strip().split(',')
    desired_columns = set(desired_columns)
    result = dict()
    for idx, value in enumerate(header_line):
        if value in desired_columns:
            if value in result:
                error_msg += f"column '{value}' occurs more than once\n"
            result[value] = idx
    for expected in desired_columns:
        if expected not in result:
            error_msg += f"could not find column '{expected}'\n"
    if len(error_msg) > 0:
        error_msg = f"errors parsing {csv_path}:\n{error_msg}"
        raise RuntimeError(error_msg)
    return result


def get_tree_above_leaves(
        csv_path,
        hierarchy):
    """
    Read the structure of a cell type tree (excluding the leaf nodes)
    from the cluster_annotation_term.csv file

    Parameters
    ----------
    csv_path:
        Path to the CSV file
    hierarchy:
        List of string encoding the hierarchy of
        cluster_annotation_term_set_labels in order from
        most gross to most fine.

    Returns
    -------
    A dict encoding the inheritance structure
     (these will be in terms of labels; not aliases)
    """

    header_lookup = get_header_map(
        csv_path=csv_path,
        desired_columns=[
            'label',
            'cluster_annotation_term_set_label',
            'parent_term_label',
            'parent_term_set_label'])

    child_to_parent = {
        l0: l1
        for l0, l1 in zip(hierarchy[1:], hierarchy[:-1])}

    label_idx = header_lookup['label']
    level_idx = header_lookup['cluster_annotation_term_set_label']
    parent_idx = header_lookup['parent_term_label']
    parent_level_idx = header_lookup['parent_term_set_label']

    result = dict()
    with open(csv_path, 'r') as src:
        src.readline()
        for line in src:
            params = line.strip().split(',')
            label = params[label_idx]
            level = params[level_idx]
            parent = params[parent_idx]
            parent_level = params[parent_level_idx]
            if level not in child_to_parent:
                continue
            if parent_level != child_to_parent[level]:
                msg = f"node {label} at level {level} expected "
                msg += f"to have a parent at level {child_to_parent[level]}; "
                msg += f"got parent {parent} and level {parent_level} instead"
                raise RuntimeError(msg)

            if parent_level not in result:
                result[parent_level] = dict()
            if parent not in result[parent_level]:
                result[parent_level][parent] = set()
            result[parent_level][parent].add(label)

    for level in result:
        for node in result[level]:
            result[level][node] = list(result[level][node])
            result[level][node].sort()
    return result


def get_alias_mapper(
        csv_path):
    """
    Read a cluster_to_cluster_annotation_membership.csv file. Return
    a dict mapping (level, label) to alias
    """
    header_lookup = get_header_map(
        csv_path=csv_path,
        desired_columns=[
            'cluster_annotation_term_set_label',
            'cluster_alias',
            'cluster_annotation_term_label'])

    level_idx = header_lookup['cluster_annotation_term_set_label']
    label_idx = header_lookup['cluster_annotation_term_label']
    alias_idx = header_lookup['cluster_alias']

    result = dict()
    used_aliases = dict()
    with open(csv_path, 'r') as src:
        src.readline()
        for line in src:
            params = line.strip().split(',')
            level = params[level_idx]
            label = params[label_idx]
            alias = params[alias_idx]
            this_key = (level, label)
            if this_key in result:
                raise RuntimeError(
                    f"level={level}, label={label} listed more than "
                    f"once in {csv_path}")
            if level not in used_aliases:
                used_aliases[level] = set()
            if alias in used_aliases[level]:
                raise RuntimeError(
                    f"alias={alias} used more than once at level="
                    f"{level}")
            used_aliases[level].add(alias)
            result[this_key] = alias
    return result


def get_cell_to_cluster_alias(
        csv_path):
    """
    Read a cell_metadata.csv file. Return a dict mapping
    cell_id to cluster_alias
    """
    header_map = get_header_map(
        csv_path=csv_path,
        desired_columns=[
            'cell_label',
            'cluster_alias'])

    cell_idx = header_map['cell_label']
    cluster_idx = header_map['cluster_alias']
    result = dict()
    with open(csv_path, 'r') as src:
        src.readline()
        for line in src:
            params = line.strip().split(',')
            cell = params[cell_idx]
            cluster = params[cluster_idx]
            if cell in result:
                raise RuntimeError(
                    f"cell {cell} listed more than once in {csv_path}")
            result[cell] = cluster
    return result
