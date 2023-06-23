"""
This module will contain utility functions to help read the CSV files that
are part of an official data release taxonomy.
"""
import json


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


def get_term_set_map(
        csv_path):
    """
    Infer a mapping from cluster_annotation_term_set_label to
    cluster_annotation_term_set_name from the
    cluster_to_cluster_annotation_membership_term.csv file

    Parameters
    ----------
    csv_path:
        Path to the CSV file

    Returns
    -------
    A dict mapping cluster_annotation_term_set_label
    to cluster_annotation_term_set_name
    """
    header_lookup = get_header_map(
        csv_path=csv_path,
        desired_columns=[
            'cluster_annotation_term_set_label',
            'cluster_annotation_term_set_name'])
    label_idx = header_lookup['cluster_annotation_term_set_label']
    name_idx = header_lookup['cluster_annotation_term_set_name']

    result = dict()
    with open(csv_path, 'r') as src:
        src.readline()
        for line in src:
            params = line.strip().split(',')
            label = params[label_idx]
            name = params[name_idx]
            if label in result:
                if name != result[label]:
                    raise RuntimeError(
                        f"label {label} maps to at least two names: "
                        f"{name} and {result[label]}")
            result[label] = name
    return result


def get_label_to_name(
        csv_path,
        valid_term_set_labels,
        name_column='cluster_alias',
        strict_alias=True):
    """
    Read a cluster_to_cluster_annotation_membership.csv file. Return
    a dict mapping (level, label) to "name" (where name is some way
    of referring to the taxon that isn't the label)

    Only record aliases in the specified valid_term_set_labels

    name_column is the name of the column to treat
    as "name" (could also be cluster_annotation_term_name, for instance)

    if strict_alias == True, can only use each alias once per level
    """
    header_lookup = get_header_map(
        csv_path=csv_path,
        desired_columns=[
            'cluster_annotation_term_set_label',
            name_column,
            'cluster_annotation_term_label'])

    level_idx = header_lookup['cluster_annotation_term_set_label']
    label_idx = header_lookup['cluster_annotation_term_label']
    alias_idx = header_lookup[name_column]

    valid_term_set_labels = set(valid_term_set_labels)

    result = dict()
    used_aliases = dict()
    with open(csv_path, 'r') as src:
        src.readline()
        for line in src:
            params = line.strip().split(',')
            level = params[level_idx]
            if level not in valid_term_set_labels:
                continue
            label = params[label_idx]
            alias = params[alias_idx]
            this_key = (level, label)
            if this_key in result:
                old_result = result[this_key]
                if alias != old_result:
                    raise RuntimeError(
                        f"level={level}, label={label} listed more than "
                        f"once in {csv_path} with mappings '{alias}' and "
                        f"'{old_result}'")
            if level not in used_aliases:
                used_aliases[level] = set()
            if alias in used_aliases[level] and strict_alias:
                # check that the alias is actually mapped
                # to more than one label at this level (and not
                # that the same data appears in more than one
                # row of the csv)
                repeat_labels = [
                    k[1] for k in result
                    if k[0] == level and result[k] == alias]
                repeat_labels.append(label)
                if len(set(repeat_labels)) > 1:
                    raise RuntimeError(
                        f"alias={alias} used more than once at level="
                        f"{level}\nvalues\n"
                        f"{json.dumps(repeat_labels,indent=2)}")
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
