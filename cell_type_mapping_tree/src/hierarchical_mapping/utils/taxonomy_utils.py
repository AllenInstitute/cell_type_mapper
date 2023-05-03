import copy
import itertools


def compute_row_order(
        obs_records: list,
        column_hierarchy: list):
    """
    Compute the order of rows necessary such that
    cells of the same taxonomic classification are
    in contiguous blocks of rows.

    Parameters
    ----------
    obs_records:
        List of the records from the obs DataFrame of an
        anndata file.

    column_hierarcy:
        The list of columns denoting taxonomic classes,
        ordered from highest (parent) to lowest (child).

    Returns
    -------
    Dict
        "row_order" order original rows need to be put in to
        keep taxonomic classes contiguous

        "tree": dict defining taxonomic hierarchy (note that
        rows referred to in tree["cluster"] will be the remapped
        row indices)
    """

    raw_tree = get_taxonomy_tree(
                    obs_records=obs_records,
                    column_hierarchy=column_hierarchy)

    row_order = []

    parent_level = raw_tree['hierarchy'][0]
    parent_list = list(raw_tree[parent_level].keys())

    parent_list.sort()  # for determinacy of output

    for this_parent in parent_list:
        row_order += _get_rows_from_tree(
                        tree=raw_tree,
                        level=parent_level,
                        this_node=this_parent)

    old_row_to_new_row = {r: ii for ii, r in enumerate(row_order)}

    leaf_level = raw_tree['hierarchy'][-1]
    raw_leaf = raw_tree.pop(leaf_level)
    new_leaf = dict()
    for k in raw_leaf:
        new_leaf[k] = [old_row_to_new_row[o] for o in raw_leaf[k]]
    raw_tree[leaf_level] = new_leaf
    return {"row_order": row_order, "tree": raw_tree}


def _get_rows_from_tree(
        tree,
        level,
        this_node):
    """
    Tree is the inheritance tree
    level is the level in the hierarchy we are looking at
    this_node is the specific node we are looking at

    Ultimately returns list of rows
    """
    if level == tree['hierarchy'][-1]:
        result = tree[level][this_node]
        return result

    for idx in range(len(tree['hierarchy'])):
        if tree['hierarchy'][idx] == level:
            break

    child_level = tree['hierarchy'][idx+1]
    result = []
    child_nodes = copy.deepcopy(list(tree[level][this_node]))
    child_nodes.sort()  # for determinacy of output
    for this_child in child_nodes:
        result += _get_rows_from_tree(
                        tree=tree,
                        level=child_level,
                        this_node=this_child)
    return result


def get_taxonomy_tree(
        obs_records: list,
        column_hierarchy: list):
    """
    Convert a list of records into a taxonomy tree

    Parameters
    ----------
    obs_records:
        List of the records from the obs DataFrame of an
        anndata file.

    column_hierarcy:
        The list of columns denoting taxonomic classes,
        ordered from highest (parent) to lowest (child).

    Returns
    -------
    tree indicating inheritance structure of taxonomy
    """

    leaf_column = column_hierarchy[-1]
    tree = dict()
    for h in column_hierarchy:
        tree[h] = dict()

    if 'hierarchy' in tree:
        raise RuntimeError(
            "Cannot use hierarchy as a taxonomic row; "
            "that is where tree will store the ordered "
            "list of taxonomy names")

    tree["hierarchy"] = column_hierarchy

    for i_row, row in enumerate(obs_records):
        this_leaf = row[leaf_column]
        if this_leaf not in tree[leaf_column]:
            tree[leaf_column][this_leaf] = []
        tree[leaf_column][this_leaf].append(i_row)

        for parent_level, child_level in zip(column_hierarchy[:-1],
                                             column_hierarchy[1:]):
            this_parent = row[parent_level]
            this_child = row[child_level]
            if this_parent not in tree[parent_level]:
                tree[parent_level][this_parent] = set()
            tree[parent_level][this_parent].add(this_child)

    validate_taxonomy_tree(tree)
    return tree


def validate_taxonomy_tree(
        taxonomy_tree):
    """
    Make sure taxonomy_tree is a strict tree
    """
    child_to_parent = dict()
    hierarchy = taxonomy_tree['hierarchy']
    for level in hierarchy:
        child_to_parent[level] = dict()
    for parent_level, child_level in zip(hierarchy[:-1],
                                         hierarchy[1:]):
        child_set = taxonomy_tree[child_level].keys()
        for this_parent in taxonomy_tree[parent_level].keys():
            for this_child in taxonomy_tree[parent_level][this_parent]:
                if this_child not in child_set:
                    msg = f"{this_child} "
                    msg += f"(child of {parent_level}:{this_parent}) "
                    msg += f"is not present in the keys at level {child_level}"
                    raise RuntimeError(msg)

                if this_child in child_to_parent[child_level]:
                    if child_to_parent[child_level][this_child] != this_parent:
                        msg = f"at level {child_level}, node {this_child} "
                        msg += "has at least two parents:\n"
                        msg += f"{this_parent}\nand "
                        msg += f"{child_to_parent[child_level][this_child]}"
                        raise RuntimeError(msg)
                else:
                    child_to_parent[child_level][this_child] = this_parent


def convert_tree_to_leaves(
         taxonomy_tree):
    """
    Read in a taxonomy tree as computed by get_taxonomy_tree.

    Return a Dict structured like
        level ('class', 'subclass', 'cluster', etc.)
            -> node1 (a node on that level of the tree)
                -> list of leaf nodes making up that node
    """
    hierarchy = taxonomy_tree['hierarchy']
    result = dict()
    for this_level in hierarchy:
        this_result = dict()
        for node in taxonomy_tree[this_level]:
            leaves = _get_leaves_from_tree(
                        tree=taxonomy_tree,
                        level=this_level,
                        this_node=node)
            this_result[node] = leaves
        result[this_level] = this_result
    return result


def _get_leaves_from_tree(
        tree,
        level,
        this_node):
    """
    iteratively get a list of the leaf nodes
    inherity from tree[level][this_node]
    """
    hierarchy = tree['hierarchy']
    if level == hierarchy[-1]:
        return [this_node]

    if level == hierarchy[-2]:
        return list(tree[level][this_node])

    for idx in range(len(tree['hierarchy'])):
        if tree['hierarchy'][idx] == level:
            break

    child_level = tree['hierarchy'][idx+1]
    result = []
    child_nodes = copy.deepcopy(list(tree[level][this_node]))
    child_nodes.sort()  # for determinacy of output
    for this_child in child_nodes:
        result += _get_leaves_from_tree(
                        tree=tree,
                        level=child_level,
                        this_node=this_child)
    return result


def get_siblings(taxonomy_tree):
    """
    Read in a taxonomy tree as computed by get_taxonomy_tree.

    Return a list of tuples (level, node1, node2) indicating
    all of the siblings (nodes on the same level with the same
    immediate ancestor) in that tree.
    """
    hierarchy = taxonomy_tree["hierarchy"]
    results = []
    parent_list = list(taxonomy_tree[hierarchy[0]].keys())
    parent_list.sort()
    for pair in itertools.combinations(parent_list, 2):
        results.append((hierarchy[0], pair[0], pair[1]))

    for parent_level, child_level in zip(hierarchy[:-1], hierarchy[1:]):
        parent_list = taxonomy_tree[parent_level].keys()
        for parent in parent_list:
            child_list = list(taxonomy_tree[parent_level][parent])
            child_list.sort()
            for pair in itertools.combinations(child_list, 2):
                results.append((child_level, pair[0], pair[1]))

    return results


def get_all_pairs(taxonomy_tree):
    """
    Return all pairs of nodes that are on the same level
    """
    hierarchy = taxonomy_tree["hierarchy"]
    results = []
    for level in hierarchy:
        element_list = list(taxonomy_tree[level].keys())
        element_list.sort()
        for pair in itertools.combinations(element_list, 2):
            results.append((level, pair[0], pair[1]))
    return results


def get_all_leaf_pairs(
        taxonomy_tree,
        parent_node):
    """
    Find all of the leaf nodes that need to be compared
    under a given parent.

    i.e., if I know I am a member of node A, find all of the
    children (B1, B2, B3) and then finda all of the pairs
    (B1.L1, B2.L1), (B1.L1, B2.L2)...(B1.LN, B2.L1)...(B1.N, BN.LN)
    where B.LN are the leaf nodes that descend from B1, B2.LN are
    the leaf nodes that descend from B2, etc.

    Parameters
    ----------
    taxonomy_tree:
        A dict encoding the cell type taxonomy we are using
    parent_node:
        A tuple of the type (level, node) denoting the node
        we know these query cells belong to (so, we are selecting
        the marker genes for discribinating the level below this)

        If parent_node is None, then assume that we are selecting
        marker genes for the highest level of the taxonomy

    Returns
    -------
    A list of (level, leaf_node1, leaf_node2) tuples indicating
    the leaf nodes that need to be compared.
    """
    hierarchy = taxonomy_tree['hierarchy']
    leaf_level = hierarchy[-1]

    if parent_node is not None:
        if parent_node[0] == leaf_level:
            return []

        # find the level in the hierarchy that is the immediate
        # child of parent_node[0]
        for child_level_idx, level in enumerate(hierarchy):
            if level == parent_node[0]:
                break
        child_level_idx += 1

        if child_level_idx >= len(hierarchy):
            raise RuntimeError(
                f"Somehow, child_level_idx={child_level_idx}\n"
                f"while the hierarchy has {len(hierarchy)} levels;\n"
                f"parent_node = {parent_node}")
        child_level = hierarchy[child_level_idx]

        # all of the siblings that directly inherit from
        # parent_node[0]
        siblings = taxonomy_tree[parent_node[0]][parent_node[1]]
    else:
        siblings = list(taxonomy_tree[hierarchy[0]].keys())
        child_level = hierarchy[0]

    result = []
    tree_as_leaves = convert_tree_to_leaves(taxonomy_tree)
    for sibling_pair in itertools.combinations(siblings, 2):
        leaf_list_0 = tree_as_leaves[child_level][sibling_pair[0]]
        leaf_list_1 = tree_as_leaves[child_level][sibling_pair[1]]
        for leaf_pair in itertools.product(leaf_list_0, leaf_list_1):

            # keep leaf nodes in alphabetical order
            if leaf_pair[0] < leaf_pair[1]:
                result.append((leaf_level, leaf_pair[0], leaf_pair[1]))
            else:
                result.append((leaf_level, leaf_pair[1], leaf_pair[0]))

    return result
