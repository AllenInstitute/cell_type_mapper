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

    old_row_to_new_row = {r:ii for ii, r in enumerate(row_order)}

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

        for parent, child in zip(column_hierarchy[:-1],
                                 column_hierarchy[1:]):
            this_parent = row[parent]
            this_child = row[child]
            if this_parent not in tree[parent]:
                tree[parent][this_parent] = set()
            tree[parent][this_parent].add(this_child)

    return tree


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
