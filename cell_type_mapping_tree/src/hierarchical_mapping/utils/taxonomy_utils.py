"""
    Compute the order of rows necessary such that
    cells of the same taxonomic classification are
    in contiguous blocks of rows.
"""

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
            
