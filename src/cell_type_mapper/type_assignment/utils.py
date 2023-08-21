import h5py


def reconcile_taxonomy_and_markers(
        taxonomy_tree,
        marker_cache_path):
    """
    Do a quick check to verify the consistency
    of a taxonomy tree and a marker cache file
    (i.e. do they contain the same types)

    Parameters
    ----------
    taxonomy_tree:
        A TaxonomyTree
    marker_cache_path:
        Path to the hdf5 file containing the marker
        cache

    Returns
    -------
    If the taxonomy_tree and marker_cache_path describe
    the same taxonomy, will return (True, '')

    Else will return False and a string describing why
    the two cannot be reconciled.
    """
    parent_list = taxonomy_tree.all_parents
    valid_levels = set()
    missing_levels = set()
    missing_nodes = []
    with h5py.File(marker_cache_path, 'r') as markers:
        for parent in parent_list:
            if parent is None:
                parent_grp = 'None'
            else:
                parent_grp = f'{parent[0]}/{parent[1]}'
                if len(taxonomy_tree.children(parent[0], parent[1])) == 1:
                    # this parent only has one child; it does not matter
                    # if there are markers for it or not
                    continue

            if parent_grp not in markers:
                if parent is None:
                    missing_nodes.append(None)
                else:
                    missing_levels.add(parent[0])
                    missing_nodes.append(parent)
            elif parent is not None:
                valid_levels.add(parent[0])

    if len(missing_nodes) == 0:
        return (True, '')

    fully_missing_levels = set()
    for level in missing_levels:
        if level not in valid_levels:
            fully_missing_levels.add(level)

    # because --drop_level only accepts one level
    if len(fully_missing_levels) > 1:
        fully_missing_levels = set()

    msg = ''
    for node in missing_nodes:
        if node is None or node[0] not in fully_missing_levels:
            if node is None:
                parent_grp = 'None'
            else:
                parent_grp = f'{node[0]}/{node[1]}'
            msg += f"marker cache is missing parent '{parent_grp}'\n"

    if len(fully_missing_levels) == 1:
        bad_level = fully_missing_levels.pop()
        msg += (f"marker cache is missing all parents at level '{bad_level}'; "
                "consider running cell_type_mapper with "
                f"--drop_level '{bad_level}'")

    return (False, msg)
