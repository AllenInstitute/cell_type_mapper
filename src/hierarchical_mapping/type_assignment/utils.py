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
    msg = ''
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
                msg += f"marker cache is missing parent '{parent_grp}'\n"

    if len(msg) == 0:
        return (True, msg)

    return (False, msg)
