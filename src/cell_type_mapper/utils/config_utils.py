import pathlib


def patch_child_to_parent(
        child_to_parent,
        do_search=True):
    """
    Take a dict mapping 'child' files to 'parent' files.
    Search through the pairs. In any case where the 'child'
    file does not exist, look for a file with the exact same
    name in the directory containing the 'parent' file and
    modify the dict to point to that file, if it exists.

    Parameters
    ----------
    child_to_parent:
        Dict mapping path to child files to path to
        parent files.
    do_search:
        A boolean. Only actually search for the alternative
        file if do_search is True. If not and the child
        path is missing, add the parent, child pair to
        missing_pairs and move on.

    Returns
    -------
    child_to_parent:
        Modified as needed. Keys and values are now pathlib.Path

    missing_pairs:
        List of (parent, child) tuples indicating cases
        where the child file was missing.
    """
    new_lookup = dict()
    missing_pairs = []
    for child in child_to_parent:
        parent = pathlib.Path(child_to_parent[child])
        child = pathlib.Path(child)

        if child.is_file():
            new_lookup[child] = parent
            continue

        found_it = False
        if do_search:
            alt_path = parent.parent / child.name
            if alt_path.is_file():
                new_lookup[alt_path] = parent
                found_it = True

        if not found_it:
            missing_pairs.append(
                 (
                  str(parent.resolve().absolute()),
                  str(child.resolve().absolute())
                 )
            )

    return new_lookup, missing_pairs
