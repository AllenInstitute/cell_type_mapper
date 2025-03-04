import h5py
import numbers
import numpy as np


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


def validate_bootstrap_factor_lookup(
        bootstrap_factor_lookup,
        taxonomy_tree,
        log=None):
    """
    Check that the bootstrap_factor_lookup contains
    all of the levels it needs to specify for a given
    taxonomy_tree

    Parameters
    ----------

    bootstrap_factor_lookup:
        A dict mapping the levels in taxonomy_tree.hierarchy to
        fractions (<=1.0) by which to sampel the marker gene set
        at each bootstrapping iteration

    taxonomy_tree:
        instance of
        cell_type_mapper.taxonomty.taxonomy_tree.TaxonomyTree
        encoding the taxonomy tree

    log:
        Optional logging object for tracking errors

    Returns
    -------
    None. Just raises an exception if something is wrong
    with bootstrap_factor_lookup
    """

    if not isinstance(bootstrap_factor_lookup, dict):
        msg = (
            "bootstrap_factor_lookup is not dict; is "
            f"{type(bootstrap_factor_lookup)}"
        )
        if log is not None:
            log.error(msg)
        else:
            raise RuntimeError(msg)

    msg = ""

    if 'None' not in bootstrap_factor_lookup:
        msg += "bootstrap_factor_lookup missing level 'None'"
    for level in taxonomy_tree.hierarchy[:-1]:
        if level not in bootstrap_factor_lookup:
            msg += f"bootstrap_factor_lookup missing level '{level}'"

    eps = 1.0e-6
    for k in bootstrap_factor_lookup:
        val = bootstrap_factor_lookup[k]
        if not isinstance(val, numbers.Number):
            msg += (
                f"bootstrap_factor_lookup['{k}'] = {val}; "
                "not a number."
            )
        else:
            if val < eps or val-1.0 > eps:
                msg += (
                    f"bootstrap_factor_lookup['{k}'] = {val} not "
                    ">0.0 and <=1.0"
                )

    if len(msg) > 0:
        if log is not None:
            log.error(msg)
        else:
            raise RuntimeError(msg)


def infer_election(
        votes,
        corr_sum,
        reference_types,
        reference_level,
        inference_level,
        taxonomy_tree):
    """
    Infer election results for a level in a cell type taxonomy
    based on election carried out at one of its child levels.

    Parameters
    ----------
    votes:
        (n_cells, n_cell_types) array of votes
    corr_sum:
        (n_cells, n_cell_types) array of correlation sums
    reference_types:
        (n_cell_types,) array of labels for cell types
    reference_level:
        level in the taxonomy at which the election was actually carried out
    inference_level:
        level in the taxonomy we want to infer
    taxonomy_tree:
        a TaxonomyTree

    Returns
    -------
    votes
    corr_sum
    reference_types

    for the inferred level
    """
    child_to_parent = {
        child: taxonomy_tree.parents(
                   level=reference_level,
                   node=child)[inference_level]
        for child in np.unique(reference_types)
    }

    parent_types = np.unique(list(child_to_parent.values()))
    parent_idx_lookup = {p: idx for idx, p in enumerate(parent_types)}
    child_to_parent_idx = np.array([
        parent_idx_lookup[child_to_parent[child]]
        for child in reference_types
    ])

    new_votes = np.zeros((votes.shape[0], len(parent_types)), dtype=float)
    new_corr = np.zeros((votes.shape[0], len(parent_types)), dtype=float)
    for child_idx, parent_idx in enumerate(child_to_parent_idx):
        new_votes[:, parent_idx] += votes[:, child_idx]
        new_corr[:, parent_idx] += corr_sum[:, child_idx]

    return new_votes, new_corr, parent_types
