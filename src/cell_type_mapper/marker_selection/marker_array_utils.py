def _create_new_pair_lookup(only_keep_pairs):
    """
    Create new pair-to-idx lookup for case where we
    are only keeping the specified pairs
    """
    new_lookup = dict()
    for ii, pair in enumerate(only_keep_pairs):
        level = pair[0]
        node1 = pair[1]
        node2 = pair[2]
        if level not in new_lookup:
            new_lookup[level] = dict()
        if node1 not in new_lookup[level]:
            new_lookup[level][node1] = dict()
        new_lookup[level][node1][node2] = ii
    return new_lookup


def _idx_of_pair(
        taxonomy_pair_to_idx,
        level,
        node1,
        node2):
    if node1 not in taxonomy_pair_to_idx[level]:
        raise RuntimeError(
            f"{node1} not under taxonomy level {level}")
    if node2 not in taxonomy_pair_to_idx[level][node1]:
        raise RuntimeError(
            f"({level},  {node1}, {node2})\n"
            "not a valid taxonomy pair specification; try reversing "
            "node1 and node2")

    pair_idx = taxonomy_pair_to_idx[level][node1][node2]
    return pair_idx
