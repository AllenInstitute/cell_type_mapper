import copy
import json
import pathlib

from hierarchical_mapping.utils.utils import (
    json_clean_dict,
    get_timestamp)

from hierarchical_mapping.taxonomy.utils import (
    validate_taxonomy_tree,
    get_all_leaf_pairs,
    get_taxonomy_tree_from_h5ad,
    convert_tree_to_leaves,
    get_all_pairs)

from hierarchical_mapping.taxonomy.data_release_utils import (
    get_tree_above_leaves,
    get_alias_mapper,
    get_cell_to_cluster_alias)


class TaxonomyTree(object):

    def __init__(
            self,
            data):
        """
        data is a dict encoding the taxonomy tree.
        Probably, users will not instantiate this class
        directly, instead using one of the classmethods

        alias_mapping maps a (hierarchy_level, alias) tuple
        to a human-readable label (this is optional)
        """
        self._data = copy.deepcopy(data)
        validate_taxonomy_tree(self._data)

    def __eq__(self, other):
        """
        Ignore keys 'metadata' and 'alias_mapping'
        """
        these_keys = set(self._data.keys())
        other_keys = set(other._data.keys())
        bad_keys = {'metadata', 'alias_mapping'}
        these_keys -= bad_keys
        other_keys -= bad_keys
        if these_keys != other_keys:
            return False
        for k in these_keys:
            if self._data[k] != other._data[k]:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def from_h5ad(cls, h5ad_path, column_hierarchy):
        """
        Instantiate from the obs dataframe of an h5ad file.

        Parameters
        ----------
        h5ad_path:
            path to the h5ad file
        column_hierarchy:
            ordered list of levels in the taxonomy (columns in the
            obs dataframe to be read in)
        """
        h5ad_path = pathlib.Path(h5ad_path)
        data = get_taxonomy_tree_from_h5ad(
            h5ad_path=h5ad_path,
            column_hierarchy=column_hierarchy)

        data['metadata'] = {
            'factory': 'from_h5ad',
            'timestamp': get_timestamp(),
            'params': {
                'h5ad_path': str(h5ad_path.resolve().absolute()),
                'column_hierarchy': column_hierarchy}}

        return cls(data=data)

    @classmethod
    def from_str(cls, serialized_dict):
        """
        Instantiate from a JSON serialized dict
        """
        return cls(
            data=json.loads(serialized_dict))

    @classmethod
    def from_json_file(cls, json_path):
        """
        Instantiate from a file containing the JSON-serialized
        tree
        """
        return cls(
            data=json.load(open(json_path, 'rb')))

    @classmethod
    def from_data_release(
            cls,
            cell_metadata_path,
            cluster_annotation_path,
            cluster_membership_path,
            hierarchy):
        """
        Construct a TaxonomyTree from the canonical CSV files
        encoding a taxonomy for a data release

        Parameters
        ----------
        cell_metadata_path:
            path to cell_metadata.csv; the file mapping cells to clusters
        cluster_annotation_path:
            path to cluster_annotation_term.csv; the file containing
            parent-child relationships
        cluster_membership_path:
            path to cluster_to_cluster_annotation_membership.csv;
            the file containing the mapping between cluster labels
            and aliases
        hierarchy:
            list of term_set labels (*not* aliases) in the hierarchy
            from most gross to most fine
        """
        cell_metadata_path = pathlib.Path(cell_metadata_path)
        cluster_annotation_path = pathlib.Path(cluster_annotation_path)
        cluster_membership_path = pathlib.Path(cluster_membership_path)

        data = dict()
        data['metadata'] = {
            'factory': 'from_data_release',
            'timestamp': get_timestamp(),
            'params': {
                'cell_metadata_path':
                    str(cell_metadata_path.resolve().absolute()),
                'cluster_annotation_path':
                    str(cluster_annotation_path.resolve().absolute()),
                'cluster_membership_path':
                    str(cluster_membership_path.resolve().absolute()),
                'hierarchy': hierarchy}}

        leaf_level = hierarchy[-1]

        cell_to_alias = get_cell_to_cluster_alias(
            csv_path=cell_metadata_path)

        rough_tree = get_tree_above_leaves(
            csv_path=cluster_annotation_path,
            hierarchy=hierarchy)

        alias_map = get_alias_mapper(
            csv_path=cluster_membership_path,
            term_set_label=leaf_level)

        data['hierarchy'] = copy.deepcopy(hierarchy)
        for parent_level, child_level in zip(hierarchy[:-1], hierarchy[1:]):
            data[parent_level] = dict()
            for node in rough_tree[parent_level]:
                data[parent_level][node] = []
                for child in rough_tree[parent_level][node]:
                    if child_level == leaf_level:
                        child = alias_map[(child_level, child)]
                    data[parent_level][node].append(child)
            for node in data[parent_level]:
                data[parent_level][node].sort()

        # now add leaves
        leaves = dict()
        for cell in cell_to_alias:
            leaf = cell_to_alias[cell]
            if leaf not in leaves:
                leaves[leaf] = []
            leaves[leaf].append(cell)
        for leaf in leaves:
            leaves[leaf].sort()
        data[hierarchy[-1]] = leaves

        # save a way to go from cluster alias back
        # to cluster label
        reverse_alias_mapping = dict()
        for node in alias_map:
            if node[1] in reverse_alias_mapping:
                raise RuntimeError(
                    f"{node[1]} occurs in alias map twice")
            alias = alias_map[node]
            reverse_alias_mapping[alias] = node[1]

        data['alias_mapping'] = reverse_alias_mapping

        return cls(data=data)

    def to_str(self, indent=None):
        """
        Return JSON-serialized dict of this taxonomy
        """
        return json.dumps(json_clean_dict(self._data), indent=indent)

    @property
    def hierarchy(self):
        return copy.deepcopy(self._data['hierarchy'])

    def nodes_at_level(self, this_level):
        """
        Return a list of all valid nodes at the specified level
        """
        if this_level not in self._data:
            raise RuntimeError(
                f"{this_level} is not a valid level in this taxonomy;\n"
                f"valid levels are:\n {self.valid_levels}")
        return list(self._data[this_level].keys())

    def children(self, level, node):
        """
        Return the immediate children of the specified node
        """
        if level is None and node is None:
            return list(self._data[self.hierarchy[0]].keys())
        if level not in self._data.keys():
            raise RuntimeError(
                f"{level} is not a valid level\ntry {self.valid_levels}")
        if node not in self._data[level]:
            raise RuntimeError(f"{node} not a valid node at level {level}")
        return list(self._data[level][node])

    @property
    def leaf_level(self):
        """
        Return the leaf level of this taxonomy
        """
        return self._data['hierarchy'][-1]

    @property
    def all_leaves(self):
        """
        List of valid leaf names
        """
        return list(self._data[self.leaf_level].keys())

    @property
    def n_leaves(self):
        return len(self.all_leaves)

    @property
    def as_leaves(self):
        """
        Return a Dict structured like
            level ('class', 'subclass', 'cluster', etc.)
                -> node1 (a node on that level of the tree)
                    -> list of leaf nodes making up that node
        """
        return convert_tree_to_leaves(self._data)

    @property
    def siblings(self):
        """
        Return all pairs of nodes that are on the same level
        """
        return get_all_pairs(self._data)

    @property
    def leaf_to_cells(self):
        """
        Return the lookup from leaf name to cells in the
        cell by gene file
        """
        return copy.deepcopy(self._data[self.leaf_level])

    @property
    def all_parents(self):
        """
        Return a list of all (level, node) tuples indicating
        valid parents in this taxonomy
        """
        parent_list = [None]
        for level in self._data['hierarchy'][:-1]:
            for node in self._data[level]:
                parent = (level, node)
                parent_list.append(parent)
        return parent_list

    def rows_for_leaf(self, leaf_node):
        """
        Return the list of rows associated with the specified
        leaf node.
        """
        if leaf_node not in self._data[self.leaf_level]:
            raise RuntimeError(
                f"{leaf_node} is not a valid {self.leaf_level} "
                "in this taxonomy")
        return self._data[self.leaf_level][leaf_node]

    def alias_to_label(self, alias):
        """
        Map from cluster alias back to label (if appropriate)

        If no mapping exists, just return input alias
        """
        if 'alias_mapping' not in self._data:
            return alias

        if alias not in self._data['alias_mapping']:
            raise RuntimeError(
                "Do not have a label associated with alias: "
                f"'{alias}'")
        return self._data['alias_mapping'][alias]

    def leaves_to_compare(
            self,
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
        parent_node:
           Either None or a (level, node) tuple specifying
           the parent whose children we are choosing between
           (None means we are at the root level)

        Returns
        -------
        A list of (level, leaf_node1, leaf_node2) tuples indicating
        the leaf nodes that need to be compared.
        """
        if parent_node is not None:
            this_level = parent_node[0]
            this_node = parent_node[1]
            if this_level not in self._data['hierarchy']:
                raise RuntimeError(
                    f"{this_level} is not a valid level in this "
                    "taxonomy\n valid levels are:\n"
                    f"{self._data['hierarchy']}")
            this_level_lookup = self._data[this_level]
            if this_node not in this_level_lookup:
                raise RuntimeError(
                    f"{this_node} is not a valid node at level "
                    f"{this_level} of this taxonomy\nvalid nodes ")

        result = get_all_leaf_pairs(
            taxonomy_tree=self._data,
            parent_node=parent_node)
        return result
