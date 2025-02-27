import copy
import h5py
import json
import pathlib

from cell_type_mapper.utils.utils import (
    clean_for_json,
    get_timestamp)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad
)

from cell_type_mapper.taxonomy.utils import (
    validate_taxonomy_tree,
    get_all_leaf_pairs,
    get_taxonomy_tree,
    convert_tree_to_leaves,
    get_all_pairs,
    get_child_to_parent,
    prune_tree)

from cell_type_mapper.taxonomy.data_release_utils import (
    get_tree_above_leaves,
    get_label_to_name,
    get_cell_to_cluster_alias,
    get_term_set_map)


class TaxonomyTree(object):

    def __init__(
            self,
            data):
        """
        data is a dict encoding the taxonomy tree.
        Probably, users will not instantiate this class
        directly, instead using one of the classmethods
        """
        self._data = copy.deepcopy(data)
        validate_taxonomy_tree(self._data)
        self._child_to_parent = get_child_to_parent(self._data)

        self._name_to_level = dict()
        for level in self.hierarchy:
            level_name = self.level_to_name(level)
            if level_name == level:
                continue
            if level_name not in self._name_to_level:
                self._name_to_level[level_name] = []
            self._name_to_level[level_name].append(level)

        self._name_to_node = dict()
        for level in self.hierarchy:
            self._name_to_node[level] = dict()
            for node in self.nodes_at_level(level):
                node_name = self.label_to_name(
                    level=level,
                    label=node,
                    name_key='name')
                if node_name == node:
                    continue
                if node_name not in self._name_to_node[level]:
                    self._name_to_node[level][node_name] = []
                self._name_to_node[level][node_name].append(node)

        self._child_to_parent_level = {
            child_level: parent_level
            for child_level, parent_level in zip(
                self.hierarchy[1:],
                self.hierarchy[:-1])
        }
        self._child_to_parent_level[self.hierarchy[0]] = None

    @property
    def metadata(self):
        if 'metadata' not in self._data:
            return None
        return self._data['metadata']

    @metadata.setter
    def metadata(self, value):
        if self.metadata is not None:
            raise RuntimeError(
                "Already set metadata for this TaxonomyTree"
            )
        self._data['metadata'] = value

    def __eq__(self, other):
        """
        Ignore keys 'metadata' and 'alias_mapping'
        """
        ignore_cells = self.is_equal_to(other)
        if not ignore_cells:
            return False
        for node in self.nodes_at_level(self.leaf_level):
            this_children = self.children(level=self.leaf_level, node=node)
            other_children = other.children(level=self.leaf_level, node=node)
            if set(this_children) != set(other_children):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def is_equal_to(self, other):
        """
        Compare to another taxonomy tree, only looking
        at the fields in 'hierarchy' and ignoring the
        specific cell-to-leaf assignments.

        Return True if the two taxonomies are equal;
        False otherwise.
        """
        if self.hierarchy != other.hierarchy:
            return False
        for level in self.hierarchy:
            if set(self._data[level].keys()) != set(other._data[level].keys()):
                return False

            if level == self.leaf_level:
                continue

            for node in self.nodes_at_level(level):
                this_children = self.children(level=level, node=node)
                other_children = other.children(level=level, node=node)
                if set(this_children) != set(other_children):
                    return False
        return True

    @classmethod
    def from_precomputed_stats(cls, stats_path):
        """
        Read the taxonomy tree from a precomputed stats file

        Parameters
        ----------
        stats_path:
            Path to an HDF5 precomputed_stats file (or any
            HDF5 file with a 'taxonomy_tree' dataset containing
            a JSON-serialized TaxonomyTree)
        """
        with h5py.File(stats_path, 'r') as src:
            return cls(
                data=json.loads(src['taxonomy_tree'][()].decode('utf-8')))

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
        obs = read_df_from_h5ad(h5ad_path, df_name='obs')
        result = cls.from_dataframe(
            dataframe=obs,
            column_hierarchy=column_hierarchy)

        result.metadata = {
            'factory': 'from_h5ad',
            'timestamp': get_timestamp(),
            'params': {
                'h5ad_path': str(h5ad_path.resolve().absolute()),
                'column_hierarchy': column_hierarchy}}

        return result

    @classmethod
    def from_dataframe(cls, dataframe, column_hierarchy, drop_rows=False):
        """
        Instantiate from the a dataframe (probably obs
        from an h5ad file)

        Parameters
        ----------
        dataframe:
            a pandas dataframe
        column_hierarchy:
            ordered list of levels in the taxonomy (columns in the
            obs dataframe to be read in)
        drop_rows:
            if True, replace leaf children with empty lists
        """
        data = get_taxonomy_tree(
            obs_records=dataframe.to_dict(orient='records'),
            column_hierarchy=column_hierarchy,
            drop_rows=drop_rows)

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
            hierarchy,
            do_pruning=False):
        """
        Construct a TaxonomyTree from the canonical CSV files
        encoding a taxonomy for a data release

        Parameters
        ----------
        cell_metadata_path:
            path to cell_metadata.csv; the file mapping cells to clusters
            (This can be None, in which case the taxonomy tree will have no
            data mapping cells to clusters; it will only encode the
            parent-child relationships between taxonomic nodes)
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
        do_pruning:
            A boolean. If True, remove all nodes from the tree that are
            not directly connected to the leaf level of the tree. This
            is useful, for instance, when creating a taxonomy tree based
            only on cells from a subset of a wider data release which may
            not include all of the cell types identified in the full
            taxonomy.
        """
        cluster_annotation_path = pathlib.Path(cluster_annotation_path)
        cluster_membership_path = pathlib.Path(cluster_membership_path)

        if cell_metadata_path is not None:
            cell_metadata_path = pathlib.Path(cell_metadata_path)
            cell_path_str = str(cell_metadata_path.resolve().absolute())
        else:
            cell_path_str = None

        data = dict()
        metadata = {
            'factory': 'from_data_release',
            'timestamp': get_timestamp(),
            'params': {
                'cell_metadata_path':
                    cell_path_str,
                'cluster_annotation_path':
                    str(cluster_annotation_path.resolve().absolute()),
                'cluster_membership_path':
                    str(cluster_membership_path.resolve().absolute()),
                'hierarchy': hierarchy,
                'do_pruning': do_pruning}}

        leaf_level = hierarchy[-1]

        rough_tree = get_tree_above_leaves(
            csv_path=cluster_annotation_path,
            hierarchy=hierarchy)

        hierarchy_level_map = get_term_set_map(
            csv_path=cluster_membership_path)

        data['hierarchy_mapper'] = hierarchy_level_map

        data['hierarchy'] = copy.deepcopy(hierarchy)
        for parent_level, child_level in zip(hierarchy[:-1], hierarchy[1:]):
            data[parent_level] = dict()
            for node in rough_tree[parent_level]:
                data[parent_level][node] = []
                for child in rough_tree[parent_level][node]:
                    data[parent_level][node].append(child)
            for node in data[parent_level]:
                data[parent_level][node].sort()

        # get mappings between labels and other ways
        # of referring to taxons
        cluster_to_alias = get_label_to_name(
            csv_path=cluster_membership_path,
            valid_term_set_labels=(leaf_level,),
            name_column='cluster_alias')

        alias_to_cluster_label = dict()
        for k in cluster_to_alias:
            label = k[1]
            alias = cluster_to_alias[k]
            if alias in alias_to_cluster_label:
                raise RuntimeError(
                    f"alias {alias} maps to clusters "
                    f"{label} and {alias_to_cluster_label[alias]}")
            alias_to_cluster_label[alias] = label

        label_to_name = get_label_to_name(
            csv_path=cluster_membership_path,
            valid_term_set_labels=hierarchy,
            name_column='cluster_annotation_term_name')

        # create a mapp from [level][node] to all
        # alternative naming schemes
        final_name_map = dict()
        for k in label_to_name:
            level = k[0]
            label = k[1]
            name = label_to_name[k]
            if level not in final_name_map:
                final_name_map[level] = dict()
            if label not in final_name_map[level]:
                final_name_map[level][label] = dict()
            if 'name' in final_name_map[level][label]:
                if final_name_map[level][label]['name'] != name:
                    raise RuntimeError(
                        f"level {level}, label {label} has at least "
                        f"two names: {name} and "
                        f"{final_name_map[level][label]['name']}")
            final_name_map[level][label]['name'] = name

        # add cluster aliases to final_name_map
        for k in cluster_to_alias:
            final_name_map[k[0]][k[1]]['alias'] = cluster_to_alias[k]
        data['name_mapper'] = final_name_map

        # now add leaves (referring to them by their labels)
        leaves = dict()

        if cell_metadata_path is not None:
            cell_to_alias = get_cell_to_cluster_alias(
                csv_path=cell_metadata_path)

            for cell in cell_to_alias:
                alias = cell_to_alias[cell]
                leaf = alias_to_cluster_label[alias]
                if leaf not in leaves:
                    leaves[leaf] = []
                leaves[leaf].append(cell)
            for leaf in leaves:
                leaves[leaf].sort()
        else:
            for parent in data[hierarchy[-2]].keys():
                for child in data[hierarchy[-2]][parent]:
                    leaves[child] = []

        data[hierarchy[-1]] = leaves

        if do_pruning:
            data = prune_tree(data)

        result = cls(data=data)
        result.metadata = metadata
        return result

    def to_str(self, indent=None, drop_cells=False):
        """
        Return JSON-serialized dict of this taxonomy

        If drop_cells == True, then do not serialize the mapping
        from leaf node to cells
        """
        if drop_cells:
            out_dict = copy.deepcopy(self._data)
            for leaf in out_dict[self.leaf_level]:
                out_dict[self.leaf_level][leaf] = []
        else:
            out_dict = self._data

        return json.dumps(clean_for_json(out_dict), indent=indent)

    def flatten(self):
        """
        Return a 'flattened' (i.e. 1-level) version of the taxonomy tree.
        """
        new_data = copy.deepcopy(self._data)
        if 'metadata' in new_data:
            new_data['metadata']['flattened'] = True
        new_data['hierarchy'] = [self._data['hierarchy'][-1]]
        for level in self._data['hierarchy'][:-1]:
            new_data.pop(level)
        return TaxonomyTree(data=new_data)

    def _drop_level(self, level_to_drop, allow_leaf=False):
        """
        Return a new taxonomy tree which has dropped the specified
        level from its hierarchy.

        Only allowed to drop leaf leave if allow_leaf is True
        """

        if len(self.hierarchy) == 1:
            raise RuntimeError(
                "Cannot drop a level from this tree. "
                f"It is flat. hierarchy={self.hierarchy}")

        if level_to_drop not in self.hierarchy:
            raise RuntimeError(
                f"Cannot drop level '{level_to_drop}' from this tree. "
                "That level is not in the hierarchy\n"
                f"hierarchy={self.hierarchy}")

        if not allow_leaf:
            if level_to_drop == self.leaf_level:
                raise RuntimeError(
                    f"Cannot drop level '{level_to_drop}' from this tree. "
                    "That is the leaf level.")

        new_data = copy.deepcopy(self._data)
        if 'metadata' in new_data:
            if 'dropped_levels' not in new_data['metadata']:
                new_data['metadata']['dropped_levels'] = []
            new_data['metadata']['dropped_levels'].append(level_to_drop)

        level_idx = -1
        for idx, level in enumerate(self.hierarchy):
            if level == level_to_drop:
                level_idx = idx
                break

        if level_idx == 0:
            new_data['hierarchy'].pop(0)
            new_data.pop(level_to_drop)
            return TaxonomyTree(data=new_data)

        parent_idx = level_idx - 1
        parent_level = self.hierarchy[parent_idx]

        new_parent = dict()
        for node in new_data[parent_level]:
            new_parent[node] = []
            for child in self.children(parent_level, node):
                new_parent[node] += self.children(level_to_drop, child)

        new_data.pop(level_to_drop)
        new_data[parent_level] = new_parent
        new_data['hierarchy'].pop(level_idx)
        return TaxonomyTree(data=new_data)

    def drop_level(self, level_to_drop):
        """
        Return a new taxonomy tree which has dropped the specified
        level from its hierarchy.

        Will not drop leaf level.
        """
        return self._drop_level(level_to_drop, allow_leaf=False)

    def drop_leaf_level(self):
        """
        Drop leaf level from tree.
        """
        return self._drop_level(self.leaf_level, allow_leaf=True)

    def drop_node(self, level, node):
        """
        Return a new TaxonomyTree having dropped the specified node
        and pruned the tree appropriately.

        Parameters
        ----------
        level:
            A string. The level of the node to drop
        node:
            A string. The node to be dropped

        Parameters
        ----------
        A new TaxonomyTree
        """
        if level not in self.hierarchy:
            raise RuntimeError(
                f"Level {level} not present in tree"
            )
        if node not in self.nodes_at_level(level):
            raise RuntimeError(
                f"Node {node} not present at level {level}"
            )
        new_data = copy.deepcopy(self._data)
        if 'metadata' in new_data:
            if 'dropped_nodes' not in new_data['metadata']:
                new_data['metadata']['dropped_nodes'] = []
            new_data['metadata']['dropped_nodes'].append(
                {'level': level, 'node': node}
            )

        for leaf in self.as_leaves[level][node]:
            new_data[self.leaf_level].pop(leaf)
        new_data = prune_tree(new_data)
        return TaxonomyTree(data=new_data)

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
                f"valid levels are:\n {self.hierarchy}")
        result = list(self._data[this_level].keys())
        result.sort()
        return result

    def parent_level(self, level):
        """
        Return the level that is directly above the specified
        level in the hierarchy
        """
        return self._child_to_parent_level[level]

    def parents(self, level, node):
        """
        return a dict listing all the ancestors of
        (level, node)
        """
        this = dict()
        hierarchy_idx = None
        for idx in range(len(self.hierarchy)):
            if self.hierarchy[idx] == level:
                hierarchy_idx = idx
                break
        for parent_level_idx in range(hierarchy_idx-1, -1, -1):
            current = self.hierarchy[parent_level_idx]
            if len(this) == 0:
                this[current] = self._child_to_parent[level][node]
            else:
                prev = self.hierarchy[parent_level_idx+1]
                prev_node = this[prev]
                this[current] = self._child_to_parent[prev][prev_node]
        return this

    def children(self, level, node):
        """
        Return the immediate children of the specified node
        """
        if level is None and node is None:
            return list(self._data[self.hierarchy[0]].keys())
        if level not in self._data.keys():
            raise RuntimeError(
                f"{level} is not a valid level\ntry {self.hierarchy}")
        if node not in self._data[level]:
            raise RuntimeError(f"{node} not a valid node at level {level}")
        result = list(self._data[level][node])
        result.sort()
        return result

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
        result = list(self._data[self.leaf_level].keys())
        result.sort()
        return result

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
            for node in self.nodes_at_level(level):
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

    def label_to_name(self, level, label, name_key='name'):
        """
        Parameters
        ----------
        level:
            the level in the hierarchy
        label:
            the machine readable label of the node
        name_key:
            the alternative name to return (e.g. 'name' or 'alias')

        Returns
        -------
        The human readable name

        Note
        ----
        if mapping is impossible, just return label
        """
        if 'name_mapper' not in self._data:
            return label
        name_mapper = self._data['name_mapper']
        if level not in name_mapper:
            return label
        if label not in name_mapper[level]:
            return label
        if name_key not in name_mapper[level][label]:
            return label
        return name_mapper[level][label][name_key]

    def name_to_node(self, level, node):
        """
        Map a level, node pair from human-readable to unique, machine-readable
        values.

        Parameters
        ----------
        level:
            the level of the node being mapped. Either human-readable or
            machine-readable
        node:
            the node being mapped. Either human-readable or machine-readable

        Returns
        -------
        A tuple of strings denoting the machine-readable (level, node) label
        pair

        Notes
        -----
        Raise an exception if:
            level does not exist in this taxonomy
            node does not exist in this taxonomy
            level maps to many labels
            node maps to many labels (under level)
        """
        input_level = level
        if level not in self.hierarchy:
            level = self.name_to_level(level)

        if level not in self.hierarchy:
            msg = f'{input_level} is not a valid level in this taxonomy'
            raise RuntimeError(msg)

        if node in self._data[level]:
            return (level, node)

        if node not in self._name_to_node[level]:
            msg = f'({input_level}, {node}) not a valid node in this taxonomy'
            raise RuntimeError(msg)

        candidates = self._name_to_node[level][node]
        if len(candidates) > 1:
            msg = f"""
            ({input_level}, {node}) maps to many nodes: {candidates}
            """
            raise RuntimeError(msg)

        return (level, candidates[0])

    def level_to_name(self, level_label):
        """
        Map the label for a hierarchy level to its name.
        If no mapper exists (or the level_label is unknown)
        just return level_label
        """
        if 'hierarchy_mapper' not in self._data:
            return level_label
        if level_label not in self._data['hierarchy_mapper']:
            return level_label
        return self._data['hierarchy_mapper'][level_label]

    def name_to_level(self, level_name):
        """
        Map human readable level_name to unique label for the level
        (if possible).

        If level_name is not a valid key in self._name_to_level,
        raise an exception.

        If there are more than one possible mappings for level_name,
        raise an exception.
        """
        if level_name in self.hierarchy:
            return level_name

        if level_name not in self._name_to_level:
            msg = f"{level_name} is not a valid level in this taxonomy"
            raise RuntimeError(msg)
        candidates = self._name_to_level[level_name]

        if len(candidates) > 1:
            msg = f"""
            {level_name} maps to many levels: {candidates}
            """
            raise RuntimeError(msg)

        return candidates[0]

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

    def backfill_assignments(self, assignments):
        """
        Take a list of cell type assignments and backfill
        any levels that were dropped or flattened away
        when the assignment was made. Data beyond the assignment
        will be copied directly from the child node of the
        level being backfilled.

        Parameters
        ----------
        assignments:
            A list of dicts representing the assignments
            being backfilled. Each dict looks like
            {'cell_id': 12345,
             'level_1': {'assignment': ...},
             'level_2': {'assignment': ...},
             ...
             'leaf_level': {'assignment': ...}}

        Returns
        -------
        assignments:
            Updated with any levels that are missing. The assignments
            for these levels will be inferred from their child
            levels (if present).

            Other data will be copied directly from the child levels,
            unless that other data is keyed with runner_up_*

        Notes
        -----
        In addition to being returned, assignments will be altered in place.
        """
        reverse_hierarchy = copy.deepcopy(self.hierarchy)
        reverse_hierarchy.reverse()
        for child_level, parent_level in zip(reverse_hierarchy[:-1],
                                             reverse_hierarchy[1:]):

            for cell in assignments:

                if parent_level in cell:
                    continue

                if child_level not in cell:
                    continue

                this_child = cell[child_level]['assignment']
                new_data = copy.deepcopy(cell[child_level])
                new_keys = list(new_data.keys())
                for k in new_keys:
                    if k.startswith('runner_up'):
                        new_data.pop(k)
                this_parent = self._child_to_parent[child_level][this_child]
                new_data['assignment'] = this_parent

                # mark this level as being backfilled
                new_data['directly_assigned'] = False

                cell[parent_level] = new_data

        return assignments
