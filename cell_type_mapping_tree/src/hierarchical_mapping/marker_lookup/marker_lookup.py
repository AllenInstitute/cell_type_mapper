import pathlib


def marker_lookup_from_tree_and_csv(
        taxonomy_tree,
        csv_dir):
    """
    Create a dict mapping parent node to the list of marker genes
    based on a taxonomy_tree and a directory of CSV files

    Parameters
    ----------
    taxonomy_tree:
        the TaxonomyTree defining our cell types taxonomy
    csv_dir:
        the directory containing the text files listing marker
        genes as produced by the science team's R code.

    Returns
    -------
    A dict mapping 'level/node' to the list of marker genes.

    Note
    ----
    These marker genes will be identified by whatever scheme is used
    in the CSV files. Conversion to EnsemblIDs will be left to a later
    step in the pipeline.
    """

    csv_dir = pathlib.Path(csv_dir)
    if not csv_dir.is_dir():
        raise RuntimeError(
            f"{csv_dir} is not a valid directory")

    level_to_idx = {n: ii+2
                    for ii, n in enumerate(taxonomy_tree.hierarchy)}

    parent_to_path = dict()
    parent_list = taxonomy_tree.all_parents
    for parent_node in parent_list:
        if parent_node is None:
            fname = 'marker.1.root.csv'
            parent_key = 'None'
        else:
            parent_key = f'{parent_node[0]}/{parent_node[1]}'
            children = taxonomy_tree.children(parent_node[0], parent_node[1])
            if len(children) < 2:
                continue
            level_idx = level_to_idx[parent_node[0]]
            readable_name = taxonomy_tree.label_to_name(
                level=parent_node[0],
                label=parent_node[1],
                name_key='name')
            prefix = readable_name.split()[0]

            try:
                int(prefix)
                is_int = True
            except ValueError:
                is_int = False

            if is_int:
                readable_name = readable_name.replace(f'{prefix} ', '')

            munged = readable_name.replace(' ', '+').replace('/', '__')
            fname = f'marker.{level_idx}.{munged}.csv'
        fpath = csv_dir / fname
        if not fpath.is_file():
            raise RuntimeError(f"{fname} does not exist")
        parent_to_path[parent_key] = fpath

    marker_lookup = dict()
    for parent_key in parent_to_path:
        fpath = parent_to_path[parent_key]
        gene_symbols = []
        with open(fpath, 'r') as src:
            src.readline()
            for line in src:
                symbol = line.strip().replace('"', '')

                # some of the genes in the CSV tables have the
                # EnsemblID erroneously listed as a part of the
                # symbol
                if ' ' in symbol and symbol.split()[1].startswith('ENS'):
                    symbol = symbol.split()[0]
                gene_symbols.append(symbol)

        marker_lookup[parent_key] = gene_symbols

    return marker_lookup
