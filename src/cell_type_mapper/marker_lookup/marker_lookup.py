import pathlib
import re


from cell_type_mapper.gene_id.gene_id_mapper import (
    GeneIdMapper)

from cell_type_mapper.data.aibs_symbol_mapping import (
    aibs_symbol_mapping)


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

    int_re = re.compile('[0-9]+')
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
            if len(int_re.findall(prefix)) > 0:
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
                gene_symbols.append(symbol)

        marker_lookup[parent_key] = gene_symbols

    return marker_lookup


def map_aibs_marker_lookup(
        raw_markers):
    """
    Translate marker genes named in raw_markers to Ensembl IDs
    using both canonical mappings and AIBS internal mapping.

    raw_markers is a dict mapping parent nodes to lists
    of marker identifiers.

    return a similar dict, with the names of the marker genes
    mapped to Ensembl IDs.
    """
    gene_id_mapper = GeneIdMapper.from_default()

    # create bespoke symbol-to-EnsemblID mapping that
    # uses AIBS conventions in cases where the gene symbol
    # maps to more than one EnsemblID
    all_markers = set()
    for k in raw_markers:
        all_markers = all_markers.union(set(raw_markers[k]))
    all_markers = list(all_markers)
    all_markers.sort()
    first_pass = gene_id_mapper.map_gene_identifiers(
        gene_id_list=all_markers)

    used_ensembl = set()
    symbol_to_ensembl = dict()
    for symbol, ensembl in zip(all_markers, first_pass):

        if not gene_id_mapper._is_ensembl(ensembl):
            if symbol in aibs_symbol_mapping:
                ensembl = aibs_symbol_mapping[symbol]
            elif " " in symbol:
                ensembl = symbol.split()[1]
            else:
                raise RuntimeError(
                    f"cannot map gene symbol {symbol} to EnsemblID")

        if not gene_id_mapper._is_ensembl(ensembl):
            raise RuntimeError(
                f"could not find EnsemblID for gene_symbol {symbol}; "
                f"best guess: {ensembl}")

        if ensembl in used_ensembl:
            raise RuntimeError(
                f"more than one gene symbol maps to {ensembl}")

        symbol_to_ensembl[symbol] = ensembl
        used_ensembl.add(ensembl)

    result = dict()
    for k in raw_markers:
        new_markers = [symbol_to_ensembl[s] for s in raw_markers[k]]
        result[k] = new_markers

    return result
