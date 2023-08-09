import json
import pathlib
import re

from cell_type_mapper.gene_id.utils import (
    is_ensembl)

from cell_type_mapper.data.cellranger_6_lookup import (
    cellranger_6_lookup)

from cell_type_mapper.data.gene_id_lookup import (
    gene_id_lookup)


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

    missing_marker_sets = []

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
        if fpath.is_file():
            parent_to_path[parent_key] = fpath
        else:
            missing_marker_sets.append((parent_key, fpath.name))

    if len(missing_marker_sets) > 0:
        msg = ''
        for marker_set in missing_marker_sets:
            msg += f'{marker_set}\n'
        raise RuntimeError(
            f"Could not find marker sets\n{msg}")

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

    print("successfully got raw marker lookup")
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

    # create bespoke symbol-to-EnsemblID mapping that
    # uses AIBS conventions in cases where the gene symbol
    # maps to more than one EnsemblID
    all_markers = set()
    for k in raw_markers:
        all_markers = all_markers.union(set(raw_markers[k]))
    all_markers = list(all_markers)

    symbol_to_ensembl = map_aibs_gene_names(all_markers)

    result = dict()
    for k in raw_markers:
        new_markers = [symbol_to_ensembl[s] for s in raw_markers[k]]
        result[k] = new_markers

    return result


def map_aibs_gene_names(raw_gene_names):
    """
    Take a list of gene names; return a dict mapping them
    to Ensembl IDs, accounting for AIBS-specific gene
    symbol conventions
    """
    error_msg = ""

    raw_gene_names = set(raw_gene_names)
    raw_gene_names = list(raw_gene_names)
    raw_gene_names.sort()

    # Look to see if any unmapped symbols are made easier
    # by the presence of Ensembl IDs in the gene symbol
    bad_ct = 0
    bad_symbols = []
    used_ensembl = set()
    symbol_to_ensembl = dict()
    for symbol in raw_gene_names:
        ensembl = None
        if symbol in cellranger_6_lookup:
            candidates = cellranger_6_lookup[symbol]
            if len(candidates) == 1:
                ensembl = candidates[0]
        else:
            if " " in symbol:
                ensembl = symbol.split()[1]
                if not is_ensembl(ensembl):
                    ensembl = None
            if ensembl is None:
                bad_symbols.append(symbol)

        if ensembl is None:
            ensembl = f"bad_{bad_ct}"
            bad_ct += 1

        if ensembl in used_ensembl:
            other = []
            for s in symbol_to_ensembl:
                if symbol_to_ensembl[s] == ensembl:
                    other.append(s)
            error_msg += (
                f"more than one gene symbol maps to {ensembl}:\n"
                f"    {symbol} and {json.dumps(other)}\n")

        symbol_to_ensembl[symbol] = ensembl
        used_ensembl.add(ensembl)

    # final pass, attempting to see if any ambiguities have been
    # resolved by the EnsemblIDs dangling in gene symbols
    for symbol in symbol_to_ensembl:
        if is_ensembl(symbol_to_ensembl[symbol]):
            continue
        if symbol not in cellranger_6_lookup:
            continue
        candidates = cellranger_6_lookup[symbol]

        ensembl = None
        valid_candidates = []
        for c in candidates:
            if c not in used_ensembl:
                valid_candidates.append(c)
        if len(valid_candidates) > 1:
            # last ditch effort; look in gene_id_lookup
            if symbol in gene_id_lookup:
                if gene_id_lookup[symbol] not in used_ensembl:
                    ensembl = gene_id_lookup[symbol]

            if ensembl is None:
                error_msg += (
                    f"Too many possible Ensembl IDs for {symbol}\n"
                    f"    {valid_candidates}\n")

        elif len(valid_candidates) == 1:
            ensembl = valid_candidates[0]

        if ensembl is None:
            bad_symbols.append(symbol)
        else:
            symbol_to_ensembl[symbol] = ensembl
            used_ensembl.add(ensembl)

    if len(bad_symbols) > 0:
        error_msg += (
            "Could not find Ensembl IDs for\n"
            f"{json.dumps(bad_symbols, indent=2)}\n")

    if len(error_msg) > 0:
        raise RuntimeError(error_msg)
    return symbol_to_ensembl
