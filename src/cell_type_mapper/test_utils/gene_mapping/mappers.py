import json
import pathlib

import cell_type_mapper


def get_mouse_gene_id_mapping():
    return _get_mapper('mouse_gene_id_lookup.json')


def get_human_gene_id_mapping():
    return _get_mapper('human_gene_id_lookup.json')


def get_cellranger_gene_id_mapping():
    return _get_mapper('cellranger_6_lookup.json')


def _get_mapper(file_name):
    """
    Return the gene mapping specified by file_name
    as a dict
    """
    module_path = pathlib.Path(cell_type_mapper.__file__)
    data_dir = module_path.parent / 'test_utils/gene_mapping'
    with open(data_dir / file_name, 'rb') as src:
        mapping = json.load(src)
    return mapping['mapping']
