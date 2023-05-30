from hierarchical_mapping.utils.anndata_utils import (
    read_df_from_h5ad)


def _check_config(config_dict, config_name, key_name, log):
    if isinstance(key_name, list):
        for el in key_name:
            _check_config(
                config_dict=config_dict,
                config_name=config_name,
                key_name=el,
                log=log)
    else:
        if key_name not in config_dict:
            log.error(f"'{config_name}' config missing key '{key_name}'")


def _get_query_gene_names(query_gene_path):
    var = read_df_from_h5ad(query_gene_path, 'var')
    result = list(var.index.values)
    return result
