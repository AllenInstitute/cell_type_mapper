import pathlib
import time

from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from hierarchical_mapping.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from hierarchical_mapping.file_tracker.file_tracker import (
    FileTracker)

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


def create_precomputed_stats_file(
        precomputed_config,
        log,
        file_tracker,
        tmp_dir):
    """
    Create the precomputed stats file (if necessary)

    Parameters
    ----------
    precomputed_config:
        Dict containing input config for precomputed stats
    log:
        CommandLogger to log messages while running
    file_tracker:
        The FileTracker used to map between real and tmp
        locations for files
    tmp_dir:
        The global tmp dir for this CLI run
    """

    log.info("creating precomputed stats file")

    reference_tracker = FileTracker(
        tmp_dir=tmp_dir,
        log=log)

    reference_path = pathlib.Path(
        precomputed_config['reference_path'])

    reference_tracker.add_file(
        reference_path,
        input_only=True)

    if 'column_hierarchy' in precomputed_config:
        column_hierarchy = precomputed_config['column_hierarchy']
        taxonomy_tree = None
    else:
        taxonomy_tree = TaxonomyTree.from_json_file(
            json_path=precomputed_config['taxonomy_tree'])
        column_hierarchy = None

    t0 = time.time()
    precompute_summary_stats_from_h5ad(
        data_path=reference_tracker.real_location(reference_path),
        column_hierarchy=column_hierarchy,
        taxonomy_tree=taxonomy_tree,
        output_path=file_tracker.real_location(precomputed_config['path']),
        rows_at_a_time=10000,
        normalization=precomputed_config['normalization'])
    log.benchmark(msg="precomputing stats",
                  duration=time.time()-t0)
