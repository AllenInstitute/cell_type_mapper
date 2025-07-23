import copy
import h5py
import json
import numpy as np
import pathlib
import time

import cell_type_mapper.utils.gene_utils as gene_utils
import mmc_gene_mapper.mapper.species_detection as species_detection

from cell_type_mapper.utils.cloud_utils import (
    sanitize_paths)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from cell_type_mapper.file_tracker.file_tracker import (
    FileTracker)

import mmc_gene_mapper.mapper.mapper as gene_mapper_module


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


def align_query_gene_names(
        query_gene_path,
        gene_id_col=None,
        precomputed_stats_path=None,
        gene_mapper_db_path=None,
        log=None):
    """
    If map_to_ensembl is True, automatically map the gene IDs in
    query_gene_path.var.index to ENSEMBL IDs

    Return the list of gene names and the number of genes that could
    not be mapped (this will be zero of map_to_ensemble is False)

    Also return boolean indicating whether or not any genes
    were meaningfully mapped (True if a gene was mapped to
    an ENSEMBL ID; false otherwise)
    """

    result = gene_utils.get_gene_identifier_list(
        h5ad_path_list=[query_gene_path],
        gene_id_col=gene_id_col,
        duplicate_prefix=gene_utils.invalid_query_prefix()
    )

    map_genes = False
    if gene_mapper_db_path is not None:
        if precomputed_stats_path is not None:
            map_genes = True

    n_unmapped = 0
    was_changed = False
    if map_genes:

        (result,
         n_unmapped,
         was_changed) = _align_gene_names(
             precomputed_stats_path=precomputed_stats_path,
             gene_mapper_db_path=gene_mapper_db_path,
             gene_list=result,
             log=log)

    return result, n_unmapped, was_changed


def _align_gene_names(
        precomputed_stats_path,
        gene_mapper_db_path,
        gene_list,
        log):

    was_changed = False

    with h5py.File(precomputed_stats_path, 'r') as src:
        input_genes = json.loads(
            src['col_names'][()].decode('utf-8')
        )

    input_gene_data = species_detection.detect_species_and_authority(
        db_path=gene_mapper_db_path,
        gene_list=input_genes
    )

    input_authority = set(input_gene_data['authority'])

    if len(input_authority) == 1 and 'symbol' in input_authority:
        raise ValueError(
            "reference data contains gene symbols; will not be "
            "able to infer a species during gene alignment"
        )

    if 'symbol' in input_authority:
        input_authority.remove('symbol')

    if len(input_authority) != 1:
        raise ValueError(
           "Could not find single authority to map genes to; "
           f"options are {input_authority}"
        )
    dst_authority = input_authority.pop()
    dst_species = input_gene_data['species'].name

    if dst_species is None:
        msg = (
            "Could not find a species for the genes you gave:\n"
            f"First five genes:\n{result[:5]}"
        )
        raise RuntimeError(msg)

    gene_mapper = gene_mapper_module.MMCGeneMapper(
        db_path=gene_mapper_db_path
    )

    original_result = np.array(gene_list)

    raw_result = gene_mapper.map_genes(
        gene_list=gene_list,
        dst_species=dst_species,
        dst_authority=dst_authority,
        ortholog_citation='NCBI',
        log=log,
        invalid_mapping_prefix=gene_utils.invalid_query_prefix()
    )

    result = raw_result['gene_list']
    n_unmapped = 0
    for gene_name in result:
        if 'UNMAPPABLE' in gene_name:
            n_unmapped += 1

    if np.array_equal(np.array(result), original_result):
        was_changed = False
    else:
        delta = np.where(np.array(result) != original_result)[0]
        for ii in delta:
            if 'UNMAPPABLE' not in result[ii]:
                was_changed = True
                break

    return result, n_unmapped, was_changed



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


def config_from_args(input_config, cloud_safe=False):
    """
    Take args from a CLI module and return a config dict
    suitable for recording in the output file's metadata.

    Parameters
    ----------
    input_config:
        a dict. self.args from an ArgSchemaParser
    cloud_safe:
        boolean. If true, clip all file paths to only reference
        the file's name (not it's directory)
    """
    config = copy.deepcopy(input_config)
    for bad_key in ('input_json',):
        if bad_key in config:
            config.pop(bad_key)

    if cloud_safe:
        for bad_key in ('extended_result_dir', 'tmp_dir'):
            if bad_key in config:
                config.pop(bad_key)
        config = sanitize_paths(config)
    return config
