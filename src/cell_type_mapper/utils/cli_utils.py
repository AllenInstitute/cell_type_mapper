import copy
import h5py
import json
import numpy as np
import pathlib
import time

import cell_type_mapper.utils.gene_utils as gene_utils

from cell_type_mapper.utils.cloud_utils import (
    sanitize_paths)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from cell_type_mapper.file_tracker.file_tracker import (
    FileTracker)


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

    Return a dict of metadata explaining the mapping that was done
    and the backing data used to do it.
    """

    result = gene_utils.get_gene_identifier_list(
        h5ad_path_list=[query_gene_path],
        gene_id_col=gene_id_col,
        duplicate_prefix=f'UNMAPPABLE.{gene_utils.duplicated_query_prefix()}',
        log=log
    )

    metadata = dict()

    map_genes = False
    if gene_mapper_db_path is not None:
        if precomputed_stats_path is not None:
            map_genes = True
    else:
        if log is not None:
            log.info(
                "No gene_mapper_db provided. Assuming that query "
                "genes have already been mapped to the same "
                "species/authority as reference genes.",
                to_stdout=True
            )

    was_changed = False
    if map_genes:

        if log is not None:
            log.info(
                "***Checking to see if we need to map query genes onto "
                "reference dataset"
            )

        (result,
         was_changed,
         metadata) = _align_query_gene_names(
             precomputed_stats_path=precomputed_stats_path,
             gene_mapper_db_path=gene_mapper_db_path,
             gene_list=result,
             log=log)

        if log is not None:
            log.info(
                "***Mapping of query genes to reference dataset complete"
            )

    n_unmapped = 0
    for gene_name in result:
        if 'UNMAPPABLE' in gene_name:
            n_unmapped += 1
    return result, n_unmapped, was_changed, metadata


def _align_query_gene_names(
        precomputed_stats_path,
        gene_mapper_db_path,
        gene_list,
        log):
    """
    This function will actually perform any gene mapping that
    needs to be done in order to align the input gene identifiers
    to the reference gene identifiers.

    Returns
    -------
    result -- list of aligned gene identifiers
    was_changed -- boolean indicating if any meaningful mapping happened
                   (this will be false if all genes were unmapped)
    metadata -- a dict recording the mapping done and citations used
    """
    was_changed = False

    with h5py.File(precomputed_stats_path, 'r') as src:
        reference_genes = json.loads(
            src['col_names'][()].decode('utf-8')
        )

    reference_gene_data = species_detection.detect_species_and_authority(
        db_path=gene_mapper_db_path,
        gene_list=reference_genes
    )

    # not obvious what species to map to;
    # cannot perform mapping
    if reference_gene_data['species'] is None:
        if log is not None:
            log.info(
                "Could not determine species for reference data. "
                "Going to assume that query genes are already "
                "mapped to the same species/authority as the "
                "reference data",
                to_stdout=True
            )
        return gene_list, False, dict()

    dst_species = reference_gene_data['species'].name
    if log is not None:
        log.info(
            "Reference data belongs to species "
            f"{reference_gene_data['species']}"
        )

    reference_authority = set(reference_gene_data['authority'])

    if len(reference_authority) == 1 and 'symbol' in reference_authority:
        msg = (
            "reference data contains gene symbols; will not be "
            "able to infer a species during gene alignment"
        )
        if log is not None:
            log.error(msg)
        else:
            raise ValueError(msg)

    if 'symbol' in reference_authority:
        reference_authority.remove('symbol')

    if len(reference_authority) != 1:
        msg = (
           "Could not find single authority to map genes to; "
           f"options are {reference_authority}"
        )
        if log is not None:
            log.error(msg)
        else:
            raise ValueError(msg)

    dst_authority = reference_authority.pop()
    if log is not None:
        log.info(
            f"Reference genes are from authority '{dst_authority}'"
        )

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

    if np.array_equal(np.array(result), original_result):
        was_changed = False
    else:
        delta = np.where(np.array(result) != original_result)[0]
        for ii in delta:
            if 'UNMAPPABLE' not in result[ii]:
                was_changed = True
                break

    metadata = {
        'provenance': raw_result['metadata'],
        'mapping': {g0: g1 for g0, g1 in zip(gene_list, result)}
    }

    return result, was_changed, metadata


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
