"""
This is an early prototype of an end-to-end hierarchical mapping job
(i.e. one that discovers its own marker genes on the fly) which some
unit tests use.
"""
import argparse
import h5py
import json
import numbers
import numpy as np
import os
import pathlib
import tempfile
import time
import traceback

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

import cell_type_mapper.utils.gene_utils as gene_utils

from cell_type_mapper.file_tracker.file_tracker import (
    FileTracker)

from cell_type_mapper.cli.cli_log import (
    CommandLog)

from cell_type_mapper.utils.cli_utils import (
    _check_config
)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.diff_exp.markers import (
    find_markers_for_all_taxonomy_pairs)

from cell_type_mapper.type_assignment.marker_cache_v2 import (
    create_marker_cache_from_reference_markers,
    serialize_markers)

from cell_type_mapper.type_assignment.election_runner import (
    run_type_assignment_on_h5ad,
    collate_hierarchical_mappings)

from cell_type_mapper.utils.cli_utils import (
    create_precomputed_stats_file)


def run_mapping(config, output_path, log_path=None):

    log = CommandLog()

    if 'tmp_dir' not in config:
        raise RuntimeError("did not specify tmp_dir")

    tmp_dir = tempfile.mkdtemp(
        dir=config['tmp_dir'],
        prefix='cell_type_mapper_')

    output = dict()

    output_path = pathlib.Path(output_path)
    if log_path is not None:
        log_path = pathlib.Path(log_path)

    # check validity of output_path and log_path
    for pth in (output_path, log_path):
        if pth is not None:
            if not pth.exists():
                try:
                    with open(pth, 'w') as out_file:
                        out_file.write('junk')
                    pth.unlink()
                except FileNotFoundError:
                    raise RuntimeError(
                        "unable to write to "
                        f"{pth.resolve().absolute()}")

    try:
        type_assignment = _run_mapping(
            config=config,
            tmp_dir=tmp_dir,
            log=log)
        output["results"] = type_assignment["assignments"]
        output["marker_genes"] = type_assignment["marker_genes"]
        log.info("RAN SUCCESSFULLY")
    except Exception:
        traceback_msg = "an ERROR occurred ===="
        traceback_msg += f"\n{traceback.format_exc()}\n"
        log.add_msg(traceback_msg)
        raise
    finally:
        _clean_up(tmp_dir)
        log.info("CLEANING UP")
        if log_path is not None:
            log.write_log(log_path)
        output["config"] = config
        output["log"] = log.log
        with open(output_path, "w") as out_file:
            out_file.write(json.dumps(output, indent=2))


def _run_mapping(config, tmp_dir, log):

    t0 = time.time()
    file_tracker = FileTracker(
        tmp_dir=tmp_dir,
        log=log)

    _validate_config(
            config=config,
            file_tracker=file_tracker,
            log=log)

    reference_marker_config = config["reference_markers"]
    precomputed_config = config["precomputed_stats"]
    query_marker_config = config["query_markers"]
    type_assignment_config = config["type_assignment"]

    query_loc = file_tracker.real_location(
        config['query_path'])

    log.benchmark(msg="validating config and copying data",
                  duration=time.time()-t0)

    # ========= precomputed stats =========

    precomputed_loc = file_tracker.real_location(precomputed_config['path'])
    precomputed_path = precomputed_config['path']
    if file_tracker.file_exists(precomputed_path):
        log.info(f"using {precomputed_loc} for precomputed_stats")
    else:
        create_precomputed_stats_file(
            precomputed_config=precomputed_config,
            log=log,
            file_tracker=file_tracker,
            tmp_dir=tmp_dir)

    log.info(f"reading taxonomy_tree from {precomputed_loc}")
    with h5py.File(precomputed_loc, "r") as in_file:
        taxonomy_tree = TaxonomyTree.from_str(
            serialized_dict=in_file["taxonomy_tree"][()].decode("utf-8"))

    # ========= reference marker cache =========

    reference_marker_path = reference_marker_config["path"]
    reference_marker_tmp = file_tracker.real_location(reference_marker_path)

    if file_tracker.file_exists(reference_marker_path):
        log.info(f"using {reference_marker_tmp} for reference markers")
    else:
        log.info("creating reference marker file")

        marker_tmp = tempfile.mkdtemp(
            dir=tmp_dir,
            prefix='reference_marker_')

        t0 = time.time()
        find_markers_for_all_taxonomy_pairs(
            precomputed_stats_path=precomputed_loc,
            taxonomy_tree=taxonomy_tree,
            output_path=reference_marker_tmp,
            tmp_dir=marker_tmp,
            n_processors=reference_marker_config['n_processors'],
            max_gb=reference_marker_config['max_gb'])

        log.benchmark(msg="finding reference markers",
                      duration=time.time()-t0)

        _clean_up(marker_tmp)
    # ========= query marker cache =========

    query_marker_tmp = pathlib.Path(
        mkstemp_clean(dir=tmp_dir,
                      prefix='query_marker_',
                      suffix='.h5'))

    t0 = time.time()
    create_marker_cache_from_reference_markers(
        output_cache_path=query_marker_tmp,
        input_cache_path=reference_marker_tmp,
        query_gene_names=_legacy_get_query_gene_names(query_loc)[0],
        taxonomy_tree=taxonomy_tree,
        n_per_utility=query_marker_config['n_per_utility'],
        n_processors=query_marker_config['n_processors'],
        behemoth_cutoff=5000000)
    log.benchmark(msg="creating query marker cache",
                  duration=time.time()-t0)

    # ========= type assignment =========

    t0 = time.time()
    bootstrap_factor_lookup = {
        level: type_assignment_config['bootstrap_factor']
        for level in taxonomy_tree.hierarchy
    }
    bootstrap_factor_lookup['None'] = type_assignment_config[
                                            'bootstrap_factor']

    rng = np.random.default_rng(type_assignment_config['rng_seed'])

    result = run_type_assignment_on_h5ad(
        query_h5ad_path=query_loc,
        precomputed_stats_path=precomputed_loc,
        marker_gene_cache_path=query_marker_tmp,
        taxonomy_tree=taxonomy_tree,
        n_processors=type_assignment_config['n_processors'],
        chunk_size=type_assignment_config['chunk_size'],
        bootstrap_factor_lookup=bootstrap_factor_lookup,
        bootstrap_iteration=type_assignment_config['bootstrap_iteration'],
        rng=rng,
        normalization=type_assignment_config['normalization'],
        tmp_dir=tmp_dir)

    result = collate_hierarchical_mappings(result)

    log.benchmark(msg="assigning cell types",
                  duration=time.time()-t0)

    # ========= copy marker gene lookup over to output file =========
    log.info("Writing marker genes to output file")
    marker_gene_lookup = serialize_markers(
        marker_cache_path=query_marker_tmp,
        taxonomy_tree=taxonomy_tree)

    del file_tracker

    return {'assignments': result, 'marker_genes': marker_gene_lookup}


def _validate_config(
        config,
        file_tracker,
        log):

    if "query_path" not in config:
        log.error("'query_path' not in config")

    if "precomputed_stats" not in config:
        log.error("'precomputed_stats' not in config")

    if "reference_markers" not in config:
        log.error("'reference_markers' not in config")

    if "query_markers" not in config:
        log.error("'query_markers' not in config")

    if "type_assignment" not in config:
        log.error("'type_assignment' not in config")

    _check_config(
        config_dict=config["type_assignment"],
        config_name="type_assignment",
        key_name=['n_processors',
                  'chunk_size',
                  'bootstrap_factor',
                  'bootstrap_iteration',
                  'rng_seed',
                  'normalization'],
        log=log)

    file_tracker.add_file(
        config["query_path"],
        input_only=True)

    reference_marker_config = config["reference_markers"]

    precomputed_config = config["precomputed_stats"]

    _check_config(
        config_dict=precomputed_config,
        config_name='precomputed_stats',
        key_name=['path'],
        log=log)

    file_tracker.add_file(
        precomputed_config["path"],
        input_only=False)

    if not file_tracker.file_exists(precomputed_config["path"]):
        _check_config(
            config_dict=precomputed_config,
            config_name='precomputed_config',
            key_name=['reference_path', 'normalization'],
            log=log)

        has_columns = 'column_hierarchy' in precomputed_config
        has_taxonomy = 'taxonomy_tree' in precomputed_config

        if has_columns and has_taxonomy:
            log.error(
                "Cannot specify both column_hierarchy and "
                "taxonomy_tree in precomputed_config")

        if not has_columns and not has_taxonomy:
            log.error(
                "Must specify one of column_hierarchy or "
                "taxonomy_tree in precomputed_config")

    reference_marker_config = config["reference_markers"]
    file_tracker.add_file(
        reference_marker_config["path"],
        input_only=False)

    if not file_tracker.file_exists(reference_marker_config["path"]):
        _check_config(
            config_dict=reference_marker_config,
            config_name='reference_markers',
            key_name=['n_processors', 'max_gb'],
            log=log)

    query_marker_config = config["query_markers"]
    _check_config(
        config_dict=query_marker_config,
        config_name="query_markers",
        key_name=['n_per_utility', 'n_processors'],
        log=log)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--result_path', type=str, default=None)
    parser.add_argument('--log_path', type=str, default=None)
    parser.add_argument('--local_tmp', default=False, action='store_true')
    args = parser.parse_args()

    with open(args.config_path, 'rb') as in_file:
        config = json.load(in_file)

    if args.local_tmp:
        config['tmp_dir'] = os.environ['TMPDIR']

    if args.result_path is None:
        result_path = config['result_path']
    else:
        result_path = args.result_path

    run_mapping(
        config=config,
        output_path=result_path,
        log_path=args.log_path)


def assert_mappings_equal(
        mapping0,
        mapping1,
        compare_cell_id=True,
        eps=1.0e-6):
    """
    Assert that two cell type mappings are equivalent
    """
    for cell0, cell1 in zip(mapping0, mapping1):
        assert set(cell0.keys()) == set(cell1.keys())
        for k in cell0:
            if not compare_cell_id:
                if k == 'cell_id':
                    continue
            if not compare_field(cell0[k], cell1[k], eps=eps):
                msg = (
                    f"{json.dumps(cell0, indent=2)}\n"
                    f"{json.dumps(cell1, indent=2)}\n"
                    f"Mismatch on {k}"
                )
                raise RuntimeError(msg)


def compare_field(value0, value1, eps=1.0e-4):
    if isinstance(value0, list):
        value0 = np.array(value0)
        value1 = np.array(value1)
    if isinstance(value0, np.ndarray):
        if np.issubdtype(value0.dtype, np.number):
            return np.allclose(
                value0,
                value1,
                atol=eps,
                rtol=eps
            )
        else:
            return np.array_equal(
                value0,
                value1
            )
    elif isinstance(value0, dict):
        if set(value0.keys()) != set(value1.keys()):
            return False
        for k in value0:
            sub_k = compare_field(value0[k], value1[k])
            if not sub_k:
                return False
        return True
    elif isinstance(value0, numbers.Number):
        return np.allclose(value0, value1, atol=eps, rtol=eps)
    else:
        return value0 == value1


def _legacy_get_query_gene_names(
        query_gene_path,
        gene_id_col=None):
    """
    Legacy implementation that relied on the hard-coded GeneIdMapper
    class, rather than the mmc_gene_mapper infrastructure

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

    n_unmapped = 0
    was_changed = False
    return result, n_unmapped, was_changed


if __name__ == "__main__":
    main()
