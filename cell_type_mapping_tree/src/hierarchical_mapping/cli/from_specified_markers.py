import argparse
import h5py
import json
import numpy as np
import os
import pathlib
import shutil
import tempfile
import time
import traceback

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.cli.cli_log import (
    CommandLog)

from hierarchical_mapping.cli.utils import (
    _copy_over_file,
    _make_temp_path,
    _check_config,
    _get_query_gene_names)

from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from hierarchical_mapping.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from hierarchical_mapping.type_assignment.marker_cache_v2 import (
    serialize_markers,
    create_marker_cache_from_specified_markers)

from hierarchical_mapping.type_assignment.election import (
    run_type_assignment_on_h5ad)


def run_mapping(config, output_path, log_path=None):

    log = CommandLog()

    if 'tmp_dir' not in config:
        raise RuntimeError("did not specify tmp_dir")

    tmp_dir = tempfile.mkdtemp(dir=config['tmp_dir'])

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
    validation_result = _validate_config(
            config=config,
            tmp_dir=tmp_dir,
            log=log)

    query_tmp = validation_result['query_tmp']
    precomputed_path = validation_result['precomputed_path']
    precomputed_tmp = validation_result['precomputed_tmp']
    precomputed_is_valid = validation_result['precomputed_is_valid']
    marker_lookup_path = validation_result['marker_lookup']

    precomputed_config = config["precomputed_stats"]
    type_assignment_config = config["type_assignment"]

    log.benchmark(msg="validating config and copying data",
                  duration=time.time()-t0)

    # ========= precomputed stats =========

    if precomputed_is_valid:
        log.info(f"using {precomputed_tmp} for precomputed_stats")
    else:
        log.info("creating precomputed stats file")

        reference_path = pathlib.Path(
            precomputed_config['reference_path'])

        ref_tmp = pathlib.Path(
            tempfile.mkdtemp(
                prefix='reference_data_',
                dir=tmp_dir))

        (reference_path,
         _) = _copy_over_file(file_path=reference_path,
                              tmp_dir=ref_tmp,
                              log=log)

        if 'column_hierarchy' in precomputed_config:
            column_hierarchy = precomputed_config['column_hierarchy']
            taxonomy_tree = None
        else:
            taxonomy_tree = TaxonomyTree.from_json_file(
                json_path=precomputed_config['taxonomy_tree'])
            column_hierarchy = None

        t0 = time.time()
        precompute_summary_stats_from_h5ad(
            data_path=reference_path,
            column_hierarchy=column_hierarchy,
            taxonomy_tree=taxonomy_tree,
            output_path=precomputed_tmp,
            rows_at_a_time=10000,
            normalization=precomputed_config['normalization'])
        log.benchmark(msg="precomputing stats",
                      duration=time.time()-t0)

        if precomputed_path is not None:
            log.info("copying precomputed stats from "
                     f"{precomputed_tmp} to {precomputed_path}")
            shutil.copy(
                src=precomputed_tmp,
                dst=precomputed_path)

        _clean_up(ref_tmp)

    log.info(f"reading taxonomy_tree from {precomputed_tmp}")
    with h5py.File(precomputed_tmp, "r") as in_file:
        taxonomy_tree = TaxonomyTree.from_str(
            serialized_dict=in_file["taxonomy_tree"][()].decode("utf-8"))
        reference_gene_names = json.loads(
            in_file["col_names"][()].decode("utf-8"))

    # ========= query marker cache =========

    query_marker_tmp = pathlib.Path(
        mkstemp_clean(dir=tmp_dir,
                      prefix='query_marker_',
                      suffix='.h5'))

    t0 = time.time()

    marker_lookup = json.load(open(marker_lookup_path, 'rb'))

    query_gene_names = _get_query_gene_names(query_tmp)

    create_marker_cache_from_specified_markers(
        marker_lookup=marker_lookup,
        reference_gene_names=reference_gene_names,
        query_gene_names=query_gene_names,
        output_cache_path=query_marker_tmp,
        log=log)

    log.benchmark(msg="creating query marker cache",
                  duration=time.time()-t0)

    # ========= type assignment =========

    t0 = time.time()
    rng = np.random.default_rng(type_assignment_config['rng_seed'])
    result = run_type_assignment_on_h5ad(
        query_h5ad_path=query_tmp,
        precomputed_stats_path=precomputed_tmp,
        marker_gene_cache_path=query_marker_tmp,
        taxonomy_tree=taxonomy_tree,
        n_processors=type_assignment_config['n_processors'],
        chunk_size=type_assignment_config['chunk_size'],
        bootstrap_factor=type_assignment_config['bootstrap_factor'],
        bootstrap_iteration=type_assignment_config['bootstrap_iteration'],
        rng=rng,
        normalization=type_assignment_config['normalization'],
        tmp_dir=tmp_dir)
    log.benchmark(msg="assigning cell types",
                  duration=time.time()-t0)

    # ========= copy marker gene lookup over to output file =========
    log.info("Writing marker genes to output file")
    marker_gene_lookup = serialize_markers(
        marker_cache_path=query_marker_tmp,
        taxonomy_tree=taxonomy_tree)
    return {'assignments': result, 'marker_genes': marker_gene_lookup}


def _validate_config(
        config,
        tmp_dir,
        log):

    result = dict()

    if "query_path" not in config:
        log.error("'query_path' not in config")

    if "precomputed_stats" not in config:
        log.error("'precomputed_stats' not in config")

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

    (query_tmp,
     query_is_valid) = _copy_over_file(
         file_path=config["query_path"],
         tmp_dir=tmp_dir,
         log=log)

    if not query_is_valid:
        log.error(
            f"{config['query_path']} is not a valid file")

    result['query_tmp'] = query_tmp

    precomputed_config = config["precomputed_stats"]
    lookup = _make_temp_path(
        config_dict=precomputed_config,
        tmp_dir=tmp_dir,
        log=log,
        prefix="precomputed_stats_",
        suffix=".h5")

    precomputed_path = lookup["path"]
    precomputed_tmp = lookup["tmp"]
    precomputed_is_valid = lookup["is_valid"]

    result['precomputed_path'] = precomputed_path
    result['precomputed_tmp'] = precomputed_tmp
    result['precomputed_is_valid'] = precomputed_is_valid

    if not precomputed_is_valid:
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

    query_marker_config = config["query_markers"]
    _check_config(
        config_dict=query_marker_config,
        config_name="query_markers",
        key_name=['serialized_lookup'],
        log=log)

    serialized_lookup_path = pathlib.Path(
        query_marker_config['serialized_lookup'])

    if not serialized_lookup_path.is_file():
        log.error(
            "serialized marker lookup\n"
            f"{serialized_lookup_path.resolve().absolute()}\n"
            "is not a file")

    result["marker_lookup"] = serialized_lookup_path

    return result


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


if __name__ == "__main__":
    main()
