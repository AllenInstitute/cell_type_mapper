import argparse
import json
import os
import pathlib
import tempfile
import time
import traceback

from hierarchical_mapping.utils.utils import (
    _clean_up)

from hierarchical_mapping.cli.cli_log import (
    CommandLog)

from hierarchical_mapping.cli.utils import (
    _copy_over_file,
    _make_temp_path,
    _check_config)

from hierarchical_mapping.corr.correlate_cells import (
    flatmap_cells)

from hierarchical_mapping.cli.processing_utils import (
    create_precomputed_stats_file)


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
        create_precomputed_stats_file(
            precomputed_config=precomputed_config,
            precomputed_path=precomputed_path,
            precomputed_tmp=precomputed_tmp,
            log=log,
            tmp_dir=tmp_dir)

    # ========= query marker cache =========

    # The marker genes will be stored as a dict mapping parent
    # node in the taxonomy tree to makers that should be used
    # when deciding between the children of that node. For flat
    # mapping, we will just concatenate *all* the marker genes in
    # that dict into a list of marker gene names.

    t0 = time.time()

    marker_gene_names = set()
    marker_tree = json.load(open(marker_lookup_path, 'rb'))
    for node in marker_tree:
        marker_gene_names = marker_gene_names.union(
            set(marker_tree[node]))
    marker_gene_names = list(marker_gene_names)
    marker_gene_names.sort()

    log.info(
        f"Read in {len(marker_gene_names)} marker genes")

    # ========= type assignment =========

    t0 = time.time()
    result = flatmap_cells(
        query_path=query_tmp,
        precomputed_path=precomputed_tmp,
        marker_gene_list=marker_gene_names,
        rows_at_a_time=type_assignment_config['chunk_size'],
        n_processors=type_assignment_config['n_processors'],
        tmp_dir=tmp_dir,
        query_normalization=type_assignment_config['normalization'])

    log.benchmark(msg="assigning cell types",
                  duration=time.time()-t0)

    # right now, this just returns all of the marker genes specified
    # in the input JSON, without regard to which ones were actually
    # present in the query and reference datasets
    return {'assignments': result, 'marker_genes': marker_gene_names}


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
