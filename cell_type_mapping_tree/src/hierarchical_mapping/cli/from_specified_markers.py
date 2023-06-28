import argschema
import h5py
import json
import multiprocessing
import numpy as np
import pathlib
import tempfile
import time
import traceback

from hierarchical_mapping.utils.torch_utils import (
    is_torch_available,
    is_cuda_available,
    use_torch)

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.utils.output_utils import (
    blob_to_csv)

from hierarchical_mapping.file_tracker.file_tracker import (
    FileTracker)

from hierarchical_mapping.cli.cli_log import (
    CommandLog)

from hierarchical_mapping.utils.cli_utils import (
    _get_query_gene_names)

from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from hierarchical_mapping.type_assignment.marker_cache_v2 import (
    serialize_markers,
    create_marker_cache_from_specified_markers)

from hierarchical_mapping.type_assignment.election_runner import (
    run_type_assignment_on_h5ad)

from hierarchical_mapping.utils.cli_utils import (
    create_precomputed_stats_file)


from hierarchical_mapping.cli.schemas import (
    SpecifiedMarkerSchema,
    HierarchicalTypeAssignmentSchema,
    PrecomputedStatsSchema)


if use_torch() and is_cuda_available():
    multiprocessing.set_start_method("spawn", force=True)


class HierarchicalSchemaSpecifiedMarkers(argschema.ArgSchema):

    tmp_dir = argschema.fields.OutputDir(
        required=False,
        default=None,
        allow_none=True,
        description="Optional temporary directory into which data "
        "will be copied for faster access (e.g. if the data "
        "naturally lives on a slow NFS drive)")

    query_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the h5ad file containing the query "
        "dataset")

    extended_result_dir = argschema.fields.OutputDir(
        required=False,
        default=None,
        allow_none=True,
        description="Optional temporary directory into which assignment "
        "results will be saved from each process.")

    extended_result_path = argschema.fields.OutputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to JSON file where extended results "
        "will be saved.")

    csv_result_path = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description="Path to CSV file where output file will be "
        "written (if None, no CSV will be produced).")

    max_gb = argschema.fields.Float(
        required=False,
        default=100.0,
        allow_none=False,
        description="In the event that a CSC matrix needs to be "
        "converted to a temporary on disk CSR matrix, how "
        "much memory (in gigabytes) can we use.")

    drop_level = argschema.fields.String(
        required=False,
        default="CCN20230504_SUPT",
        allow_none=True,
        description="If this level exists in the taxonomy, drop "
        "it before doing type assignment (this is to accommmodate "
        "the fact that the official taxonomy includes the "
        "'supertype', even though that level is not used "
        "during hierarchical type assignment")

    precomputed_stats = argschema.fields.Nested(
        PrecomputedStatsSchema,
        required=True)

    query_markers = argschema.fields.Nested(
        SpecifiedMarkerSchema,
        required=True)

    type_assignment = argschema.fields.Nested(
        HierarchicalTypeAssignmentSchema,
        required=True)

    flatten = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description="If true, flatten the taxonomy so that we are "
        "mapping directly to the leaf node")


class FromSpecifiedMarkersRunner(argschema.ArgSchemaParser):
    default_schema = HierarchicalSchemaSpecifiedMarkers

    def run(self):
        run_mapping(
            config=self.args,
            output_path=self.args['extended_result_path'],
            log_path=None)


def get_assignments(config, type_assignment):
    """Get the assignments from the type assignment output.
    If extended_results_dir is given, then the results were saved in
    individual {r0}_{r1}_assignment.json files, so parse these. Otherwise,
    the assignments are given in the 'assignments' key in type_assignment.
    """
    tmp_output_dir = config.get("extended_result_dir")
    if tmp_output_dir:
        assignments = []
        import glob
        temp_output_files = glob.glob(f"{tmp_output_dir}/*_assignment.json")
        for temp_output_file in temp_output_files:
            with open(temp_output_file, 'r') as f:
                chunk_assignments = json.load(f)
            assignments.extend(chunk_assignments)
            pathlib.Path(temp_output_file).unlink()
    else:  # temp output path not given
        assignments = type_assignment["assignments"]
    return assignments


def run_mapping(config, output_path, log_path=None):

    log = CommandLog()

    if 'tmp_dir' not in config:
        raise RuntimeError("did not specify tmp_dir")

    if config['tmp_dir'] is not None:
        tmp_dir = tempfile.mkdtemp(
            dir=config['tmp_dir'],
            prefix='hierarchical_mapping_')
    else:
        tmp_dir = None

    output = dict()
    csv_result = dict()

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
        assignments = get_assignments(config, type_assignment)
        output["results"] = assignments
        output["marker_genes"] = type_assignment["marker_genes"]
        csv_result["taxonomy_tree"] = type_assignment["taxonomy_tree"]
        csv_result["assignments"] = assignments

        if config['csv_result_path'] is not None:

            if config['type_assignment']['bootstrap_iteration'] == 1:
                confidence_key = 'avg_correlation'
                confidence_label = 'correlation_coefficient'
            else:
                confidence_key = 'bootstrapping_probability'
                confidence_label = 'bootstrapping_probability'

            blob_to_csv(
                results_blob=csv_result.get("assignments"),
                taxonomy_tree=csv_result.get("taxonomy_tree"),
                output_path=config['csv_result_path'],
                metadata_path=config['extended_result_path'],
                confidence_key=confidence_key,
                confidence_label=confidence_label)

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

    if log is not None:
        log.env(f"is_torch_available: {is_torch_available()}")
        log.env(f"is_cuda_available: {is_cuda_available()}")
        log.env(f"use_torch: {use_torch()}")
        log.env("multiprocessing start method: "
                f"{multiprocessing.get_start_method()}")
        log.log_software_env()

    t0 = time.time()
    file_tracker = FileTracker(
        tmp_dir=tmp_dir,
        log=log)

    file_tracker.add_file(
        config['query_path'],
        input_only=True)

    precomputed_config = config["precomputed_stats"]

    file_tracker.add_file(
        precomputed_config['path'],
        input_only=False)

    query_loc = file_tracker.real_location(config['query_path'])
    precomputed_config = config["precomputed_stats"]
    type_assignment_config = config["type_assignment"]

    log.benchmark(msg="validating config and copying data",
                  duration=time.time()-t0)

    # ========= precomputed stats =========

    precomputed_loc = file_tracker.real_location(
        precomputed_config['path'])
    precomputed_path = precomputed_config['path']
    if file_tracker.file_exists(precomputed_path):
        log.info(f"using {precomputed_loc} for precomputed_stats")
    else:
        create_precomputed_stats_file(
            precomputed_config=precomputed_config,
            file_tracker=file_tracker,
            log=log,
            tmp_dir=tmp_dir)

    log.info(f"reading taxonomy_tree from {precomputed_loc}")
    with h5py.File(precomputed_loc, "r") as in_file:
        taxonomy_tree = TaxonomyTree.from_str(
            serialized_dict=in_file["taxonomy_tree"][()].decode("utf-8"))
        reference_gene_names = json.loads(
            in_file["col_names"][()].decode("utf-8"))

    if config['drop_level'] is not None:
        if config['drop_level'] in taxonomy_tree.hierarchy:
            taxonomy_tree = taxonomy_tree.drop_level(config['drop_level'])

    # ========= query marker cache =========

    query_marker_tmp = pathlib.Path(
        mkstemp_clean(dir=tmp_dir,
                      prefix='query_marker_',
                      suffix='.h5'))

    t0 = time.time()

    marker_lookup_path = config['query_markers']['serialized_lookup']
    marker_lookup = json.load(open(marker_lookup_path, 'rb'))
    if config['flatten']:

        taxonomy_tree = taxonomy_tree.flatten()

        all_markers = set()
        for k in marker_lookup:
            if k == 'metadata':
                continue
            all_markers = all_markers.union(set(marker_lookup[k]))
        all_markers = list(all_markers)
        all_markers.sort()
        marker_lookup = {'None': all_markers}

    query_gene_names = _get_query_gene_names(query_loc)

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
        query_h5ad_path=query_loc,
        precomputed_stats_path=precomputed_loc,
        marker_gene_cache_path=query_marker_tmp,
        taxonomy_tree=taxonomy_tree,
        n_processors=type_assignment_config['n_processors'],
        chunk_size=type_assignment_config['chunk_size'],
        bootstrap_factor=type_assignment_config['bootstrap_factor'],
        bootstrap_iteration=type_assignment_config['bootstrap_iteration'],
        rng=rng,
        normalization=type_assignment_config['normalization'],
        tmp_dir=tmp_dir,
        log=log,
        max_gb=config['max_gb'],
        results_output_path=config.get("extended_result_dir"))

    log.benchmark(msg="assigning cell types",
                  duration=time.time()-t0)

    # ========= copy marker gene lookup over to output file =========
    log.info("Writing marker genes to output file")
    marker_gene_lookup = serialize_markers(
        marker_cache_path=query_marker_tmp,
        taxonomy_tree=taxonomy_tree)

    return {'assignments': result,
            'marker_genes': marker_gene_lookup,
            "taxonomy_tree": taxonomy_tree}


def main():
    runner = FromSpecifiedMarkersRunner()
    runner.run()


if __name__ == "__main__":
    main()
