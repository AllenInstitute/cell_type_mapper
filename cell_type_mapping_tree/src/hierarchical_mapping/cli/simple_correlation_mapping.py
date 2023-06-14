import argschema
import json
import pathlib
import tempfile
import time
import traceback

from hierarchical_mapping.utils.utils import (
    _clean_up)

from hierarchical_mapping.utils.output_utils import (
    blob_to_csv)

from hierarchical_mapping.file_tracker.file_tracker import (
    FileTracker)

from hierarchical_mapping.cli.cli_log import (
    CommandLog)

from hierarchical_mapping.corr.correlate_cells import (
    corrmap_cells)

from hierarchical_mapping.utils.cli_utils import (
    create_precomputed_stats_file)

from hierarchical_mapping.cli.schemas import (
    SpecifiedMarkerSchema,
    CorrTypeAssignmentSchema,
    PrecomputedStatsSchema)


class CorrSchemaSpecifiedMarkers(argschema.ArgSchema):

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

    precomputed_stats = argschema.fields.Nested(
        PrecomputedStatsSchema,
        required=True)

    query_markers = argschema.fields.Nested(
        SpecifiedMarkerSchema,
        required=True)

    type_assignment = argschema.fields.Nested(
        CorrTypeAssignmentSchema,
        required=True)


class CorrMapSpecifiedMarkersRunner(argschema.ArgSchemaParser):
    default_schema = CorrSchemaSpecifiedMarkers

    def run(self):
        run_mapping(
            config=self.args,
            output_path=self.args['extended_result_path'],
            log_path=None)


def run_mapping(config, output_path, log_path=None):

    log = CommandLog()

    if 'tmp_dir' not in config:
        raise RuntimeError("did not specify tmp_dir")

    if config['tmp_dir'] is not None:
        tmp_dir = tempfile.mkdtemp(
            dir=config['tmp_dir'],
            prefix='flat_mapping_')
    else:
        tmp_dir = None

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

    precomputed_path = precomputed_config['path']
    precomputed_loc = file_tracker.real_location(precomputed_path)
    if file_tracker.file_exists(precomputed_path):
        log.info(f"using {precomputed_loc} for precomputed_stats")
    else:
        create_precomputed_stats_file(
            precomputed_config=precomputed_config,
            file_tracker=file_tracker,
            log=log,
            tmp_dir=tmp_dir)

    # ========= query marker cache =========

    # The marker genes will be stored as a dict mapping parent
    # node in the taxonomy tree to makers that should be used
    # when deciding between the children of that node. For flat
    # mapping, we will just concatenate *all* the marker genes in
    # that dict into a list of marker gene names.

    t0 = time.time()
    marker_lookup_path = config['query_markers']['serialized_lookup']
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
    result = corrmap_cells(
        query_path=query_loc,
        precomputed_path=precomputed_loc,
        marker_gene_list=marker_gene_names,
        rows_at_a_time=type_assignment_config['chunk_size'],
        n_processors=type_assignment_config['n_processors'],
        tmp_dir=tmp_dir,
        query_normalization=type_assignment_config['normalization'],
        log=log,
        max_gb=config['max_gb'])

    log.benchmark(msg="assigning cell types",
                  duration=time.time()-t0)

    class DummyTree(object):
        @property
        def hierarchy(self):
            return ['cluster']

    if config['csv_result_path'] is not None:
        blob_to_csv(
            results_blob=result,
            taxonomy_tree=DummyTree(),
            output_path=config['csv_result_path'],
            metadata_path=config['extended_result_path'])

    # right now, this just returns all of the marker genes specified
    # in the input JSON, without regard to which ones were actually
    # present in the query and reference datasets
    return {'assignments': result, 'marker_genes': marker_gene_names}


def main():
    runner = CorrMapSpecifiedMarkersRunner()
    runner.run()


if __name__ == "__main__":
    main()
