import argschema
import copy
import h5py
import json
import multiprocessing
import numpy as np
import pathlib
import tempfile
import time
import traceback

import cell_type_mapper
import cell_type_mapper.utils.write_hierarchical_results as write_hier
import cell_type_mapper.utils.write_hann_results as write_hann

from cell_type_mapper.utils.cloud_utils import (
    sanitize_paths)

from cell_type_mapper.utils.torch_utils import (
    is_torch_available,
    is_cuda_available,
    use_torch)

from cell_type_mapper.utils.utils import (
    get_timestamp,
    mkstemp_clean,
    _clean_up,
    warn_on_parallelization)

from cell_type_mapper.utils.output_utils import (
    get_execution_metadata)

from cell_type_mapper.file_tracker.file_tracker import (
    FileTracker)

from cell_type_mapper.cli.cli_log import (
    CommandLog)

from cell_type_mapper.utils.cli_utils import (
    align_query_gene_names,
    config_from_args)

from cell_type_mapper.diff_exp.precompute_utils import (
    drop_nodes_from_precomputed_stats
)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.type_assignment.marker_cache_v2 import (
    serialize_markers,
    create_marker_cache_from_specified_markers)

from cell_type_mapper.type_assignment.election_runner import (
    run_type_assignment_on_h5ad,
)

import cell_type_mapper.type_assignment.hierarchical_mapping as hier
import cell_type_mapper.hann_mapping.hann_mapping as hann

from cell_type_mapper.schemas.from_specified_markers import (
    FromSpecifiedMarkersSchema)


if use_torch() and is_cuda_available():
    multiprocessing.set_start_method("spawn", force=True)


class FromSpecifiedMarkersRunner(argschema.ArgSchemaParser):
    default_schema = FromSpecifiedMarkersSchema

    log_obj = None

    def set_log_obj(self, log_obj):
        self.log_obj = log_obj

    def run(self):
        self.run_mapping(write_to_disk=True)

    def run_mapping(self, write_to_disk=True):

        mapping_exception = None
        t0 = time.time()
        metadata_config = config_from_args(
            input_config=self.args,
            cloud_safe=self.args['cloud_safe']
        )

        if self.args['type_assignment']['algorithm'] == 'hierarchical':
            stdout_name = 'Hierarchical'
        elif self.args['type_assignment']['algorithm'] == 'hann':
            stdout_name = 'HANN'

        msg = (f'=== Running {stdout_name} Mapping '
               f'{cell_type_mapper.__version__} ')
        if self.args['verbose_stdout']:
            msg += ('with config ===\n'
                    f'{json.dumps(metadata_config, indent=2)}')
        else:
            msg += '\n'
        print(msg)

        if self.log_obj is None:
            log = CommandLog(
                verbose_stdout=self.args['verbose_stdout']
            )
        else:
            log = self.log_obj

        if self.args['type_assignment']['n_processors'] > 1:
            warn_on_parallelization(log=log)

        if 'tmp_dir' not in self.args:
            raise RuntimeError("did not specify tmp_dir")

        if self.args['tmp_dir'] is not None:
            timestamp = get_timestamp().replace('-', '')
            tmp_dir = tempfile.mkdtemp(
                dir=self.args['tmp_dir'],
                prefix=f'cell_type_mapper_{timestamp}_')
        else:
            tmp_dir = None

        # create this now in case _run_mapping errors
        # before creating the output dict (the finally
        # block will add some logging info to output)
        metadata = dict()
        sub_result_list = None
        tmp_result_dir = None

        try:
            if self.args['tmp_dir'] is not None:
                tmp_result_dir = tempfile.mkdtemp(
                    dir=tmp_dir,
                    prefix='result_buffer_')
            else:
                tmp_result_dir = tempfile.mkdtemp(
                    dir=self.args['extended_result_dir'],
                    prefix='result_buffer_')

            (sub_result_list,
             metadata) = _run_mapping(
                config=self.args,
                tmp_dir=tmp_dir,
                tmp_result_dir=tmp_result_dir,
                log=log)

            log.info(
                "MAPPING FROM SPECIFIED MARKERS RAN SUCCESSFULLY",
                to_stdout=True)

        except Exception as err:
            mapping_exception = err
            traceback_msg = "an ERROR occurred ===="
            traceback_msg += f"\n{traceback.format_exc()}\n"
            log.add_msg(traceback_msg)
            raise

        finally:
            log.info(
                "WRITING OUTPUT AND CLEANING UP",
                to_stdout=True)

            metadata = add_metadata_to_output(
                output=metadata,
                metadata_config=metadata_config,
                log=log,
                t0=t0,
                cloud_safe=self.args['cloud_safe']
            )

            algorithm = self.args['type_assignment']['algorithm']
            if algorithm == 'hierarchical':
                hier_result = None
                if sub_result_list is not None:
                    hier_result = hier.collate_hierarchical_mappings(
                        sub_result_list
                    )

                return_packet = write_hier.write_hierarchical_output(
                    mapping_result=hier_result,
                    metadata=metadata,
                    config=self.args,
                    log=log,
                    mapping_exception=mapping_exception,
                    write_to_disk=write_to_disk
                )

            elif algorithm == 'hann':
                if sub_result_list is not None:
                    hann.collate_hann_mappings(
                        tmp_path_list=sub_result_list,
                        dst_path=self.args['hdf5_result_path']
                    )

                if write_to_disk:
                    write_hann.write_hann_metadata(
                        metadata=metadata,
                        log=log,
                        log_path=self.args['log_path'],
                        hdf5_output_path=self.args['hdf5_result_path'],
                        cloud_safe=self.args['cloud_safe']
                    )
                    return_packet = None
                else:
                    return_packet = {
                        'metadata': metadata,
                        'mapping_exception': mapping_exception
                    }

            if tmp_result_dir is not None:
                _clean_up(tmp_result_dir)

            _clean_up(tmp_dir)

            if return_packet is not None:
                return return_packet


def _run_mapping(
        config,
        tmp_dir,
        tmp_result_dir,
        log):

    if log is not None:
        log.env(f"is_torch_available: {is_torch_available()}")
        log.env(f"is_cuda_available: {is_cuda_available()}")
        log.env(f"use_torch: {use_torch()}")
        log.env("multiprocessing start method: "
                f"{multiprocessing.get_start_method()}")
        log.log_software_env()

    config = drop_nodes_from_taxonomy(
        tmp_dir=tmp_dir,
        config=config
    )

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
        input_only=True)

    query_loc = file_tracker.real_location(config['query_path'])
    type_assignment_config = config["type_assignment"]

    log.benchmark(msg="validating config and copying data",
                  duration=time.time()-t0)

    # ========= precomputed stats =========

    precomputed_loc = file_tracker.real_location(
        precomputed_config['path'])
    log.info(f"using ../{precomputed_loc.name} for precomputed_stats")

    log.info(f"reading taxonomy_tree from ../{precomputed_loc.name}")
    with h5py.File(precomputed_loc, "r") as in_file:
        taxonomy_tree = TaxonomyTree.from_str(
            serialized_dict=in_file["taxonomy_tree"][()].decode("utf-8"))
        reference_gene_names = json.loads(
            in_file["col_names"][()].decode("utf-8"))

    # Save the tree as it was originally read in, without flattening
    # or dropping of levels. This is what will be saved in the output
    # metadata.
    tree_for_metadata = TaxonomyTree(
        data=json.loads(taxonomy_tree.to_str(drop_cells=True)))

    if config['drop_level'] is not None:
        if config['drop_level'] in taxonomy_tree.hierarchy:
            taxonomy_tree = taxonomy_tree.drop_level(config['drop_level'])

    # ========= query marker cache =========

    if config['gene_mapping'] is not None:
        gene_mapping_db = config['gene_mapping']['db_path']
    else:
        gene_mapping_db = None

    (query_gene_names,
     n_genes_unmapped,
     _,
     gene_mapping_metadata) = align_query_gene_names(
        query_loc,
        gene_id_col=config['query_gene_id_col'],
        precomputed_stats_path=precomputed_loc,
        gene_mapper_db_path=gene_mapping_db,
        log=log)

    query_marker_tmp = pathlib.Path(
        mkstemp_clean(dir=tmp_dir,
                      prefix='query_marker_',
                      suffix='.h5'))

    t0 = time.time()

    marker_lookup = None
    if config['query_markers']['serialized_lookup'] is not None:
        marker_lookup_path = config['query_markers']['serialized_lookup']
        marker_lookup = json.load(open(marker_lookup_path, 'rb'))

        if 'metadata' in marker_lookup:
            marker_lookup.pop('metadata')
        if 'log' in marker_lookup:
            marker_lookup.pop('log')
    elif not config['query_markers']['collapse_markers']:
        msg = (
            "You did not specify a marker gene lookup table, "
            "but collapse_markers is False; unclear how to "
            "proceed."
        )
        log.error(msg)

    if config['query_markers']['collapse_markers']:

        if marker_lookup is not None:
            all_markers = set()
            for k in marker_lookup:
                all_markers = all_markers.union(
                    set(marker_lookup[k])
                )
            all_markers = sorted(all_markers)
        else:
            marker_lookup = dict()
            query_gene_set = set(query_gene_names)
            reference_gene_set = set(reference_gene_names)
            all_markers = query_gene_set.intersection(reference_gene_set)

            if len(all_markers) == 0:
                msg = (
                    "There was no overlap between the genes in "
                    "the query dataset and the genes in the "
                    "reference dataset.\n"
                    f"Example query genes: {query_gene_names[:5]}\n"
                    f"Example reference genes: {reference_gene_names[:5]}\n"
                )
                log.error(msg)
            diff = query_gene_set-all_markers
            if len(diff) > 0:
                msg = (
                    f"{len(diff)} of {len(query_gene_set)} genes in the "
                    "query dataset were not present in the reference dataset. "
                    "These genes could not be used as markers and "
                    "were ignored."
                )
                log.warn(msg)

            all_markers = sorted(all_markers)

            for parent in taxonomy_tree.all_parents:
                if parent is None:
                    key = str(parent)
                else:
                    key = f'{parent[0]}/{parent[1]}'
                marker_lookup[key] = []

        if len(all_markers) >= 10000:
            msg = (
                "Your query_marker configuration has resulted in "
                f"{len(all_markers)} marker genes being used at "
                "every decision point in the taxonomy. This will be "
                "very resource-intensive. You should consider specifying "
                "a more limited set of marker genes."
            )
            log.warn(msg)

        for k in marker_lookup:
            marker_lookup[k] = all_markers

    if config['flatten']:

        taxonomy_tree = taxonomy_tree.flatten()

        all_markers = set()
        for k in marker_lookup:
            if k not in ('log', 'metadata'):
                all_markers = all_markers.union(set(marker_lookup[k]))
        all_markers = sorted(all_markers)
        marker_lookup = {'None': all_markers}

    create_marker_cache_from_specified_markers(
        marker_lookup=marker_lookup,
        reference_gene_names=reference_gene_names,
        query_gene_names=query_gene_names,
        output_cache_path=query_marker_tmp,
        log=log,
        taxonomy_tree=taxonomy_tree,
        min_markers=config['type_assignment']['min_markers'])

    log.benchmark(msg="creating query marker cache",
                  duration=time.time()-t0)

    # ========= type assignment =========

    t0 = time.time()
    rng = np.random.default_rng(type_assignment_config['rng_seed'])

    if type_assignment_config['bootstrap_factor_lookup'] is not None:
        bootstrap_factor_lookup = dict()
        for pair in type_assignment_config['bootstrap_factor_lookup']:
            bootstrap_factor_lookup[pair[0]] = pair[1]
    else:
        bootstrap_factor_lookup = {
            level: type_assignment_config['bootstrap_factor']
            for level in taxonomy_tree.hierarchy[:-1]
        }
        bootstrap_factor_lookup['None'] = type_assignment_config[
                                                'bootstrap_factor']

    sub_result_list = run_type_assignment_on_h5ad(
        query_h5ad_path=query_loc,
        precomputed_stats_path=precomputed_loc,
        marker_gene_cache_path=query_marker_tmp,
        taxonomy_tree=taxonomy_tree,
        n_processors=type_assignment_config['n_processors'],
        chunk_size=type_assignment_config['chunk_size'],
        bootstrap_factor_lookup=bootstrap_factor_lookup,
        bootstrap_iteration=type_assignment_config['bootstrap_iteration'],
        rng=rng,
        n_assignments=type_assignment_config['n_runners_up']+1,
        normalization=type_assignment_config['normalization'],
        tmp_dir=tmp_dir,
        log=log,
        max_gb=config['max_gb'],
        output_taxonomy_tree=tree_for_metadata,
        results_output_path=tmp_result_dir,
        algorithm=type_assignment_config['algorithm'])

    log.benchmark(msg="assigning cell types",
                  duration=time.time()-t0)

    # ========= copy marker gene lookup over to output file =========
    log.info("Writing marker genes to output file")
    marker_gene_lookup = serialize_markers(
        marker_cache_path=query_marker_tmp,
        taxonomy_tree=taxonomy_tree)

    metadata = dict()
    metadata["marker_genes"] = marker_gene_lookup
    metadata["taxonomy_tree"] = json.loads(tree_for_metadata.to_str())
    metadata["n_unmapped_genes"] = n_genes_unmapped
    metadata["gene_identifier_mapping"] = gene_mapping_metadata

    return (sub_result_list, metadata)


def drop_nodes_from_taxonomy(tmp_dir, config):
    """
    Drop nodes from precomputed_stats files, if needed.
    Write the files with the amended taxonomies to new files in
    tmp_dir.
    Update config to point to the new files.
    Return config
    """
    if config['nodes_to_drop'] is None:
        return config

    precompute_dir = tempfile.mkdtemp(
        dir=tmp_dir,
        prefix='munged_precomputed_stats_files'
    )

    src_path = pathlib.Path(config['precomputed_stats']['path'])

    main_tmp_path = mkstemp_clean(
        dir=precompute_dir,
        prefix=src_path.name,
        suffix='.h5',
        delete=True)

    drop_nodes_from_precomputed_stats(
        src_path=src_path,
        dst_path=main_tmp_path,
        node_list=config['nodes_to_drop'],
        clobber=False
    )

    config['precomputed_stats']['path'] = main_tmp_path
    return config


def add_metadata_to_output(
        output,
        metadata_config,
        log,
        t0,
        cloud_safe):
    """
    Add metadata entries to output dict.

    Parameters
    ----------
    output:
        the dict we are adding metadata entries to
    metadata_config:
        the config dict to be recorded as metadata
    log:
        the CommandLog for this run
    t0:
        the time at which this run started
    cloud_safe:
        a boolean indicating whether or not to run
        this in 'cloud safe' mode (cloud save mode
        sanitizes paths so that the S3 bucket we are
        running in cannot be inferred from the
        metadata packet)

    Returns
    -------
    updated output dict (output is also altered in place)
    """
    output["config"] = metadata_config
    output_log = copy.deepcopy(log.log)
    if cloud_safe:
        output_log = sanitize_paths(output_log)
    output["log"] = output_log

    metadata = get_execution_metadata(
        module_file=__file__,
        t0=t0)

    output['metadata'] = metadata

    return output


def main():
    runner = FromSpecifiedMarkersRunner()
    runner.run()


if __name__ == "__main__":
    main()
