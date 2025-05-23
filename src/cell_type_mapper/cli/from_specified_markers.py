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

from cell_type_mapper.utils.cloud_utils import (
    sanitize_paths)

from cell_type_mapper.utils.torch_utils import (
    is_torch_available,
    is_cuda_available,
    use_torch)

from cell_type_mapper.utils.utils import (
    get_timestamp,
    mkstemp_clean,
    clean_for_json,
    _clean_up,
    warn_on_parallelization)

from cell_type_mapper.utils.anndata_utils import (
    read_uns_from_h5ad,
    read_df_from_h5ad,
    append_to_obsm)

from cell_type_mapper.utils.output_utils import (
    blob_to_csv,
    blob_to_df,
    get_execution_metadata,
    blob_to_hdf5)

from cell_type_mapper.file_tracker.file_tracker import (
    FileTracker)

from cell_type_mapper.cli.cli_log import (
    CommandLog)

from cell_type_mapper.utils.cli_utils import (
    _get_query_gene_names,
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
    run_type_assignment_on_h5ad)

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

        msg = ('=== Running Hierarchical Mapping '
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

        # create this now in case _run_mapping errors
        # before creating the output dict (the finally
        # block will add some logging info to output)
        output = dict()

        if 'tmp_dir' not in self.args:
            raise RuntimeError("did not specify tmp_dir")

        if self.args['tmp_dir'] is not None:
            timestamp = get_timestamp().replace('-', '')
            tmp_dir = tempfile.mkdtemp(
                dir=self.args['tmp_dir'],
                prefix=f'cell_type_mapper_{timestamp}_')
        else:
            tmp_dir = None

        if self.args['extended_result_path'] is not None:
            output_path = pathlib.Path(self.args['extended_result_path'])
        else:
            output_path = None

        if self.args['hdf5_result_path'] is not None:
            hdf5_output_path = pathlib.Path(self.args['hdf5_result_path'])
        else:
            hdf5_output_path = None

        if self.args['log_path'] is not None:
            log_path = pathlib.Path(self.args['log_path'])
        else:
            log_path = None

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
            if self.args['tmp_dir'] is not None:
                tmp_result_dir = tempfile.mkdtemp(
                    dir=tmp_dir,
                    prefix='result_buffer_')
            else:
                tmp_result_dir = tempfile.mkdtemp(
                    dir=self.args['extended_result_dir'],
                    prefix='result_buffer_')

            output = _run_mapping(
                config=self.args,
                tmp_dir=tmp_dir,
                tmp_result_dir=tmp_result_dir,
                log=log)

            if self.args['summary_metadata_path'] is not None:
                n_mapped_cells = len(output['results'])
                uns = read_uns_from_h5ad(
                        self.args['query_path'])
                n_total_genes = len(
                        read_df_from_h5ad(
                            self.args['query_path'],
                            df_name='var'))

                local_unmapped = output.pop('n_unmapped_genes')

                if self.args['map_to_ensembl']:
                    n_mapped_genes = n_total_genes - local_unmapped
                else:
                    gene_key = 'AIBS_CDM_n_mapped_genes'
                    if gene_key in uns:
                        n_mapped_genes = uns[gene_key]
                    else:
                        n_mapped_genes = len(
                            read_df_from_h5ad(
                                self.args['query_path'],
                                df_name='var'))

                with open(self.args['summary_metadata_path'], 'w') as dst:
                    dst.write(
                        json.dumps(
                            {
                             'n_mapped_cells': int(n_mapped_cells),
                             'n_mapped_genes': int(n_mapped_genes)
                            },
                            indent=2))

            _clean_up(tmp_result_dir)

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
            _clean_up(tmp_dir)
            log.info(
                "CLEANING UP",
                to_stdout=True)

            output["config"] = metadata_config
            output_log = copy.deepcopy(log.log)
            if self.args['cloud_safe']:
                output_log = sanitize_paths(output_log)
            output["log"] = output_log

            metadata = get_execution_metadata(
                module_file=__file__,
                t0=t0)
            output['metadata'] = metadata

            uns = read_uns_from_h5ad(self.args["query_path"])
            if "AIBS_CDM_gene_mapping" in uns:
                output["gene_identifier_mapping"] = \
                    uns["AIBS_CDM_gene_mapping"]

            if write_to_disk:
                write_mapping_to_disk(
                    output=output,
                    log=log,
                    log_path=log_path,
                    output_path=output_path,
                    hdf5_output_path=hdf5_output_path,
                    cloud_safe=self.args['cloud_safe']
                )
            else:
                return {
                    'output': output,
                    'log': log,
                    'log_path': log_path,
                    'output_path': output_path,
                    'hdf5_output_path': hdf5_output_path,
                    'mapping_exception': mapping_exception
                }


def write_mapping_to_disk(
        output,
        log,
        log_path,
        output_path,
        hdf5_output_path,
        cloud_safe):

    if log_path is not None:
        log.write_log(log_path, cloud_safe=cloud_safe)

    if output_path is not None:
        with open(output_path, "w") as out_file:
            out_file.write(
                json.dumps(
                    clean_for_json(output), indent=2
                )
            )

    if hdf5_output_path is not None:
        blob_to_hdf5(
            output_blob=output,
            dst_path=hdf5_output_path)


def _run_mapping(config, tmp_dir, tmp_result_dir, log):

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

    (query_gene_names,
     n_unmapped,
     _) = _get_query_gene_names(
        query_loc,
        map_to_ensembl=config['map_to_ensembl'],
        gene_id_col=config['query_gene_id_col'])

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
        n_assignments=type_assignment_config['n_runners_up']+1,
        normalization=type_assignment_config['normalization'],
        tmp_dir=tmp_dir,
        log=log,
        max_gb=config['max_gb'],
        output_taxonomy_tree=tree_for_metadata,
        results_output_path=tmp_result_dir)

    log.benchmark(msg="assigning cell types",
                  duration=time.time()-t0)

    # ========= copy marker gene lookup over to output file =========
    log.info("Writing marker genes to output file")
    marker_gene_lookup = serialize_markers(
        marker_cache_path=query_marker_tmp,
        taxonomy_tree=taxonomy_tree)

    if config['csv_result_path'] is not None:

        if config['type_assignment']['bootstrap_iteration'] == 1 \
                or config['verbose_csv']:

            confidence_key = 'avg_correlation'
            confidence_label = 'correlation_coefficient'

        else:

            confidence_key = 'bootstrapping_probability'
            confidence_label = 'bootstrapping_probability'

        if config['verbose_csv']:
            valid_suffixes = [
                '_aggregate_probability',
                '_bootstrapping_probability',
                '_avg_correlation'
            ]
        else:
            valid_suffixes = None

        check_consistency = False
        if config['type_assignment']['bootstrap_iteration'] > 1:
            if config['flatten']:
                check_consistency = True

        blob_to_csv(
            results_blob=result,
            taxonomy_tree=tree_for_metadata,
            output_path=config['csv_result_path'],
            metadata_path=config['extended_result_path'],
            confidence_key=confidence_key,
            confidence_label=confidence_label,
            config=config,
            valid_suffixes=valid_suffixes,
            check_consistency=check_consistency,
            rows_at_a_time=100000)

    if config['obsm_key']:

        df = blob_to_df(
            results_blob=result,
            taxonomy_tree=tree_for_metadata).set_index('cell_id')

        # need to make sure that the rows are written in
        # the same order that they occur in the obs
        # dataframe

        obs = read_df_from_h5ad(
            h5ad_path=config['query_path'],
            df_name='obs')

        df = df.loc[obs.index.values]

        append_to_obsm(
            h5ad_path=config['query_path'],
            obsm_key=config['obsm_key'],
            obsm_value=df,
            clobber=config['obsm_clobber'])

    output = dict()
    output["results"] = result
    output["marker_genes"] = marker_gene_lookup
    output["taxonomy_tree"] = json.loads(tree_for_metadata.to_str())
    output["n_unmapped_genes"] = n_unmapped

    return output


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


def main():
    runner = FromSpecifiedMarkersRunner()
    runner.run()


if __name__ == "__main__":
    main()
