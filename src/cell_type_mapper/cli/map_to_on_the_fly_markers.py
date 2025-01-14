import argschema
import copy
import pathlib
import tempfile
import time

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.diff_exp.precompute_utils import (
    drop_nodes_from_precomputed_stats
)

from cell_type_mapper.utils.cli_utils import (
    config_from_args
)

from cell_type_mapper.utils.output_utils import (
    get_execution_metadata)

from cell_type_mapper.schemas.mixins import (
    NProcessorsMixin,
    NodesToDropMixin,
    VerboseStdoutMixin)

from cell_type_mapper.schemas.reference_marker_finder import (
    ReferenceFinderConfigMixin)

from cell_type_mapper.schemas.query_marker_finder import (
    QueryFinderConfigMixin)

from cell_type_mapper.schemas.search_mixins import (
    SearchSchemaMixin_noNProcessors)

from cell_type_mapper.cli.reference_markers import (
    ReferenceMarkerRunner)

from cell_type_mapper.cli.query_markers import (
    QueryMarkerRunner)

from cell_type_mapper.cli.from_specified_markers import (
    FromSpecifiedMarkersRunner,
    write_mapping_to_disk)

from cell_type_mapper.cli.cli_log import CommandLog


class QueryMarkerSchema_OTF(
        argschema.ArgSchema,
        QueryFinderConfigMixin):

    pass


class ReferenceMarkerSchema_OTF(
        argschema.ArgSchema,
        ReferenceFinderConfigMixin):

    precomputed_path_list = argschema.fields.List(
        argschema.fields.InputFile,
        required=True,
        default=None,
        allow_none=True,
        cli_as_single_argument=True,
        description=(
            "List of paths to precomputed stats files "
            "for which reference markers will be computed. "
            "If None, the precomputed_stats.path parameter "
            "will be used. You would specify many if there "
            "were multiple modalities whose markers needed "
            "to be calculated separately and aggregated, "
            "as in Whole Mouse Brain 10X."))


class MapperSchema_OTF(
        argschema.ArgSchema,
        SearchSchemaMixin_noNProcessors,
        NProcessorsMixin,
        NodesToDropMixin,
        VerboseStdoutMixin):

    query_markers = argschema.fields.Nested(
        QueryMarkerSchema_OTF,
        required=True)

    reference_markers = argschema.fields.Nested(
        ReferenceMarkerSchema_OTF,
        required=True)


class OnTheFlyMapper(argschema.ArgSchemaParser):

    default_schema = MapperSchema_OTF

    def run(self):
        t0 = time.time()
        log = CommandLog()
        tmp_dir = tempfile.mkdtemp(
            dir=self.args['tmp_dir'])
        try:
            metadata_config = config_from_args(
                input_config=self.args,
                cloud_safe=self.args['cloud_safe']
            )

            mapping_result = self._run(tmp_dir=tmp_dir, log=log)

            mapping_result['output']['config'] = metadata_config
            mapping_result['output']['metadata'] = get_execution_metadata(
                    module_file=__file__,
                    t0=t0)

            write_mapping_to_disk(
                output=mapping_result['output'],
                log=log,
                log_path=self.args['log_path'],
                output_path=mapping_result['output_path'],
                hdf5_output_path=mapping_result['hdf5_output_path'],
                cloud_safe=self.args['cloud_safe']
            )

            if mapping_result['mapping_exception'] is not None:
                raise mapping_result['mapping_exception']

        finally:
            _clean_up(tmp_dir)

    def _run(self, tmp_dir, log):

        self.drop_nodes_from_taxonomy(tmp_dir=tmp_dir)

        reference_marker_dir = tempfile.mkdtemp(dir=tmp_dir)

        if self.args['reference_markers']['precomputed_path_list'] is None:
            ref_stats_list = [self.args['precomputed_stats']['path']]
        else:
            ref_stats_list = self.args[
                'reference_markers']['precomputed_path_list']

        reference_marker_config = copy.deepcopy(
            self.args['reference_markers'])

        reference_marker_update = {
            'precomputed_path_list': ref_stats_list,
            'output_dir': reference_marker_dir,
            'query_path': self.args['query_path'],
            'n_processors': self.args['n_processors'],
            'drop_level': self.args['drop_level'],
            'cloud_safe': self.args['cloud_safe']
        }

        reference_marker_config.update(reference_marker_update)

        reference_marker_runner = ReferenceMarkerRunner(
            args=[],
            input_data=reference_marker_config)

        log.info("starting to find reference markers")
        reference_marker_runner.run()
        log.info("found reference markers")

        reference_marker_files = [
            str(n) for n in
            pathlib.Path(reference_marker_dir).iterdir()
            if n.is_file()]

        if len(reference_marker_files) == 0:
            log.error("No reference marker files created")

        query_marker_path = mkstemp_clean(
            dir=tmp_dir,
            prefix='query_markers_',
            suffix='.json')
        query_marker_config = copy.deepcopy(self.args['query_markers'])
        query_marker_update = {
            'n_processors': self.args['n_processors'],
            'drop_level': self.args['drop_level'],
            'tmp_dir': self.args['tmp_dir'],
            'output_path': query_marker_path,
            'query_path': self.args['query_path'],
            'reference_marker_path_list': reference_marker_files
        }

        query_marker_config.update(query_marker_update)
        query_marker_runner = QueryMarkerRunner(
            args=[], input_data=query_marker_config)
        query_marker_runner.run()
        log.info("found query markers")

        type_assignment = copy.deepcopy(self.args['type_assignment'])
        type_assignment['n_processors'] = self.args['n_processors']
        mapping_config = {
            'type_assignment': type_assignment,
            'tmp_dir': tmp_dir,
            'query_markers': {
                'serialized_lookup': query_marker_path,
            }}

        for k in ('query_path',
                  'extended_result_path',
                  'extended_result_dir',
                  'hdf5_result_path',
                  'csv_result_path',
                  'drop_level',
                  'flatten',
                  'max_gb',
                  'cloud_safe',
                  'precomputed_stats',
                  'summary_metadata_path',
                  'verbose_csv',
                  'verbose_stdout'):
            mapping_config[k] = self.args[k]

        mapping_runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=mapping_config)

        mapping_runner.set_log_obj(log)

        mapping_result = mapping_runner.run_mapping(write_to_disk=False)
        if mapping_result['mapping_exception'] is None:
            log.info("MAPPING FROM ON-THE-FLY MARKERS RAN SUCCESSFULLY")
        return mapping_result

    def drop_nodes_from_taxonomy(self, tmp_dir):
        """
        Drop nodes from precomputed_stats files, if needed.
        Write the files with the amended taxonomies to new files in
        tmp_dir.
        Update self.args to point to the new files.
        """
        if self.args['nodes_to_drop'] is None:
            return

        precompute_dir = tempfile.mkdtemp(
            dir=tmp_dir,
            prefix='munged_precomputed_stats_files'
        )

        src_path = pathlib.Path(self.args['precomputed_stats']['path'])

        main_tmp_path = mkstemp_clean(
            dir=precompute_dir,
            prefix=src_path.name,
            suffix='.h5',
            delete=True)

        drop_nodes_from_precomputed_stats(
            src_path=src_path,
            dst_path=main_tmp_path,
            node_list=self.args['nodes_to_drop'],
            clobber=False
        )

        self.args['precomputed_stats']['path'] = main_tmp_path

        if self.args['reference_markers']['precomputed_path_list'] is not None:
            new_list = []
            for src_path in self.args[
                    'reference_markers']['precomputed_path_list']:
                src_path = pathlib.Path(src_path)
                tmp_path = mkstemp_clean(
                    dir=precompute_dir,
                    prefix=src_path.name,
                    suffix='.h5',
                    delete=True
                )
                drop_nodes_from_precomputed_stats(
                    src_path=src_path,
                    dst_path=tmp_path,
                    node_list=self.args['nodes_to_drop'],
                    clobber=False
                )
                new_list.append(tmp_path)
            self.args['reference_markers']['precomputed_path_list'] = new_list


def main():
    runner = OnTheFlyMapper()
    runner.run()


if __name__ == "__main__":
    main()
