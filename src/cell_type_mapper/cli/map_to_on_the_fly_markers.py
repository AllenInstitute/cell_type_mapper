import argschema
import copy
import json
import pathlib
import tempfile
import time

from cell_type_mapper.utils.output_utils import (
    get_execution_metadata)

from cell_type_mapper.utils.cloud_utils import (
    sanitize_paths)

from cell_type_mapper.schemas.mixins import (
    NProcessorsMixin)

from cell_type_mapper.schemas.reference_marker_finder import (
    ReferenceFinderConfigMixin)

from cell_type_mapper.schemas.query_marker_finder import (
    QueryFinderConfigMixin)

from cell_type_mapper.schemas.search_mixins import (
    SearchSchemaMixin)

from cell_type_mapper.cli.reference_markers import (
    ReferenceMarkerRunner)

from cell_type_mapper.cli.query_markers import (
    QueryMarkerRunner)

from cell_type_mapper.cli.from_specified_markers import (
    FromSpecifiedMarkersRunner)

from cell_type_mapper.cli.cli_log import CommandLog

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)


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
        SearchSchemaMixin,
        NProcessorsMixin):

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
            self._run(tmp_dir=tmp_dir, log=log)

            # modify metadata to reflect that the code was
            # actually run with this module, rather than
            # the from_specified_markers module
            output_path = self.args['extended_result_path']
            if output_path is not None:
                metadata_config = copy.deepcopy(self.args)
                if self.args['cloud_safe']:
                    metadata_config = sanitize_paths(metadata_config)
                    metadata_config.pop('extended_result_dir')
                    metadata_config.pop('tmp_dir')

                with open(output_path, 'rb') as src:
                    results = json.load(src)

                results.pop('config')
                results['config'] = metadata_config
                results.pop('metadata')
                results['metadata'] = get_execution_metadata(
                    module_file=__file__,
                    t0=t0)

                with open(output_path, 'w') as dst:
                    dst.write(json.dumps(results, indent=2))

        finally:
            _clean_up(tmp_dir)

    def _run(self, tmp_dir, log):
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
                  'csv_result_path',
                  'drop_level',
                  'flatten',
                  'max_gb',
                  'cloud_safe',
                  'precomputed_stats',
                  'summary_metadata_path'):
            mapping_config[k] = self.args[k]

        mapping_runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=mapping_config)

        mapping_runner.run()
        log.info("RAN SUCCESSFULLY")


def main():
    runner = OnTheFlyMapper()
    runner.run()


if __name__ == "__main__":
    main()
