import argschema
import copy
import os
import pathlib
import tempfile

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
        SearchSchemaMixin):

    query_markers = argschema.fields.Nested(
        QueryMarkerSchema_OTF,
        required=True)

    reference_markers = argschema.fields.Nested(
        ReferenceMarkerSchema_OTF,
        required=True)

    n_processors = argschema.fields.Int(
        required=False,
        default=32,
        allow_none=False,
        description="Number of independendent processes to use when "
        "parallelizing work for mapping job")

    use_gpu = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description=(
            "If False, do not use GPU implementation, even "
            "if a GPU and CUDA are present on the system."
        ))


class OnTheFlyMapper(argschema.ArgSchemaParser):

    default_schema = MapperSchema_OTF

    def run(self):
        log = CommandLog()
        tmp_dir = tempfile.mkdtemp(
            dir=self.args['tmp_dir'])

        cached_env = None
        env_var = 'AIBS_BKP_USE_TORCH'
        if not self.args['use_gpu']:
            if env_var in os.environ:
                cached_env = os.environ[env_var]
            os.environ[env_var] = 'false'

        try:
            self._run(tmp_dir=tmp_dir, log=log)
        finally:
            if cached_env is not None:
                os.environ[env_var] = cached_env
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
