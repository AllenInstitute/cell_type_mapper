"""
This module defines the CLI tool for validating an H5AD file against our
normalization and gene_id requirements
"""
import argschema
import traceback
import pathlib
from marshmallow import post_load

from hierarchical_mapping.gene_id.gene_id_mapper import (
    GeneIdMapper)

from hierarchical_mapping.cli.cli_log import CommandLog

from hierarchical_mapping.validation.validate_h5ad import (
    validate_h5ad)


class ValidationInputSchema(argschema.ArgSchema):

    h5ad_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the h5ad file to be validated")

    valid_h5ad_path = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description="Path to the valid h5ad file")

    log_path = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description="Path to the log file to be written")

    layer = argschema.fields.String(
        required=False,
        default='X',
        allow_none=False,
        description="Layer in the h5ad file where cell by gene "
        "data is found. If not 'X', layer is relative to 'layers/'. "
        "Regardless, validated data will be written to 'X' matrix "
        "in new h5ad file.")

    output_dir = argschema.fields.OutputDir(
        required=True,
        default=None,
        allow_none=False,
        descriptipn="Directory where reformatted h5ad file "
        "will be written (if necessary). Name of file will be "
        "the same as the name of the input file, but with "
        "_VALIDATED_{timestamp}.h5ad appended.")

    tmp_dir = argschema.fields.OutputDir(
        required=False,
        default=None,
        allow_none=True,
        description="Directory where temporary data products "
        "will be written (if necessary). If None, the data "
        "products will be written where ever tempfile.mdtemp "
        "defaults to.")

    @post_load
    def check_for_output_json(self, data, **kwargs):
        is_valid = True
        if 'output_json' not in data:
            is_valid = False
        elif data['output_json'] is None:
            is_valid = False

        if not is_valid:
            raise RuntimeError(
                "must specify a path for output_json")
        return data


class ValidationOutputSchema(argschema.ArgSchema):

    valid_h5ad_path = argschema.fields.String(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the valid h5ad file, whether "
        "it is the input file (because no changes were needed) "
        "or the reformatted file created by this module")

    config = argschema.fields.Dict(
        required=True,
        default=None,
        allow_none=False,
        description="Serialization of the input configuration for "
        "this module")

    log_messages = argschema.fields.List(
        argschema.fields.String,
        required=True,
        default=None,
        allow_none=False,
        description="Messages logged while validating the "
        "h5ad file")


class ValidateH5adRunner(argschema.ArgSchemaParser):

    default_schema = ValidationInputSchema
    default_output_schema = ValidationOutputSchema

    def run(self):
        command_log = CommandLog()
        log_path = self.args['log_path']
        if log_path is not None:
            log_path = pathlib.Path(log_path)

        try:
            gene_id_mapper = GeneIdMapper.from_default(log=command_log)
            result_path, has_warnings = validate_h5ad(
                h5ad_path=self.args['h5ad_path'],
                output_dir=self.args['output_dir'],
                layer=self.args['layer'],
                gene_id_mapper=gene_id_mapper,
                log=command_log,
                tmp_dir=self.args['tmp_dir'],
                valid_h5ad_path=self.args["valid_h5ad_path"])

            output_manifest = dict()
            if result_path is None:
                output_manifest['valid_h5ad_path'] = self.args['h5ad_path']
            else:
                result_path = str(result_path.resolve().absolute())
                output_manifest['valid_h5ad_path'] = result_path

            output_manifest['log_messages'] = command_log.log
            output_manifest['config'] = self.args
            self.output(output_manifest, indent=2)
            self.has_warnings = has_warnings
        except Exception:
            traceback_msg = "an ERROR occurred ===="
            traceback_msg += f"\n{traceback.format_exc()}\n"
            command_log.add_msg(traceback_msg)
            raise
        finally:
            command_log.info("CLEANING UP")
            if log_path is not None:
                command_log.write_log(log_path)


def main():
    runner = ValidateH5adRunner()
    runner.run()


if __name__ == "__main__":
    main()
