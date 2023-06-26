"""
This module defines the CLI tool for validating an H5AD file against our
normalization and gene_id requirements
"""
import argschema
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

    output_dir = argschema.fields.OutputDir(
        required=True,
        default=None,
        allow_none=False,
        descriptipn="Directory where reformatted h5ad file "
        "will be written (if necessary)")

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
        gene_id_mapper = GeneIdMapper.from_default(log=command_log)
        result_path = validate_h5ad(
             h5ad_path=self.args['h5ad_path'],
             output_dir=self.args['output_dir'],
             gene_id_mapper=gene_id_mapper,
             log=command_log,
             tmp_dir=self.args['tmp_dir'])

        output_manifest = dict()
        if result_path is None:
            output_manifest['valid_h5ad_path'] = self.args['h5ad_path']
        else:
            result_path = str(result_path.resolve().absolute())
            output_manifest['valid_h5ad_path'] = result_path

        output_manifest['log_messages'] = command_log.log
        output_manifest['config'] = self.args
        self.output(output_manifest, indent=2)


def main():
    runner = ValidateH5adRunner()
    runner.run()


if __name__ == "__main__":
    main()
