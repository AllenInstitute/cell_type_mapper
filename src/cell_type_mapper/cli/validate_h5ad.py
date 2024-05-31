"""
This module defines the CLI tool for validating an H5AD file against our
normalization and gene_id requirements
"""
import argschema
import traceback
import pathlib
import shutil
from marshmallow import post_load

from cell_type_mapper.cli.cli_log import CommandLog

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad,
    update_uns)

from cell_type_mapper.validation.validate_h5ad import (
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
        description="Path to the valid h5ad file that will be "
        "written by this tool. If this is not specified, the tool "
        "will write the file to the location specified by the "
        "output_dir config parameter, appending "
        "_VALIDATED_{timestamp} to the name of the input file. "
        "If it is specified, this file will be written, "
        "even if it is just a straight copy of the input file.")

    log_path = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description="Optional path to a log file where this tool "
        "will record logging messages produced during validation.")

    layer = argschema.fields.String(
        required=False,
        default='X',
        allow_none=False,
        description="Layer in the h5ad file where cell by gene "
        "data is found. If not 'X', layer is relative to 'layers/'. "
        "Regardless, validated data will be written to 'X' matrix "
        "in new h5ad file.")

    round_to_int = argschema.fields.Bool(
        required=False,
        default=True,
        allow_none=False,
        description="If True, the X matrix of the validated h5ad file "
        "will be a form of integer. If False, it will contain the same "
        "cell by gene data as the input h5ad file.")

    check_max = argschema.fields.Bool(
        required=False,
        default=True,
        allow_none=False,
        description="If true, check that the maximum value of "
        "the data is >= 20; if not, warn the user that the data "
        "may actually be log normalized")

    output_dir = argschema.fields.OutputDir(
        required=False,
        default=None,
        allow_none=True,
        descriptipn="Directory where reformatted h5ad file "
        "will be written (if valid_h5ad_path not specified). "
        "Name of file will be the same as the name of the input "
        "file, but with _VALIDATED_{timestamp}.h5ad appended.")

    tmp_dir = argschema.fields.OutputDir(
        required=False,
        default=None,
        allow_none=True,
        description="Directory where temporary data products "
        "will be written (if necessary). If None, the data "
        "products will be written where ever tempfile.mkdtemp "
        "defaults to.")

    cloud_safe = argschema.fields.Boolean(
        required=False,
        default=True,
        allow_nonw=False,
        description="If True, full file paths not recorded in log")

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

    @post_load
    def check_output_destination(self, data, **kwargs):
        """
        Check that one and only one of valid_h5ad_path and output_dir
        are specified
        """
        output_dir = data['output_dir']
        valid_path = data['valid_h5ad_path']
        if output_dir is None and valid_path is None:
            raise RuntimeError(
                "Must specify one of either output_dir or valid_h5ad_path")
        if output_dir is not None and valid_path is not None:
            raise RuntimeError(
                "Can only specify one of output_dir or valid_h5ad_path")
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
            if self.args['check_max']:
                expected_max = 20
            else:
                expected_max = None

            result_path, has_warnings = validate_h5ad(
                h5ad_path=self.args['h5ad_path'],
                output_dir=self.args['output_dir'],
                layer=self.args['layer'],
                gene_id_mapper=None,
                log=command_log,
                expected_max=expected_max,
                tmp_dir=self.args['tmp_dir'],
                valid_h5ad_path=self.args["valid_h5ad_path"],
                round_to_int=self.args["round_to_int"])

            output_manifest = dict()
            if result_path is None:
                if self.args['valid_h5ad_path'] is not None:
                    new_path = self.args['valid_h5ad_path']
                    shutil.copy(
                        src=self.args['h5ad_path'],
                        dst=new_path)
                    # need to update uns to contain the number of mapped genes
                    n_genes = len(read_df_from_h5ad(new_path, df_name='var'))
                    update_uns(
                        new_path,
                        new_uns={'AIBS_CDM_n_mapped_genes': n_genes},
                        clobber=False)
                    output_manifest['valid_h5ad_path'] = new_path
                else:
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
                command_log.write_log(
                    log_path,
                    cloud_safe=self.args['cloud_safe'])


def main():
    runner = ValidateH5adRunner()
    runner.run()


if __name__ == "__main__":
    main()
