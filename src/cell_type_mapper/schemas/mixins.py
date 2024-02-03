import argschema
from marshmallow import post_load
import pathlib


class TmpDirMixin(object):

    tmp_dir = argschema.fields.OutputDir(
        required=False,
        default=None,
        allow_none=True,
        description=("Temporary directory for writing out "
                     "scratch files"))


class OutFileWithClobberMixin(object):

    output_path = argschema.fields.OutputFile(
        required=True,
        default=None,
        allow_none=False,
        description=(
            "Path to the output file that will be written."
        ))

    clobber = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description=(
            "Set to True to allow the code to overwrite an existing file."
        ))

    @post_load
    def check_output_exists(self, data, **kwargs):
        output_path = pathlib.Path(data['output_path'])
        if output_path.exists() and not data['clobber']:
            msg = (
                f"Output file\n{output_path.resolve().absolute()}\n"
                "already exists. To overwrite, run with clobber=True"
            )
            raise RuntimeError(msg)
        return data
