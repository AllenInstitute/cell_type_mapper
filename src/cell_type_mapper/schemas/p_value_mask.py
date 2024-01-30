import argschema
from marshmallow import post_load
import pathlib

from cell_type_mapper.schemas.reference_marker_finder import (
    ReferenceMarkerParamMixin)


class PValueMaskSchema(
        argschema.ArgSchema,
        ReferenceMarkerParamMixin):

    precomputed_stats_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description=(
            "Path to the precomputed stats file off of which "
            "this p-value mask will be based."
        ))

    output_path = argschema.fields.OutputFile(
        required=True,
        default=None,
        allow_none=False,
        description=(
            "Path to the HDF5 file that will be written."
        ))

    n_processors = argschema.fields.Integer(
        required=False,
        default=3,
        allow_none=False,
        description=(
            "Number of worker processes to spin up."
        ))

    tmp_dir = argschema.fields.OutputDir(
        required=False,
        default=None,
        allow_none=True,
        description=(
            "Directory where temprorary scratch files will be written out."
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
