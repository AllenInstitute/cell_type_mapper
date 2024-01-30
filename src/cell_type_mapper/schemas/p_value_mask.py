import argschema

from cell_type_mapper.schemas.output_file import (
    OutFileWithClobberMixin)

from cell_type_mapper.schemas.reference_marker_finder import (
    ReferenceMarkerStatsParamMixin)


class PValueMaskSchema(
        argschema.ArgSchema,
        ReferenceMarkerStatsParamMixin,
        OutFileWithClobberMixin):

    output_path = argschema.fields.OutputFile(
       required=True,
       default=None,
       allow_none=False,
       description=(
           "Path to the HDF5 file that will be written."
       ))

    precomputed_stats_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description=(
            "Path to the precomputed stats file off of which "
            "this p-value mask will be based."
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
