import argschema

from cell_type_mapper.schemas.mixins import (
    OutFileWithClobberMixin,
    TmpDirMixin)

from cell_type_mapper.schemas.reference_marker_finder import (
    ReferenceMarkerStatsParamMixin)


class PValueMaskSchema(
        argschema.ArgSchema,
        ReferenceMarkerStatsParamMixin,
        OutFileWithClobberMixin,
        TmpDirMixin):

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

    rows_at_a_time = argschema.fields.Integer(
        required=False,
        default=10000,
        allow_none=False,
        description=(
            "Number of cluster pairs each worker should "
            "process at a time."
        ))
