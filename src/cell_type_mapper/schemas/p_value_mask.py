import argschema

from cell_type_mapper.schemas.mixins import (
    OutFileWithClobberMixin,
    TmpDirMixin,
    NProcessorsMixin,
    PrecomputedStatsPathMixin)

from cell_type_mapper.schemas.reference_marker_finder import (
    ReferenceMarkerStatsParamMixin)


class PValueMaskSchema(
        argschema.ArgSchema,
        ReferenceMarkerStatsParamMixin,
        OutFileWithClobberMixin,
        TmpDirMixin,
        NProcessorsMixin,
        PrecomputedStatsPathMixin):

    output_path = argschema.fields.OutputFile(
       required=True,
       default=None,
       allow_none=False,
       description=(
           "Path to the HDF5 file that will be written."
       ))

    rows_at_a_time = argschema.fields.Integer(
        required=False,
        default=10000,
        allow_none=False,
        description=(
            "Number of cluster pairs each worker should "
            "process at a time."
        ))
