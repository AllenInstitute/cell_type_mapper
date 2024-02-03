import argschema

from cell_type_mapper.schemas.mixins import (
    OutFileWithClobberMixin,
    TmpDirMixin)

from cell_type_mapper.schemas.reference_marker_finder import (
    NValidMixin,
    ReferenceRunnerConfigMixin)


class PValueMarkersSchema(
        argschema.ArgSchema,
        OutFileWithClobberMixin,
        NValidMixin,
        ReferenceRunnerConfigMixin,
        TmpDirMixin):

    precomputed_stats_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description=(
            "Path to the precomputed_stats file from which "
            "these markers are being derived."
        ))

    p_value_mask_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description=(
            "Path to the p-value mask file from which "
            "these markers are being derived."
        ))

    output_path = argschema.fields.OutputFile(
        required=True,
        default=None,
        allow_none=False,
        description=(
            "Path to the HDF5 file that will be written."
        ))
