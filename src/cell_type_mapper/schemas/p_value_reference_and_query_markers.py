"""
This module defines the schema for the CLI tool that takes a p_value_mask file
and computes the query markers 'directly,' storing reference markers in a
temporary file that is deleted upon completion.
"""

import argschema

from cell_type_mapper.schemas.mixins import (
    NProcessorsMixin,
    QueryPathMixinForMarkers,
    DropLevelMixin,
    TmpDirMixin,
    OutFileWithClobberMixin)

from cell_type_mapper.schemas.reference_marker_finder import (
    MaxGBMixin)

from cell_type_mapper.schemas.p_value_markers import (
    PValueMaskPathMixin)

from cell_type_mapper.schemas.query_marker_finder import (
    QueryFinderConfigMixin)


class ReferenceMarkerStageSchema(
        argschema.ArgSchema):

    n_valid = argschema.fields.Int(
        required=False,
        default=None,
        allow_none=True,
        description=("Try to find this many marker genes per pair. "
                     "Used only if exact_penetrance is False. "
                     "If None, value will be inferred from "
                     "query_markers.n_per_utility."))


class QueryMarkerStageSchema(
        argschema.ArgSchema,
        QueryFinderConfigMixin):

    pass


class QueryMarkersFromPValueMaskSchema(
        argschema.ArgSchema,
        PValueMaskPathMixin,
        QueryPathMixinForMarkers,
        OutFileWithClobberMixin,
        DropLevelMixin,
        NProcessorsMixin,
        TmpDirMixin,
        MaxGBMixin):

    reference_markers = argschema.fields.Nested(
        ReferenceMarkerStageSchema,
        required=False)

    query_markers = argschema.fields.Nested(
        QueryMarkerStageSchema,
        required=False)
