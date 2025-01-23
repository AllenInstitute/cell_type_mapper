import argschema

from cell_type_mapper.schemas.base_schemas import (
    QueryMarkerInputSchema)

from cell_type_mapper.schemas.search_mixins import (
    SearchSchemaMixin)

from cell_type_mapper.schemas.mixins import (
    NodesToDropMixin,
    VerboseStdoutMixin,
    MapToEnsemblMixin
)


class FromSpecifiedMarkersSchema(
        argschema.ArgSchema,
        SearchSchemaMixin,
        NodesToDropMixin,
        VerboseStdoutMixin,
        MapToEnsemblMixin):

    query_markers = argschema.fields.Nested(
        QueryMarkerInputSchema,
        required=True)
