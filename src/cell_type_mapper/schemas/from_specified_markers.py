import argschema

from cell_type_mapper.schemas.base_schemas import (
    QueryMarkerInputSchema)

from cell_type_mapper.schemas.search_mixins import (
    SearchSchemaMixin)

from cell_type_mapper.schemas.mixins import (
    NodesToDropMixin,
    VerboseStdoutMixin,
    MapToEnsemblMixin,
    GeneMappingMixin
)


class FromSpecifiedMarkersSchema(
        argschema.ArgSchema,
        SearchSchemaMixin,
        NodesToDropMixin,
        VerboseStdoutMixin,
        MapToEnsemblMixin,
        GeneMappingMixin):

    query_markers = argschema.fields.Nested(
        QueryMarkerInputSchema,
        required=True)
