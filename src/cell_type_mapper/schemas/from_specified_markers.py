import argschema

from cell_type_mapper.schemas.base_schemas import (
    QueryMarkerInputSchema)

from cell_type_mapper.schemas.search_mixins import (
    SearchSchemaMixin)

from cell_type_mapper.schemas.mixins import (
    NodesToDropMixin,
    VerboseStdoutMixin
)


class FromSpecifiedMarkersSchema(
        argschema.ArgSchema,
        SearchSchemaMixin,
        NodesToDropMixin,
        VerboseStdoutMixin):

    map_to_ensembl = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description="If True, map the gene names in query_path to "
        "ENSEMBL IDs before performing cell type mapping.")

    query_markers = argschema.fields.Nested(
        QueryMarkerInputSchema,
        required=True)
