import argschema

from cell_type_mapper.schemas.base_schemas import (
    QueryMarkerInputSchema)

from cell_type_mapper.schemas.search_mixins import (
    SearchSchemaMixin)

from cell_type_mapper.schemas.mixins import (
    NodesToDropMixin
)


class FromSpecifiedMarkersSchema(
        argschema.ArgSchema,
        SearchSchemaMixin,
        NodesToDropMixin):

    map_to_ensembl = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description="If True, map the gene names in query_path to "
        "ENSEMBL IDs before performing cell type mapping.")

    log_path = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description="Path to the log file to be written")

    query_markers = argschema.fields.Nested(
        QueryMarkerInputSchema,
        required=True)
