import argschema

from cell_type_mapper.schemas.base_schemas import (
    QueryMarkerInputSchema)

from cell_type_mapper.schemas.search_mixins import (
    SearchSchemaMixin)


class FromSpecifiedMarkersSchema(
        argschema.ArgSchema,
        SearchSchemaMixin):

    map_to_ensembl = argschema.fields.Boolean(
        reauired=False,
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
