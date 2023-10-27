import argschema
from marshmallow import post_load

from cell_type_mapper.utils.anndata_utils import (
    does_obsm_have_key)

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

    obsm_key = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description="If not None, save the results of the "
        "mapping in query_path.obsm under this key")

    obsm_clobber = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description="If True, allow the code to overwrite an "
        "existing element in query_path.obsm")

    log_path = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description="Path to the log file to be written")

    query_markers = argschema.fields.Nested(
        QueryMarkerInputSchema,
        required=True)

    @post_load
    def check_obsm_key(self, data, **kwargs):
        """
        If obsm_key is not None, make sure that key has not already
        been assigned in query_path.obsm
        """
        if data['obsm_key'] is None:
            return data

        if does_obsm_have_key(data['query_path'], data['obsm_key']):
            if not data['obsm_clobber']:
                msg = (f"obsm in {data['query_path']} already has key "
                       f"{data['obsm_key']}; to overwrite, set obsm_clobber "
                       "to True.")
                raise RuntimeError(msg)

        return data
