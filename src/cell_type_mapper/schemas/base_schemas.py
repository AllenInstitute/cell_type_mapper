import argschema
from marshmallow import post_load


class PrecomputedStatsInputSchema(argschema.ArgSchema):

    path = argschema.fields.InputFile(
                required=True,
                default=None,
                allow_none=False,
                description="The path to the file where the precomputed "
                "stats will be saved. If it already exists, this "
                "file will be read in and used as the precomputed "
                "stats file for this mapping job.")


class QueryMarkerInputSchema(argschema.ArgSchema):

    serialized_lookup = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=True,
        description="Path to the JSON file that specifies the marker genes to "
        "use for this mapping job.")

    collapse_markers = argschema.fields.Boolean(
        required=True,
        default=False,
        description=(
            "If True and serialized_lookup is not None, all marker genes "
            "will be compiled into a single list that is used at all levels "
            "in the taxonomy. If True and serialized_lookup is None, all "
            "genes in the unlabeled dataset will be used as markers at all "
            "levels of the taxonomy. Cannot be False if serialized_lookup "
            "is None."
        )
    )

    @post_load
    def check_collapse_consistence(self, data, **kwargs):
        if data['serialized_lookup'] is None:
            if not data['collapse_markers']:
                raise RuntimeError(
                    "Cannot have collapse_markers = False if you "
                    "are not specifying a serialized_lookup for "
                    "query_markers."
                )
        return data


class GeneMappingSchema(argschema.ArgSchema):

    db_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description=(
            "Path to sqlite db file that mmc_gene_mapper "
            "will use when mapping query data genes "
            "to reference data genes."
        )
    )
