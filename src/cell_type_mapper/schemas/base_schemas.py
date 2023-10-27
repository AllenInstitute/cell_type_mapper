import argschema


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
        allow_none=False,
        description="Path to the JSON file that specifies the marker genes to "
        "use for this mapping job.")
