import argschema

from cell_type_mapper.schemas.mixins import (
    TmpDirMixin,
    DropLevelMixin,
    NProcessorsMixin,
    QueryPathMixinForMarkers)


class QueryFinderConfigMixin(object):

    n_per_utility = argschema.fields.Int(
        required=False,
        default=30,
        allow_none=False,
        description="Number of marker genes to find per "
        "(taxonomy_node_A, taxonomy_node_B, up/down) combination.")

    n_per_utility_override = argschema.fields.List(
        argschema.fields.Tuple(
            (argschema.fields.String,
             argschema.fields.Integer)
        ),
        required=False,
        default=None,
        allow_none=True,
        cli_as_single_argument=True,
        description=(
            "Optional override for n_per_utilty at specific "
            "parent nodes in the taxonomy tree. Encoded as a "
            "list of ('parent_node', n_per_utility) tuples."
        ))

    genes_at_a_time = argschema.fields.Int(
        required=False,
        default=1,
        allow_none=False,
        description=(
            "Number of marker genes to choose in a single pass "
            "before updating the statistics governing marker "
            "gene selection. Higher numbers will cause the "
            "code to run faster, but may result in more markers "
            "being selected than are strictly necessary."
        ))


class QueryMarkerFinderSchema(
        argschema.ArgSchema,
        QueryFinderConfigMixin,
        TmpDirMixin,
        DropLevelMixin,
        NProcessorsMixin,
        QueryPathMixinForMarkers):

    output_path = argschema.fields.OutputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the JSON file that will contain "
        "the marker gene lookup for the query dataset.")

    reference_marker_path_list = argschema.fields.List(
        argschema.fields.InputFile,
        required=True,
        default=None,
        allow_none=False,
        cli_as_single_argument=True,
        description=(
            "List of reference marker files to use "
            "when creating this query marker file.")
        )

    search_for_stats_file = argschema.fields.Boolean(
        required=True,
        default=False,
        allow_none=False,
        description=(
            "If True, look for the precomputed_stats file associated "
            "with a reference_marker file in the same directory where "
            "the reference_marker file is stored. This is meant for use "
            "in cloud environments where the absolute paths of files have "
            "been changed due to movement in and out of cloud storage."
        )
    )
