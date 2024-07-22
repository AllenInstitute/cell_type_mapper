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

    precomputed_stats_path_list = argschema.fields.List(
        argschema.fields.InputFile,
        required=True,
        default=None,
        allow_none=True,
        cli_as_single_argument=True,
        description=(
            "List of paths to the precomputed_stats files "
            "associated with the reference marker files "
            "specified in reference_marker_path_list. "
            "The precomputed_stats files must be in the same order "
            "as the paths in reference_marker_path_list. If "
            "precomputed_stats_path_list is None, then the "
            "precomputed_stats paths will be "
            "read directly from the metadata fields in the "
            "refernce marker files. THE PREFERRED USAGE PATTERN "
            "IS TO LEAVE precomputed_stats_path_list = None. "
            "That will ensure self-consistency between the reference "
            "marker files and the precomputed_stats files. You should "
            "only specify this field if, for some reason, the metadata "
            "fields in the reference marker files no longer point to the "
            "current locations of your precomputed_stats files.")
        )

    output_path = argschema.fields.OutputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the JSON file that will contain "
        "the marker gene lookup for the query dataset.")
