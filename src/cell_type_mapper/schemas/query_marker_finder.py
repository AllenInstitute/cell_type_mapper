import argschema


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


class QueryMarkerFinderSchema(
        argschema.ArgSchema,
        QueryFinderConfigMixin):

    query_path = argschema.fields.InputFile(
        required=False,
        default=None,
        allow_none=True,
        description=(
            "Path to the h5ad file containing the query "
            "dataset (used to read the list of available genes). "
            "If None, will assume any gene that occurs in all of the "
            "reference marker files is a legal choice."
        ))

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

    n_processors = argschema.fields.Int(
        required=False,
        default=32,
        allow_none=False,
        description="Number of independendent processes to use when "
        "parallelizing work for mapping job")

    drop_level = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description="If this level exists in the taxonomy, drop "
        "it before doing type assignment (this is to accommmodate "
        "the fact that the official taxonomy includes the "
        "'supertype', even though that level is not used "
        "during hierarchical type assignment")

    output_path = argschema.fields.OutputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the JSON file that will contain "
        "the marker gene lookup for the query dataset.")

    tmp_dir = argschema.fields.OutputDir(
        required=False,
        default=None,
        allow_none=True,
        description="Optional temporary directory for scratch files.")
