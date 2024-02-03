import argschema

from cell_type_mapper.schemas.mixins import (
    TmpDirMixin)

from cell_type_mapper.schemas.base_schemas import (
    PrecomputedStatsInputSchema)

from cell_type_mapper.schemas.hierarchical_type_assignment import (
    HierarchicalTypeAssignmentSchema)


class SearchSchemaMixin(TmpDirMixin):

    query_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the h5ad file containing the query "
        "dataset")

    extended_result_path = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description="Path to JSON file where extended results "
        "will be saved.")

    extended_result_dir = argschema.fields.OutputDir(
        required=False,
        default=None,
        allow_none=True,
        description="Optional temporary directory into which assignment "
        "results will be saved from each process.")

    csv_result_path = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description="Path to CSV file where output file will be "
        "written (if None, no CSV will be produced).")

    summary_metadata_path = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description=(
            "If not None, the path to a JSON file where summary "
            "metadata (e.g. number of mapped genes and number "
            "of mapped cells) will be stored")
        )

    drop_level = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description="If this level exists in the taxonomy, drop "
        "it before doing type assignment (this is to accommmodate "
        "the fact that the official taxonomy includes the "
        "'supertype', even though that level is not used "
        "during hierarchical type assignment")

    flatten = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description="If true, flatten the taxonomy so that we are "
        "mapping directly to the leaf node")

    max_gb = argschema.fields.Float(
        required=False,
        default=100.0,
        allow_none=False,
        description="In the event that a CSC matrix needs to be "
        "converted to a temporary on disk CSR matrix, how "
        "much memory (in gigabytes) can we use.")

    cloud_safe = argschema.fields.Boolean(
        required=False,
        default=True,
        allow_nonw=False,
        description="If True, full file paths not recorded in log")

    type_assignment = argschema.fields.Nested(
        HierarchicalTypeAssignmentSchema,
        required=True)

    precomputed_stats = argschema.fields.Nested(
        PrecomputedStatsInputSchema,
        required=True)
