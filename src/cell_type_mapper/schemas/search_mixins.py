import argschema
from marshmallow import post_load

from cell_type_mapper.schemas.base_schemas import (
    PrecomputedStatsInputSchema)

from cell_type_mapper.schemas.hierarchical_type_assignment import (
    HierarchicalTypeAssignmentSchema)


class SearchSchemaMixin(object):

    tmp_dir = argschema.fields.OutputDir(
        required=False,
        default=None,
        allow_none=True,
        description="Optional temporary directory into which data "
        "will be copied for faster access (e.g. if the data "
        "naturally lives on a slow NFS drive)")

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

    @post_load
    def check_result_dst(self, data, **kwargs):
        """
        Make sure that there is somewhere, either extended_result_path
        or obsm_key, where we can store the extended results.
        """
        if data['extended_result_path'] is None:
            if data['obsm_key'] is None:
                msg = ("You must specify at least one of extended_result_path "
                       "and/or obsm_key")
                raise RuntimeError(msg)
        return data
