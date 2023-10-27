import argschema
from marshmallow import post_load, ValidationError

from cell_type_mapper.utils.anndata_utils import (
    does_obsm_have_key)


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


class HierarchicalTypeAssignmentSchema(argschema.ArgSchema):

    bootstrap_iteration = argschema.fields.Int(
        required=False,
        default=100,
        allow_none=False,
        description="Number of bootstrap nearest neighbor iterations to run "
        "when assigning cell types.")

    bootstrap_factor = argschema.fields.Float(
        required=False,
        default=0.9,
        allow_none=False,
        description="Factor by which to downsample the number of genes when "
        "performing bootstrapped nearest neighbor cell type searches.")

    n_processors = argschema.fields.Int(
        required=False,
        default=32,
        allow_none=False,
        description="Number of independendent processes to use when "
        "parallelizing work for mapping job")

    chunk_size = argschema.fields.Int(
        required=False,
        default=10000,
        allow_none=False,
        description="Number of rows each worker process should load at "
        "a time from the query dataset")

    normalization = argschema.fields.String(
        required=True,
        default=None,
        allow_none=False,
        description="Normalization of the query dataset. "
        "Must be 'raw' or 'log2CPM'. If 'raw', the code will "
        "convert the data to log2(CPM+1) before mapping. "
        "If 'log2CPM', the code will use the query data as-is "
        "without applying further normalization.")

    rng_seed = argschema.fields.Int(
        required=False,
        default=11235813,
        allow_none=False,
        description="Seed value for random number generator used in "
        "bootstrapping")

    n_runners_up = argschema.fields.Int(
        required=False,
        default=0,
        allow_none=False,
        dsecription="The number of runner up node assignments "
        "to record at each level of the taxonomy.")

    @post_load
    def check_bootstrap_factor(self, data, **kwargs):
        """
        Verify that bootstrap_factor > 0 and <= 1
        and that normalization is either 'raw' or 'log2CPM'
        """
        factor = data['bootstrap_factor']
        eps = 1.0e-6
        if factor <= 0.0 or factor > 1.0+eps:
            raise ValidationError(
                f"bootstrap_factor must be in (0, 1); you gave {factor}")

        norm = data['normalization']
        if norm not in ('raw', 'log2CPM'):
            raise ValidationError(
                f"{norm} is not a valid query normalization;\n"
                "must be either 'raw' or 'log2CP'")

        return data


class FromSpecifiedMarkersSchema(argschema.ArgSchema):

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

    map_to_ensembl = argschema.fields.Boolean(
        reauired=False,
        default=False,
        allow_none=False,
        description="If True, map the gene names in query_path to "
        "ENSEMBL IDs before performing cell type mapping.")

    extended_result_dir = argschema.fields.OutputDir(
        required=False,
        default=None,
        allow_none=True,
        description="Optional temporary directory into which assignment "
        "results will be saved from each process.")

    extended_result_path = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description="Path to JSON file where extended results "
        "will be saved.")

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

    csv_result_path = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description="Path to CSV file where output file will be "
        "written (if None, no CSV will be produced).")

    log_path = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description="Path to the log file to be written")

    max_gb = argschema.fields.Float(
        required=False,
        default=100.0,
        allow_none=False,
        description="In the event that a CSC matrix needs to be "
        "converted to a temporary on disk CSR matrix, how "
        "much memory (in gigabytes) can we use.")

    drop_level = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description="If this level exists in the taxonomy, drop "
        "it before doing type assignment (this is to accommmodate "
        "the fact that the official taxonomy includes the "
        "'supertype', even though that level is not used "
        "during hierarchical type assignment")

    precomputed_stats = argschema.fields.Nested(
        PrecomputedStatsInputSchema,
        required=True)

    query_markers = argschema.fields.Nested(
        QueryMarkerInputSchema,
        required=True)

    type_assignment = argschema.fields.Nested(
        HierarchicalTypeAssignmentSchema,
        required=True)

    flatten = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description="If true, flatten the taxonomy so that we are "
        "mapping directly to the leaf node")

    cloud_safe = argschema.fields.Boolean(
        required=False,
        default=True,
        allow_nonw=False,
        description="If True, full file paths not recorded in log")

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
