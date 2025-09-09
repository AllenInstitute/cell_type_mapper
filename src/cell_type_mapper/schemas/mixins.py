import argschema
from marshmallow import post_load
import pathlib

from cell_type_mapper.utils.anndata_utils import (
    does_obsm_have_key)

from cell_type_mapper.schemas.base_schemas import (
    GeneMappingSchema)


class GeneMappingMixin(object):

    gene_mapping = argschema.fields.Nested(
        GeneMappingSchema,
        required=False,
        default=None,
        allow_none=True,
        description=(
            "Parameters used when mapping query data genes "
            "to reference data genes (if so desired)."
        )
    )


class MapToEnsemblMixin(object):

    map_to_ensembl = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description=(
            "OBSOLETE PARAMETER. Mapping of query genes to "
            "reference genes is now handled via the database "
            "specified in gene_mapping.db_path"
        )
    )

    @post_load
    def map_to_ensembl_obsolete(self, data, **kwargs):
        if data['map_to_ensembl']:
            msg = (
               "'map_to_ensembl' is no longer used. To map query "
               "genes onto reference genes, specify an mmc_gene_mapper "
               "compliant database via the gene_mapping.db_path parameter."
            )
            raise RuntimeError(msg)
        return data


class VerboseStdoutMixin(object):

    verbose_stdout = argschema.fields.Boolean(
        required=True,
        default=True,
        allow_none=False,
        description=(
            "Controls verbosity of output written to "
            "stdout (does not affect output recorded "
            "in log files)"
        )
    )


class PrecomputedStatsPathMixin(object):

    precomputed_stats_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description=(
            "Path to the precomputed_stats file for this "
            "taxonomy."
        ))


class LayerMixin(object):

    layer = argschema.fields.String(
        required=False,
        default='X',
        allow_none=False,
        description=(
            "The layer in the h5ad file from which data "
            "will be read. If 'X', will read directly from "
            "the 'X' object. If a string containing '/', e.g. "
            "'raw/X', will read directly from that layer. If a "
            "string like 'alt', then will look for the layer under "
            "layers (i.e. as 'layers/alt')."
        )
    )


class QueryPathMixinForSearch(object):

    query_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the h5ad file containing the query "
        "dataset")

    query_gene_id_col = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description=(
            "Column in the query h5ad file's var dataframe "
            "where you can find the gene identifiers. "
            "If None, use index of var."
        )
    )


class QueryPathMixinForMarkers(object):

    query_path = argschema.fields.InputFile(
        required=False,
        default=None,
        allow_none=True,
        description=(
            "Optional path to h5ad file containing query data. Used "
            "to assemble list of genes that are acceptable "
            "as markers."
        ))

    query_gene_id_col = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description=(
            "Column in the query h5ad file's var dataframe "
            "where you can find the gene identifiers. "
            "If None, use index of var."
        )
    )


class NProcessorsMixin(object):

    n_processors = argschema.fields.Int(
        required=False,
        default=32,
        allow_none=False,
        description=("Number of independent worker processes to spin up."))


class DropLevelMixin(object):

    drop_level = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description="If this level exists in the taxonomy, drop "
        "it before doing type assignment (this is to accommmodate "
        "the fact that the official taxonomy includes the "
        "'supertype', even though that level is not used "
        "during hierarchical type assignment). Note: though we "
        "use the term 'drop', it is more accurate to say that "
        "the level is ignored. It still exists in the taxonomy, "
        "but it is not directly mapped to. Mapping values will be "
        "backfilled from lower levels in the taxonomy after "
        "the mapping is complete.")


class TmpDirMixin(object):

    tmp_dir = argschema.fields.OutputDir(
        required=False,
        default=None,
        allow_none=True,
        description=("Temporary directory for writing out "
                     "scratch files"))


class OutFileWithClobberMixin(object):

    output_path = argschema.fields.OutputFile(
        required=True,
        default=None,
        allow_none=False,
        description=(
            "Path to the output file that will be written."
        ))

    clobber = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description=(
            "Set to True to allow the code to overwrite an existing file."
        ))

    @post_load
    def check_output_exists(self, data, **kwargs):
        output_path = pathlib.Path(data['output_path'])
        if output_path.exists() and not data['clobber']:
            msg = (
                f"Output file\n{output_path.resolve().absolute()}\n"
                "already exists. To overwrite, run with clobber=True"
            )
            raise RuntimeError(msg)
        return data


class OutputDstForSearchMixin(object):

    extended_result_path = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description="Path to JSON file where extended results "
        "will be saved.")

    hdf5_result_path = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description=(
            "Path to an hdf5 file where extended results "
            "will be saved. This can be a factor of 10 smaller "
            "than the extended results JSON file."
        ))

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

    verbose_csv = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description=(
            "If True, record all confidence metrics as separate columns "
            "in the CSV file. If False, only return one metric "
            "(bootstrapping_probability if available; else avg_correlation)"
        )
    )

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

    @post_load
    def check_result_dst(self, data, **kwargs):
        """
        Make sure that there is somewhere, either extended_result_path
        or obsm_key, where we can store the extended results.
        """
        output_params = (
            'extended_result_path',
            'hdf5_result_path',
            'obsm_key')
        has_output_path = False
        for param in output_params:
            if data[param] is not None:
                has_output_path = True
        if not has_output_path:
            msg = ("You must specify at least one of:\n"
                   f"{output_params}")
            raise RuntimeError(msg)
        return data


class NodesToDropMixin(object):

    nodes_to_drop = argschema.fields.List(
        argschema.fields.Tuple(
            (argschema.fields.String,
             argschema.fields.String)
        ),
        cli_as_single_argument=True,
        default=None,
        allow_none=True,
        description=(
            "Nodes to drop from the taxonomy before performing any "
            "operations. They are of the form (level, node) where "
            "level and node are strings referring to the level in the "
            "taxonomy and the node at that level to be dropped. Pass this "
            "to the CLI as a list in nested quotation marks like "
            """'[("level0", "node0"), ("level1", "node1")]' """
            "If None, the taxonomy will be used as read from the "
            "specified precomputed_stats files. Note: dropping "
            "a node from the taxonomy tree also drops all of "
            "the child nodes descended from it. If the resulting "
            "tree has parent nodes that are no longer connected "
            "to the leaf level of the taxonomy, those are dropped "
            "as well."
        )
    )
