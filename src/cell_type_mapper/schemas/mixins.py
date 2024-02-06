import argschema
from marshmallow import post_load
import pathlib


class PrecomputedStatsPathMixin(object):

    precomputed_stats_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description=(
            "Path to the precomputed_stats file for this "
            "taxonomy."
        ))


class QueryPathMixinForSearch(object):

    query_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the h5ad file containing the query "
        "dataset")


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
        "during hierarchical type assignment")


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
