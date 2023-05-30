import argschema
from marshmallow import post_load, ValidationError
import pathlib


class PrecomputedStatsSchema(argschema.ArgSchema):

    path = argschema.fields.String(
                required=True,
                default=None,
                allow_none=False,
                help="The path to the file where the precomputed "
                "stats will be saved. If it already exists, this "
                "file will be read in and used as the precomputed "
                "stats file for this mapping job.")

    reference_path = argschema.fields.String(
                required=False,
                default=None,
                allow_none=True,
                help="The path to the h5ad file containing the reference "
                "dataset. Only used if precomputed_stats.path does not "
                "already exist.")

    taxonomy_tree = argschema.fields.String(
                required=False,
                default=None,
                allow_none=True,
                help="The path to the JSON file specifying the taxonomy "
                "tree for this mapping job. ONly used if "
                "precomputed_stats.path does not already exist.")

    normalization = argschema.fields.String(
                required=False,
                default=None,
                allow_none=True,
                help="The normalization of the cell by gene matrix in "
                "the reference dataset. Must be ('raw' or 'log2CPM'). "
                "Only used if precomputed_stats.path does not already "
                "exist.")

    @post_load
    def check_all_paths(self, data, **kwargs):
        output_path = pathlib.Path(data['path'])
        if output_path.is_file():
            # the precomputed stats file already exists;
            # none of the other parameters actually matter
            return data

        reference_path = pathlib.Path(data['reference_path'])
        normalization = data['normalization']
        taxonomy_tree = pathlib.Path(data['taxonomy_tree'])
        error_msg = ''

        if not output_path.parent.is_dir():
            error_msg += f"{output_path.parent} is not a valid dir;\n"
            error_msg += "    will not be able to write output."

        if reference_path is None:
            error_msg += "precomputed_stats.path does not exist; must "
            error_msg += "specify a reference_file\n"
        elif not reference_path.is_file():
            error_msg += f"{reference_path}\nis not a valid file\n"

        if taxonomy_tree is None:
            error_msg += "precomputed_stats.path does not exist; must "
            error_msg += "specify a taxonomy_tree\n"
        elif not taxonomy_tree.is_file():
            error_msg += f"{taxonomy_tree}\nis not a valid file\n"

        if normalization is None:
            error_msg += "precomputed_stats.path does not exist; must "
            error_msg += "specify a normalization for the reference file\n"
        elif normalization not in ('raw', 'log2CPM'):
            error_msg += f"{normalization} is not a valid normalization\n"
            error_msg += "    must be either 'raw' or 'log2CPM'\n"

        if len(error_msg) > 0:
            raise ValidationError(error_msg)

        return data
