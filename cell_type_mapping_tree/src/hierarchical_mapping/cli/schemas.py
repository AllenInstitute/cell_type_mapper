import argschema
from marshmallow import post_load, ValidationError
import pathlib


class PrecomputedStatsSchema(argschema.ArgSchema):

    path = argschema.fields.String(
                required=True,
                default=None,
                allow_none=False,
                description="The path to the file where the precomputed "
                "stats will be saved. If it already exists, this "
                "file will be read in and used as the precomputed "
                "stats file for this mapping job.")

    reference_path = argschema.fields.String(
                required=False,
                default=None,
                allow_none=True,
                description="The path to the h5ad file containing the "
                "reference dataset. Only used if precomputed_stats.path "
                "does not already exist.")

    taxonomy_tree = argschema.fields.String(
                required=False,
                default=None,
                allow_none=True,
                description="The path to the JSON file specifying the "
                "taxonomy tree for this mapping job. ONly used if "
                "precomputed_stats.path does not already exist.")

    normalization = argschema.fields.String(
                required=False,
                default=None,
                allow_none=True,
                description="The normalization of the cell by gene matrix in "
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


class SpecifiedMarkerSchema(argschema.ArgSchema):

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
        description="Normalization of the query dataset "
        "(must be 'raw' or 'log2CPM')")

    rng_seed = argschema.fields.Int(
        required=False,
        default=11235813,
        allow_none=False,
        description="Seed value for random number generator used in "
        "bootstrapping")

    @post_load
    def check_bootstrap_factor(self, data, **kwargs):
        """
        Verify that bootstrap_factor > 0 and < 1
        and that normalization is either 'raw' or 'log2CPM'
        """
        factor = data['bootstrap_factor']
        if factor <= 0.0 or factor >= 1.0:
            raise ValidationError(
                f"bootstrap_factor must be in (0, 1); you gave {factor}")

        norm = data['normalization']
        if norm not in ('raw', 'log2CPM'):
            raise ValidationError(
                f"{norm} is not a valid query normalization;\n"
                "must be either 'raw' or 'log2CP'")

        return data


class FlatTypeAssignmentSchema(argschema.ArgSchema):

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
        description="Normalization of the query dataset "
        "(must be 'raw' or 'log2CPM')")

    @post_load
    def check_normalization(self, data, **kwargs):
        """
        Verify that normalization is either 'raw' or 'log2CPM'
        """
        norm = data['normalization']
        if norm not in ('raw', 'log2CPM'):
            raise ValidationError(
                f"{norm} is not a valid query normalization;\n"
                "must be either 'raw' or 'log2CP'")

        return data
