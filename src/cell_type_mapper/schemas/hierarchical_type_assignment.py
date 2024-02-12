import argschema

from marshmallow import post_load, ValidationError

from cell_type_mapper.schemas.mixins import (
    NProcessorsMixin)


class HierarchicalConfigMixin(object):

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
        default=5,
        allow_none=False,
        description="The number of runner up node assignments "
        "to record at each level of the taxonomy.")

    min_markers = argschema.fields.Int(
        required=False,
        default=40,
        allow_none=False,
        description=(
            "If a parent node has fewer marker genes than this, "
            "inherit the marker genes from its parent (and so on "
            "up the tree) until there are at least this many "
            "markers."
        ))

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


class HierarchicalTypeAssignmentSchema(
        argschema.ArgSchema,
        HierarchicalConfigMixin,
        NProcessorsMixin):

    pass
