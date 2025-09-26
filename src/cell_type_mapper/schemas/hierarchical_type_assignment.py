import argschema

from marshmallow.validate import OneOf
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
        default=0.5,
        allow_none=True,
        description="Factor by which to downsample the number of genes when "
        "performing bootstrapped nearest neighbor cell type searches.")

    bootstrap_factor_lookup = argschema.fields.List(
        argschema.fields.Tuple(
            (argschema.fields.String,
             argschema.fields.Float)
        ),
        required=False,
        default=None,
        allow_none=True,
        cli_as_single_argument=True,
        description=(
            "A list of tuples (level_name, value) that indicate the "
            "bootstrapping_factor values to use at each level in the "
            "taxonomy."
        )
    )

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
        default=10,
        allow_none=False,
        description=(
            "If a parent node has fewer marker genes than this, "
            "inherit the marker genes from its parent (and so on "
            "up the tree) until there are at least this many "
            "markers."
        ))

    algorithm = argschema.fields.String(
        required=False,
        default="hierarchical",
        allow_none=False,
        description=(
            "Either 'hierarchical' or 'hann'. Indicates which "
            "mapping algorithm to use."
        ),
        validate=OneOf(("hierarchical", "hann"))
    )

    @post_load
    def check_bootstrap_factor(self, data, **kwargs):
        """
        Verify that bootstrap_factor > 0 and <= 1
        and that normalization is either 'raw' or 'log2CPM'
        """
        factor = data['bootstrap_factor']

        if factor is None:
            return data

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

    @post_load
    def check_bootstrap_factor_lookup(self, data, **kwargs):
        """
        Check that only one of bootstrap_factor or
        bootstrap_factor_lookup are specified
        """
        no_factor = int((data['bootstrap_factor'] is None))
        no_lookup = int((data['bootstrap_factor_lookup'] is None))
        if no_factor + no_lookup != 1:
            msg = (
                "Must specify one and only one of 'bootstrap_factor' "
                "or 'bootstrap_factor_lookup'"
            )
            raise ValidationError(msg)
        return data


class HierarchicalTypeAssignmentSchema(
        argschema.ArgSchema,
        HierarchicalConfigMixin,
        NProcessorsMixin):
    pass


class HierarchicalTypeAssignmentSchema_noNProcessors(
        argschema.ArgSchema,
        HierarchicalConfigMixin):
    pass
