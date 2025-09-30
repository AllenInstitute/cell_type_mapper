import argschema
from marshmallow import post_load, ValidationError

from cell_type_mapper.schemas.mixins import (
    TmpDirMixin,
    DropLevelMixin,
    QueryPathMixinForSearch,
    OutputDstForSearchMixin)

from cell_type_mapper.schemas.base_schemas import (
    PrecomputedStatsInputSchema)

from cell_type_mapper.schemas.hierarchical_type_assignment import (
    HierarchicalTypeAssignmentSchema,
    HierarchicalTypeAssignmentSchema_noNProcessors)


class TypeAssignmentMixin(object):

    type_assignment = argschema.fields.Nested(
        HierarchicalTypeAssignmentSchema,
        required=True)


class TypeAssignmentMixin_noNProcessors(object):

    type_assignment = argschema.fields.Nested(
        HierarchicalTypeAssignmentSchema_noNProcessors,
        required=True)


class SearchSchemaMixinBase(
        TmpDirMixin,
        DropLevelMixin,
        QueryPathMixinForSearch,
        OutputDstForSearchMixin):

    extended_result_dir = argschema.fields.OutputDir(
        required=False,
        default=None,
        allow_none=True,
        description="Optional temporary directory into which assignment "
        "results will be saved from each process.")

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
        default=False,
        allow_nonw=False,
        description="If True, full file paths not recorded in log")

    log_path = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description="Path to the log file to be written")

    precomputed_stats = argschema.fields.Nested(
        PrecomputedStatsInputSchema,
        required=True)

    @post_load
    def align_output_files(self, data, **kwargs):
        """
        Check that algorithm selection aligns with output file
        specification
        """
        algorithm = data['type_assignment']['algorithm']
        if algorithm == 'hann':
            msg = ""
            if data['csv_result_path'] is not None:
                msg += "; you specified csv_result_path"
            if data['extended_result_path'] is not None:
                msg += "; you specified extended_result_path"
            if data['obsm_key'] is not None:
                msg += "; you specified obsm_key"
            if data['hdf5_result_path'] is None:
                msg += "; you did not specify hdf5_result_path"
            if len(msg) > 0:
                msg = (
                    "HANN algorithm can only output to hdf5_result_path"
                    f"{msg}"
                )
                raise ValidationError(msg)
        return data


class SearchSchemaMixin(
        SearchSchemaMixinBase,
        TypeAssignmentMixin):
    pass


class SearchSchemaMixin_noNProcessors(
        SearchSchemaMixinBase,
        TypeAssignmentMixin_noNProcessors):
    pass
