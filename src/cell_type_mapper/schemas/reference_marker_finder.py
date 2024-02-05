import argschema

from cell_type_mapper.schemas.mixins import (
    TmpDirMixin,
    DropLevelMixin,
    NProcessorsMixin,
    QueryPathMixinForMarkers)


class ReferenceMarkerStatsParamMixin(object):

    p_th = argschema.fields.Float(
        required=False,
        default=0.01,
        allow_none=False,
        description=("The corrected p-value that a gene's distribution "
                     "differs between two clusters must be less than this "
                     "for that gene to be considered a marker gene."))

    q1_th = argschema.fields.Float(
        required=False,
        default=0.5,
        allow_none=False,
        description=("Threshold on q1 (fraction of cells in at "
                     "least one cluster of a pair that express "
                     "a gene above 1 CPM) for a gene to be considered "
                     "a marker"))

    q1_min_th = argschema.fields.Float(
        required=False,
        default=0.1,
        allow_none=False,
        description=("If q1 less than this value, a gene "
                     "cannot be considered a marker, even if "
                     "exact_penetrance is False"))

    qdiff_th = argschema.fields.Float(
        required=False,
        default=0.7,
        allow_none=False,
        description=("Threshold on qdiff (differential penetrance) "
                     "above which a gene is considered a marker gene"))

    qdiff_min_th = argschema.fields.Float(
        required=False,
        default=0.1,
        allow_none=False,
        description=("If qdiff less than this value, a gene "
                     "cannot be considered a marker, even if "
                     "exact_penetrance is False"))

    log2_fold_th = argschema.fields.Float(
        required=False,
        default=1.0,
        allow_none=False,
        description=("The log2 fold change of a gene between two "
                     "clusters should be above this for that gene "
                     "to be considered a marker gene"))

    log2_fold_min_th = argschema.fields.Float(
        required=False,
        default=0.8,
        allow_none=False,
        description=("If the log2 fold change of a gene between two "
                     "clusters is less than this value, that gene cannot "
                     "be a marker, even if exact_penetrance is False"))


class NValidMixin(object):

    n_valid = argschema.fields.Int(
        required=False,
        default=30,
        allow_none=False,
        description=("Try to find this many marker genes per pair. "
                     "Used only if exact_penetrance is False."))


class ReferenceFinderConfigMixin(
        ReferenceMarkerStatsParamMixin,
        NValidMixin):

    exact_penetrance = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description=("If False, allow genes that technically fail "
                     "penetrance and fold-change thresholds to pass "
                     "through as reference genes."))

    cloud_safe = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_nonw=False,
        description="If True, full file paths not recorded in log")


class ReferenceRunnerConfigMixin(
        NProcessorsMixin,
        QueryPathMixinForMarkers):

    max_gb = argschema.fields.Int(
        required=False,
        default=20,
        allow_none=False,
        description=(
            "Total amount of memory (in GB) the process is "
            "allowed to consume (approximate)."
        ))


class ReferenceMarkerFinderSchema(
        argschema.ArgSchema,
        ReferenceFinderConfigMixin,
        ReferenceRunnerConfigMixin,
        TmpDirMixin,
        DropLevelMixin):

    precomputed_path_list = argschema.fields.List(
        argschema.fields.InputFile,
        required=True,
        default=None,
        allow_none=False,
        cli_as_single_argument=True,
        description=(
            "List of paths to precomputed stats files "
            "for which reference markers will be computed"))

    output_dir = argschema.fields.OutputDir(
        required=True,
        default=None,
        allow_none=False,
        description=(
            "Path to directory where refernce marker files "
            "will be written. Specific file names will be inferred "
            "from precomputed stats files."))

    clobber = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description=("If False, do not allow overwrite of existing "
                     "output files."))
