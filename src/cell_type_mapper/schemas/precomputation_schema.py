import argschema
from marshmallow import post_load

from cell_type_mapper.schemas.mixins import (
    OutFileWithClobberMixin,
    TmpDirMixin,
    NProcessorsMixin,
    LayerMixin)


class PrecomputedStatsSchemaMixin(
        OutFileWithClobberMixin,
        TmpDirMixin,
        NProcessorsMixin,
        LayerMixin):

    clobber = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description=(
            "Set to True to allow the code to overwrite an existing file."
        ))

    hierarchy = argschema.fields.List(
        argschema.fields.String,
        required=True,
        default=None,
        allow_none=False,
        cli_as_single_argument=True,
        description="List of term_set_labels in our cell types taxonomy "
        "ordered from most gross to most fine")

    normalization = argschema.fields.String(
        required=False,
        default='raw',
        allow_none=False,
        description="Normalization of the h5ad files; must be either "
        "'raw' or 'log2CPM'")

    output_path = argschema.fields.String(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the HDF5 file that will be written with the "
        "precomputed stats. The serialized taxonomy tree will also be "
        "saved here")

    @post_load
    def check_norm(self, data, **kwargs):
        if data['normalization'] not in ('raw', 'log2CPM'):
            raise ValueError(
                "normalization must be either 'raw' or 'log2CPM'; "
                f"you gave {data['nomralization']}")
        return data


class PrecomputedStatsScrattchSchema(
        PrecomputedStatsSchemaMixin,
        argschema.ArgSchema):

    h5ad_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description=(
            "Path to the h5ad file containing the cell-by-gene "
            "data along with the taxonomy (stored in the obs "
            "dataframe)."
        ))


class PrecomputedStatsABCSchema(
        PrecomputedStatsSchemaMixin,
        argschema.ArgSchema):

    h5ad_path_list = argschema.fields.List(
        argschema.fields.InputFile,
        required=True,
        default=None,
        allow_none=False,
        cli_as_single_argument=True,
        description="List of paths to h5ad files that contain the "
        "cell-by-gene data for which we are precomputing statistics")

    cell_metadata_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to cell_metadata.csv; the file mapping cells "
        "to clusters in our cell types taxonomy.")

    cell_to_cluster_path = argschema.fields.InputFile(
        requred=False,
        default=None,
        allow_none=True,
        description=(
            "Path to cell_to_cluster_membership.csv. If provided, "
            "the code will get the mapping from cell_label to "
            "cluster_alias from this file, limiting itself to the "
            "cells in cell_metadata.csv. If not provided, the code "
            "will assume that `cluster_alias` is a column in "
            "cell_metadata.csv"
        )
    )

    cluster_annotation_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to cluster_annotation_term.csv; the file "
        "containing parent-child relationships within our cell types "
        "taxonomy")

    cluster_membership_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to cluster_to_cluster_annotation_membership.csv; "
        "the file containing the mapping between cluster labels and aliases "
        "in our cell types taxonomy")

    split_by_dataset = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description=(
            "If true, split the dataset by the 'dataset_label' field in "
            "cell_metadata.csv, storing each dataset in a separate HDF5 file. "
            "Files will be named like output_path but with a secondary suffix "
            "added before .h5 specifying which dataset they contain."
        ))

    do_pruning = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description=(
            "If True, remove all nodes from the tree that are "
            "not directly connected to the leaf level of the tree. This "
            "is useful, for instance, when creating a taxonomy tree based "
            "only on cells from a subset of a wider data release which may "
            "not include all of the cell types identified in the full "
            "taxonomy."
        )
    )
