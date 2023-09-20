import argschema
import copy
import h5py
import json
from marshmallow import post_load
import pathlib

from cell_type_mapper.utils.utils import (
    get_timestamp)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad_list_and_tree)


class PrecomputedStatsSchema(argschema.ArgSchema):

    h5ad_path_list = argschema.fields.List(
        argschema.fields.InputFile,
        required=True,
        default=None,
        allow_none=False,
        cli_as_single_argument=True,
        description="List of paths to h5ad files that contain the "
        "cell-by-gene data for which we are precomputing statistics")

    normalization = argschema.fields.String(
        required=False,
        default='raw',
        allow_none=False,
        description="Normalization of the h5ad files; must be either "
        "'raw' or 'log2CPM'")

    cell_metadata_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to cell_metadata.csv; the file mapping cells "
        "to clusters in our cell types taxonomy.")

    cluster_annotation_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to cluster_annotation_term.csv; the file "
        "containing parent-child reslationships within our cell types "
        "taxonomy")

    cluster_membership_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to cluster_to_cluster_annotation_membership.csv; "
        "the file containing the mapping between cluster labels and aliases "
        "in our cell types taxonomy")

    hierarchy = argschema.fields.List(
        argschema.fields.String,
        required=True,
        default=None,
        allow_none=False,
        description="List of term_set_labels in our cell types taxonomy "
        "ordered from most gross to most fine")

    output_path = argschema.fields.String(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the HDF5 file that will be written with the "
        "precomputed stats. The serialized taxonomy tree will also be "
        "saved here")

    clobber = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description=(
            "Set to True to allow the code to overwrite an existing file."
        ))

    @post_load
    def check_norm(self, data, **kwargs):
        if data['normalization'] not in ('raw', 'log2CPM'):
            raise ValueError(
                "normalization must be either 'raw' or 'log2CPM'; "
                f"you gave {data['nomralization']}")
        return data

    @post_load
    def check_output(self, data, **kwargs):
        output_path = pathlib.Path(data['output_path'])
        if output_path.exists():
            if not data['clobber']:
                raise RuntimeError(
                    f"{output_path} already exists; run with clobber=True "
                    "to overwite")

        # make sure we can write here
        with open(output_path, 'wb') as dst:
            dst.write(b'gar')

        return data


class PrecomputationRunner(argschema.ArgSchemaParser):

    default_schema = PrecomputedStatsSchema

    def run(self):

        metadata = copy.deepcopy(self.args)
        assert 'timestamp' not in metadata

        taxonomy_tree = TaxonomyTree.from_data_release(
           cell_metadata_path=self.args['cell_metadata_path'],
           cluster_annotation_path=self.args['cluster_annotation_path'],
           cluster_membership_path=self.args['cluster_membership_path'],
           hierarchy=self.args['hierarchy'])

        precompute_summary_stats_from_h5ad_list_and_tree(
            data_path_list=self.args['h5ad_path_list'],
            taxonomy_tree=taxonomy_tree,
            rows_at_a_time=10000,
            normalization=self.args['normalization'],
            output_path=self.args['output_path'])

        metadata['timestamp'] = get_timestamp()

        with h5py.File(self.args['output_path'], 'a') as out_file:
            out_file.create_dataset(
                'metadata',
                data=json.dumps(metadata).encode('utf-8'))


def main():
    runner = PrecomputationRunner()
    runner.run()


if __name__ == "__main__":
    main()
