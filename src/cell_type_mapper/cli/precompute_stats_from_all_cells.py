import argschema
import copy
import h5py
import json
from marshmallow import post_load
import pandas as pd
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

    split_by_dataset = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description=(
            "If true, split the dataset by the 'dataset_label' field in "
            "cell_metadata.csv, storing each dataset in a separate HDF5 file. "
            "Files will be named like ouptut_path but with a secondary suffix "
            "added before .h5 specifying which dataset they contain."
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
        final_output_lookup = dict()
        run_default = False
        if data['split_by_dataset']:
            df = pd.read_csv(data['cell_metadata_path'])
            if 'dataset_label' in df.columns:
                dataset_values = set(df.dataset_label.values)
                baseline_path = pathlib.Path(data['output_path'])
                output_parent = baseline_path.absolute().resolve().parent
                baseline_suffix = baseline_path.suffix
                baseline_stem = baseline_path.stem
                for dataset in dataset_values:
                    new_suffix = f'{dataset}{baseline_suffix}'
                    new_name = f'{baseline_stem}.{new_suffix}'
                    final_output_lookup[dataset] = str(output_parent/new_name)
            else:
                run_default = True
        else:
            run_default = True

        if run_default:
            output_path = pathlib.Path(data['output_path'])
            final_output_lookup['None'] = str(output_path.resolve().absolute())

        # make sure we can write here
        for output_path in final_output_lookup:
            output_path = pathlib.Path(output_path)
            if output_path.exists():
                if not data['clobber']:
                    raise RuntimeError(
                        f"{output_path} already exists; run with clobber=True "
                        "to overwite")

            with open(output_path, 'wb') as dst:
                dst.write(b'gar')

        data['final_output_path'] = final_output_lookup
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

        cell_metadata = pd.read_csv(self.args['cell_metadata_path'])
        dataset_to_cell_set = dict()
        if 'dataset_label' in cell_metadata.columns:
            for cell_id, dataset_label in zip(
                            cell_metadata.cell_label.values,
                            cell_metadata.dataset_label.values):
                if dataset_label not in dataset_to_cell_set:
                    dataset_to_cell_set[dataset_label] = set()
                dataset_to_cell_set[dataset_label].add(cell_id)

        for dataset in self.args['final_output_path'].keys():
            output_path = self.args['final_output_path'][dataset]
            if dataset in dataset_to_cell_set:
                cell_set = dataset_to_cell_set[dataset]
            else:
                cell_set = None
            precompute_summary_stats_from_h5ad_list_and_tree(
                data_path_list=self.args['h5ad_path_list'],
                taxonomy_tree=taxonomy_tree,
                rows_at_a_time=10000,
                normalization=self.args['normalization'],
                output_path=output_path,
                cell_set=cell_set)

            metadata['timestamp'] = get_timestamp()

            with h5py.File(output_path, 'a') as out_file:
                out_file.create_dataset(
                    'metadata',
                    data=json.dumps(metadata).encode('utf-8'))


def main():
    runner = PrecomputationRunner()
    runner.run()


if __name__ == "__main__":
    main()
