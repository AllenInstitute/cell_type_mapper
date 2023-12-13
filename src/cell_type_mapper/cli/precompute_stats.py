import argschema
import copy
import h5py
import json
import pandas as pd
import pathlib
import time

from cell_type_mapper.schemas.precomputation_schema import (
    PrecomputedStatsSchema)

from cell_type_mapper.utils.utils import (
    get_timestamp)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad_list_and_tree)

from cell_type_mapper.diff_exp.precompute_utils import (
    merge_precompute_files)


class PrecomputationRunner(argschema.ArgSchemaParser):

    default_schema = PrecomputedStatsSchema

    def run(self):

        t0 = time.time()

        dataset_to_output = self.create_dataset_to_output_map()

        metadata = {
            'config': copy.deepcopy(self.args),
            'dataset_to_output_map': dataset_to_output
        }
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

        files_to_merge = []
        for dataset in dataset_to_output.keys():
            if dataset == 'combined':
                continue

            output_path = dataset_to_output[dataset]
            files_to_merge.append(output_path)
            print(f'writing {output_path} from dataset {dataset}')
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
            metadata['dataset'] = dataset

            with h5py.File(output_path, 'a') as out_file:
                out_file.create_dataset(
                    'metadata',
                    data=json.dumps(metadata).encode('utf-8'))
            dur = time.time()-t0
            print(f'completed {dataset} after {dur:.2e} seconds')

        if 'combined' in dataset_to_output:
            print('merging')
            merged_path = pathlib.Path(
                dataset_to_output['combined'])
            merge_precompute_files(
                precompute_path_list=files_to_merge,
                output_path=merged_path)
            metadata.pop('dataset')
            metadata['timestamp'] = get_timestamp()

            with h5py.File(merged_path, 'a') as dst:
                dst.create_dataset(
                    'metadata',
                    data=json.dumps(metadata).encode('utf-8'))

            dur = time.time()-t0
            print(f'wrote {merged_path} after {dur:.2e} seconds')

    def create_dataset_to_output_map(self):
        """
        Return dict mapping dataset name to output paths
        """
        baseline_path = pathlib.Path(self.args['output_path'])
        baseline_suffix = baseline_path.suffix
        baseline_stem = baseline_path.stem
        output_parent = baseline_path.absolute().resolve().parent

        final_output_lookup = dict()
        run_default = False
        output_file_set = set()
        if self.args['split_by_dataset']:
            df = pd.read_csv(self.args['cell_metadata_path'])
            if 'dataset_label' in df.columns:
                dataset_values = set(df.dataset_label.values)

                for dataset in dataset_values:
                    sanitized = dataset.replace(" ", "_").replace("/", ".")
                    new_suffix = f'{sanitized}{baseline_suffix}'
                    new_name = f'{baseline_stem}.{new_suffix}'
                    new_path = (output_parent/new_name).resolve().absolute()
                    new_path = str(new_path)
                    if new_path in output_file_set:
                        raise RuntimeError(
                            f"Dataset labels {dataset_values} require that "
                            f"output path {new_path} occur more than once")
                    output_file_set.add(new_path)
                    final_output_lookup[dataset] = new_path
            else:
                run_default = True
        else:
            run_default = True

        if run_default:
            final_output_lookup = dict()
            output_path = pathlib.Path(self.args['output_path'])
            final_output_lookup['None'] = str(output_path.resolve().absolute())
        else:
            if 'combined' in final_output_lookup:
                raise RuntimeError(
                    "'combined' is the name of a dataset; unclear how "
                    "to proceed")
            new_suffix = f'combined{baseline_suffix}'
            new_name = f'{baseline_stem}.{new_suffix}'
            new_path = (output_parent/new_name).resolve().absolute()
            final_output_lookup['combined'] = str(new_path)

        output_path_list = list(final_output_lookup.values())
        output_path_set = set(output_path_list)
        if len(output_path_list) != len(output_path_set):
            raise RuntimeError(
                f"output path names\n{output_path_list}\nare degenerate")

        # make sure we can write here
        for dataset in final_output_lookup:
            output_path = pathlib.Path(final_output_lookup[dataset])
            if output_path.exists():
                if not self.args['clobber']:
                    raise RuntimeError(
                        f"{output_path} already exists; run with clobber=True "
                        "to overwite")

            with open(output_path, 'wb') as dst:
                dst.write(b'gar')

        return final_output_lookup


def main():
    runner = PrecomputationRunner()
    runner.run()


if __name__ == "__main__":
    main()
