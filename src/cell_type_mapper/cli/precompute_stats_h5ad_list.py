"""
This is the module used to generate the precomputed_stats.h5 file
from a list of h5ad files and a CSV file denoting which cells
belong to which taxons
"""

import argschema
import copy
import h5py
import pandas as pd
import json
import time
import warnings

from cell_type_mapper.utils.output_utils import (
    get_execution_metadata)

from cell_type_mapper.utils.cli_utils import (
    config_from_args
)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree
)

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad_list_and_tree)

from cell_type_mapper.schemas.precomputation_schema import (
    PrecomputedStatsH5adListSchema)

import cell_type_mapper.utils.anndata_utils as anndata_utils


class PrecomputationH5adListRunner(argschema.ArgSchemaParser):

    default_schema = PrecomputedStatsH5adListSchema

    def run(self):
        t0 = time.time()

        metadata = {
            'config': config_from_args(
                        input_config=self.args,
                        cloud_safe=False)
        }

        df = pd.read_csv(
            self.args['annotation_path']
        )
        error_msg = ""
        for col in [self.args['qc_column'],
                    self.args['cell_label_column']] + self.args['hierarchy']:
            if col not in df.columns:
                pth = self.args['annotation_path']
                error_msg += (
                    f"column '{col}' not in file '{pth}'\n"
                )
        if len(error_msg) > 0:
            raise RuntimeError(error_msg)

        chosen_columns = [
            self.args['cell_label_column']
        ] + self.args['hierarchy']
        df[self.args['qc_column']] = df[self.args['qc_column']].astype(bool)
        df = df[df[self.args['qc_column']]][chosen_columns]

        expected_cells = set(df[self.args['cell_label_column']].values)
        actual_cells = set()
        for h5ad_path in self.args['h5ad_path_list']:
            obs = anndata_utils.read_df_from_h5ad(
                h5ad_path,
                df_name='obs'
            )
            actual_cells = actual_cells.union(
                set(obs.index.values)
            )
            del obs

        if len(expected_cells.intersection(actual_cells)) == 0:
            cell_col = self.args['cell_label_column']
            annotation_path = self.args['annotation_path']
            raise RuntimeError(
                f"No values in column '{cell_col}' of "
                f"'{annotation_path}' correspond with "
                "values in the index of your h5ad "
                "files."
            )

        omitted_cells = sorted(expected_cells-actual_cells)
        if len(omitted_cells) > 0:
            annotation_path = self.args['annotation_path']
            msg = (
                f"{len(omitted_cells)} cells from '{annotation_path}' "
                "were not present in your h5ad files and thus "
                "will not be used to create the precomputed "
                "stats file. An example of the offending cells "
                f"is\n{omitted_cells[:5]}"
            )
            warnings.warn(
                message=msg,
                category=OmittedCellsWarning
            )

        raw_tree = TaxonomyTree.from_dataframe(
            dataframe=df,
            column_hierarchy=self.args['hierarchy'],
            drop_rows=True
        )
        raw_data = copy.deepcopy(raw_tree._data)

        # patch the tree so that the leaves actually
        # have cell labels
        leaf_level = raw_tree.leaf_level
        for leaf in raw_tree.all_leaves:
            subset = df[df[leaf_level] == leaf]
            raw_data[leaf_level][leaf] = set(
                subset[self.args['cell_label_column']]
            )
        del raw_tree
        taxonomy_tree = TaxonomyTree(data=raw_data)

        precompute_summary_stats_from_h5ad_list_and_tree(
            data_path_list=self.args['h5ad_path_list'],
            taxonomy_tree=taxonomy_tree,
            cell_set=set(df[self.args['cell_label_column']].values),
            output_path=self.args['output_path'],
            rows_at_a_time=self.args['chunk_size'],
            normalization=self.args['normalization'],
            tmp_dir=self.args['tmp_dir'],
            n_processors=self.args['n_processors'],
            layer=self.args['layer'],
            gene_id_col=self.args['gene_id_col']
        )

        metadata.update(
            get_execution_metadata(
                module_file=__file__,
                t0=t0))

        with h5py.File(self.args['output_path'], 'a') as dst:
            dst.create_dataset(
                'metadata',
                data=json.dumps(metadata).encode('utf-8'))


class OmittedCellsWarning(UserWarning):
    pass


def main():
    runner = PrecomputationH5adListRunner()
    runner.run()


if __name__ == "__main__":
    main()
