import argschema
import copy
import h5py
import json
import time

from cell_type_mapper.utils.output_utils import (
    get_execution_metadata)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad)

from cell_type_mapper.diff_exp.p_value_markers import (
    find_markers_for_all_taxonomy_pairs_from_p_mask)

from cell_type_mapper.schemas.p_value_markers import (
    PValueMarkersSchema)


class PValueMarkersRunner(argschema.ArgSchemaParser):

    default_schema = PValueMarkersSchema

    def run(self):
        if self.args['query_path'] is not None:
            gene_list = list(
                read_df_from_h5ad(
                    self.args['query_path'],
                    df_name='var').index.values)
        else:
            gene_list = None

        metadata = {'config': copy.deepcopy(self.args)}

        t0 = time.time()
        find_markers_for_all_taxonomy_pairs_from_p_mask(
            precomputed_stats_path=self.args['precomputed_stats_path'],
            p_value_mask_path=self.args['p_value_mask_path'],
            output_path=self.args['output_path'],
            n_processors=self.args['n_processors'],
            tmp_dir=self.args['tmp_dir'],
            max_gb=self.args['max_gb'],
            n_valid=self.args['n_valid'],
            gene_list=gene_list,
            drop_level=self.args['drop_level'])

        metadata.update(
            get_execution_metadata(
                module_file=__file__,
                t0=t0))

        with h5py.File(self.args['output_path'], 'a') as dst:
            dst.create_dataset(
                'metadata',
                data=json.dumps(metadata).encode('utf-8'))


def main():
    runner = PValueMarkersRunner()
    runner.run()


if __name__ == "__main__":
    main()
