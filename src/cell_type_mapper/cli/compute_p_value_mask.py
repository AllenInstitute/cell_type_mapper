import argschema
import copy
import h5py
import json
import time

import cell_type_mapper

from cell_type_mapper.diff_exp.p_value_mask import (
    create_p_value_mask_file)

from cell_type_mapper.schemas.p_value_mask import (
    PValueMaskSchema)


class PValueRunner(argschema.ArgSchemaParser):

    default_schema = PValueMaskSchema

    def run(self):
        t0 = time.time()
        metadata = {'config': copy.deepcopy(self.args)}

        create_p_value_mask_file(
            precomputed_stats_path=self.args['precomputed_stats_path'],
            dst_path=self.args['output_path'],
            p_th=self.args['p_th'],
            q1_th=self.args['q1_th'],
            q1_min_th=self.args['q1_min_th'],
            qdiff_th=self.args['qdiff_th'],
            qdiff_min_th=self.args['qdiff_min_th'],
            log2_fold_th=self.args['log2_fold_th'],
            log2_fold_min_th=self.args['log2_fold_min_th'],
            n_processors=self.args['n_processors'],
            tmp_dir=self.args['tmp_dir'],
            n_per=self.args['rows_at_a_time'])

        duration = time.time()-t0
        metadata['duration'] = duration
        metadata['version'] = cell_type_mapper.__version__
        with h5py.File(self.args['output_path'], 'a') as dst:
            dst.create_dataset(
                'metadata',
                data=json.dumps(metadata).encode('utf-8'))


def main():
    runner = PValueRunner()
    runner.run()


if __name__ == "__main__":
    main()
