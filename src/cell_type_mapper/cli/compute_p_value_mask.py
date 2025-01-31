import argschema
import h5py
import json
import time

from cell_type_mapper.utils.cli_utils import (
    config_from_args
)

from cell_type_mapper.utils.output_utils import (
    get_execution_metadata)

from cell_type_mapper.diff_exp.p_value_mask import (
    create_p_value_mask_file)

from cell_type_mapper.schemas.p_value_mask import (
    PValueMaskSchema)


class PValueRunner(argschema.ArgSchemaParser):

    default_schema = PValueMaskSchema

    def run(self):
        t0 = time.time()
        metadata = {'config': config_from_args(
                                input_config=self.args,
                                cloud_safe=False)
                    }

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

        metadata.update(
            get_execution_metadata(
                module_file=__file__,
                t0=t0))

        with h5py.File(self.args['output_path'], 'a') as dst:
            dst.create_dataset(
                'metadata',
                data=json.dumps(metadata).encode('utf-8'))

        duration = time.time()-t0
        print(
            "======P-VALUE MASK RAN SUCCESSFULLY in "
            f"{duration/3600.0:.2e} hrs======")


def main():
    runner = PValueRunner()
    runner.run()


if __name__ == "__main__":
    main()
