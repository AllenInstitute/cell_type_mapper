"""
This is the module used to generate the precomputed_stats.h5 file
from a scrattch-compliant h5ad file.
"""

import argschema

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from cell_type_mapper.schemas.precomputation_schema import (
    PrecomputedStatsScrattchSchema)


class PrecomputationScrattchRunner(argschema.ArgSchemaParser):

    default_schema = PrecomputedStatsScrattchSchema

    def run(self):
        precompute_summary_stats_from_h5ad(
            data_path=self.args['h5ad_path'],
            column_hierarchy=self.args['hierarchy'],
            taxonomy_tree=None,
            output_path=self.args['output_path'],
            rows_at_a_time=10000,
            normalization=self.args['normalization'],
            tmp_dir=self.args['tmp_dir'],
            n_processors=self.args['n_processors'])


def main():
    runner = PrecomputationScrattchRunner()
    runner.run()


if __name__ == "__main__":
    main()
