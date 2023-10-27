import argschema
import copy
import json
import time

from cell_type_mapper.utils.utils import (
    get_timestamp)

from cell_type_mapper.utils.anndata_utils import (
     read_df_from_h5ad)

from cell_type_mapper.type_assignment.marker_cache_v2 import (
    create_marker_gene_lookup_from_ref_list)

from cell_type_mapper.schemas.schemas import (
    QueryMarkerFinderSchema)


class QueryMarkerRunner(argschema.ArgSchemaParser):
    default_schema = QueryMarkerFinderSchema

    def run(self):

        t0 = time.time()

        var = read_df_from_h5ad(
            self.args['query_path'],
            df_name='var')
        query_gene_names = list(var.index.values)

        n_per_utility_override = None
        if self.args['n_per_utility_override'] is not None:
            n_per_utility_override = dict()
            for pair in self.args['n_per_utility_override']:
                if pair[0].lower() == 'none':
                    k = None
                else:
                    k = pair[0]
                n_per_utility_override[k] = pair[1]

        marker_lookup = create_marker_gene_lookup_from_ref_list(
            reference_marker_path_list=self.args['reference_marker_path_list'],
            query_gene_names=query_gene_names,
            n_per_utility=self.args['n_per_utility'],
            n_per_utility_override=n_per_utility_override,
            n_processors=self.args['n_processors'],
            behemoth_cutoff=5000000,
            tmp_dir=self.args['tmp_dir'],
            drop_level=self.args['drop_level'])

        marker_lookup['metadata'] = {'config': copy.deepcopy(self.args)}
        marker_lookup['metadata']['timestamp'] = get_timestamp()
        with open(self.args['output_path'], 'w') as dst:
            dst.write(
                json.dumps(marker_lookup, indent=2))

        dur = time.time()-t0
        print(f"RAN SUCCESSFULLY in {dur:.2e} seconds")


if __name__ == "__main__":
    runner = QueryMarkerRunner()
    runner.run()
