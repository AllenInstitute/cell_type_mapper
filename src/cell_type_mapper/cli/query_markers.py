import argschema
import copy
import h5py
import json
import time

from cell_type_mapper.utils.output_utils import (
    get_execution_metadata)

from cell_type_mapper.utils.anndata_utils import (
     read_df_from_h5ad)

from cell_type_mapper.type_assignment.marker_cache_v2 import (
    create_marker_gene_lookup_from_ref_list)

from cell_type_mapper.schemas.query_marker_finder import (
    QueryMarkerFinderSchema)


class QueryMarkerRunner(argschema.ArgSchemaParser):
    default_schema = QueryMarkerFinderSchema

    def run(self):

        t0 = time.time()

        if self.args['query_path'] is not None:
            var = read_df_from_h5ad(
                self.args['query_path'],
                df_name='var')
            query_gene_names = list(var.index.values)

        else:
            # find all of the genes that exist in every reference marker
            # file
            query_gene_names = None
            for ref_path in self.args['reference_marker_path_list']:
                with h5py.File(ref_path, 'r') as src:
                    these = json.loads(src['gene_names'][()].decode('utf-8'))
                these = set(these)
                if query_gene_names is None:
                    query_gene_names = these
                else:
                    query_gene_names = query_gene_names.intersection(these)

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
            drop_level=self.args['drop_level'],
            genes_at_a_time=self.args['genes_at_a_time'])

        metadata = {'config': copy.deepcopy(self.args)}
        metadata.update(
            get_execution_metadata(
                module_file=__file__,
                t0=t0))

        marker_lookup['metadata'] = metadata

        with open(self.args['output_path'], 'w') as dst:
            dst.write(
                json.dumps(marker_lookup, indent=2))

        dur = time.time()-t0
        print(f"QUERY MARKER FINDER RAN SUCCESSFULLY in {dur:.2e} seconds")


if __name__ == "__main__":
    runner = QueryMarkerRunner()
    runner.run()
