import argschema

import h5py
import json
import pathlib
import tempfile
import time

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.utils.config_utils import (
    patch_child_to_parent
)

from cell_type_mapper.utils.cli_utils import (
    config_from_args
)

from cell_type_mapper.utils.output_utils import (
    get_execution_metadata)

from cell_type_mapper.schemas.p_value_reference_and_query_markers import (
    QueryMarkersFromPValueMaskSchema)

from cell_type_mapper.cli.reference_markers_from_p_value_mask import (
    PValueMarkersRunner)

from cell_type_mapper.cli.query_markers import (
    QueryMarkerRunner)


class QueryMarkersFromPValueMaskRunner(
        argschema.ArgSchemaParser):

    default_schema = QueryMarkersFromPValueMaskSchema

    def run(self):
        t0 = time.time()

        metadata = {
            'config': config_from_args(
                        input_config=self.args,
                        cloud_safe=False)
        }

        tmp_dir = tempfile.mkdtemp(
            dir=self.args['tmp_dir'],
            prefix='markers_from_p_values_')
        try:
            timing_by_stages = self._run(tmp_dir=tmp_dir)
        finally:
            _clean_up(tmp_dir)

        # update the metadata in the output file to reflect
        # which CLI tool was actually run
        output_path = pathlib.Path(self.args['output_path'])
        if output_path.exists():
            metadata['duration_by_stages'] = timing_by_stages
            metadata.update(
                get_execution_metadata(
                    module_file=__file__,
                    t0=t0))

            with open(output_path, 'rb') as src:
                result = json.load(src)
            result.pop('metadata')
            result['metadata'] = metadata
            with open(output_path, 'w') as dst:
                dst.write(json.dumps(result, indent=2))

    def _run(self, tmp_dir):

        timing_by_stages = dict()

        with h5py.File(self.args['p_value_mask_path'], 'r') as src:
            p_value_metadata = json.loads(
                src['metadata'][()].decode('utf-8'))

        n_valid = None
        if 'reference_markers' in self.args:
            if self.args['reference_markers']['n_valid'] is not None:
                n_valid = self.args['reference_markers']['n_valid']

        if n_valid is None:
            n_valid = self.args['query_markers']['n_per_utility']*2
            if self.args['query_markers'][
                            'n_per_utility_override'] is not None:
                for element in self.args['query_markers'][
                                'n_per_utility_override']:
                    if 2*element[1] > n_valid:
                        n_valid = 2*element[1]

        precomputed_stats_path = p_value_metadata[
                        'config']['precomputed_stats_path']

        (lookup,
         missing_pairs) = patch_child_to_parent(
             child_to_parent={
                 precomputed_stats_path: self.args['p_value_mask_path']
             },
             do_search=self.args['search_for_stats_file'])

        if len(missing_pairs) > 0:
            parent = missing_pairs[0][0]
            child = missing_pairs[0][1]
            msg = (
                f"Could not find\n{child}\nwhich is referenced in\n"
                f"{parent}\nTry running with search_for_stats_file=True "
                "and saving the missing file in the same directory as "
                f"\n{parent}"
            )
            raise FileNotFoundError(msg)

        precomputed_stats_path = str(
            list(lookup.keys())[0].resolve().absolute()
        )

        reference_marker_path = mkstemp_clean(
            dir=tmp_dir,
            prefix='reference_markers_',
            suffix='.h5')

        ref_config = {
            'output_path': reference_marker_path,
            'clobber': True,
            'p_value_mask_path': self.args['p_value_mask_path'],
            'precomputed_stats_path': precomputed_stats_path,
            'tmp_dir': tmp_dir,
            'query_path': self.args['query_path'],
            'query_gene_id_col': self.args['query_gene_id_col'],
            'drop_level': self.args['drop_level'],
            'n_processors': self.args['n_processors'],
            'max_gb': self.args['max_gb'],
            'n_valid': n_valid
        }

        t0 = time.time()
        ref_runner = PValueMarkersRunner(
            args=[],
            input_data=ref_config)
        ref_runner.run()
        timing_by_stages['reference_markers'] = time.time()-t0

        query_config = {
            'output_path': self.args['output_path'],
            'reference_marker_path_list': [reference_marker_path],
            'query_path': self.args['query_path'],
            'query_gene_id_col': self.args['query_gene_id_col'],
            'n_processors': self.args['n_processors'],
            'drop_level': self.args['drop_level'],
            'tmp_dir': tmp_dir,
            'n_per_utility': self.args[
                'query_markers']['n_per_utility'],
            'n_per_utility_override': self.args[
                'query_markers']['n_per_utility_override'],
            'genes_at_a_time': self.args[
                'query_markers']['genes_at_a_time'],
        }

        t0 = time.time()
        query_runner = QueryMarkerRunner(
            args=[],
            input_data=query_config)
        query_runner.run()
        timing_by_stages['query_markers'] = time.time()-t0

        return timing_by_stages


def main():

    runner = QueryMarkersFromPValueMaskRunner()
    runner.run()


if __name__ == "__main__":
    main()
