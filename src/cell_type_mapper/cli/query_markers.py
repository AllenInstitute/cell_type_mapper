import argschema
import copy
import h5py
import json

from cell_type_mapper.utils.utils import (
    get_timestamp)

from cell_type_mapper.utils.anndata_utils import (
     read_df_from_h5ad)

from cell_type_mapper.diff_exp.precompute_utils import (
    run_leaf_census)

from cell_type_mapper.type_assignment.marker_cache_v2 import (
    create_raw_marker_gene_lookup)


class QueryMarkerSchema(argschema.ArgSchema):

    query_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the h5ad file containing the query "
        "dataset (used to read the list of available genes).")

    reference_marker_path_list = argschema.fields.List(
        argschema.fields.InputFile,
        required=True,
        default=None,
        allow_none=False,
        cli_as_single_argument=True,
        description=(
            "List of reference marker files to use "
            "when creating this query marker file.")
        )

    n_processors = argschema.fields.Int(
        required=False,
        default=32,
        allow_none=False,
        description="Number of independendent processes to use when "
        "parallelizing work for mapping job")

    n_per_utility = argschema.fields.Int(
        required=False,
        default=30,
        allow_none=False,
        description="Number of marker genes to find per "
        "(taxonomy_node_A, taxonomy_node_B, up/down) combination.")

    n_per_utility_override = argschema.fields.List(
        argschema.fields.Tuple(
            (argschema.fields.String,
             argschema.fields.Integer)
        ),
        required=False,
        default=None,
        allow_none=True,
        cli_as_single_argument=True,
        description=(
            "Optional override for n_per_utilty at specific "
            "parent nodes in the taxonomy tree. Encoded as a "
            "list of ('parent_node', n_per_utility) tuples."
        ))

    drop_level = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description="If this level exists in the taxonomy, drop "
        "it before doing type assignment (this is to accommmodate "
        "the fact that the official taxonomy includes the "
        "'supertype', even though that level is not used "
        "during hierarchical type assignment")

    output_path = argschema.fields.OutputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the JSON file that will contain "
        "the marker gene lookup for the query dataset.")

    tmp_dir = argschema.fields.OutputDir(
        required=False,
        default=None,
        allow_none=True,
        description="Optional temporary directory for scratch files.")


class QueryMarkerRunner(argschema.ArgSchemaParser):
    default_schema = QueryMarkerSchema

    def run(self):

        error_msg = ""
        ref_to_precompute = dict()
        for ref_path in self.args['reference_marker_path_list']:
            with h5py.File(ref_path, 'r') as src:
                metadata = json.loads(src['metadata'][()].decode('utf-8'))
            if 'config' not in metadata:
                error_msg += (
                    "===\n"
                    f"{ref_path} has no 'config' in metadata\n===\n"
                )
                continue
            config = metadata['config']
            if 'precomputed_path' not in config:
                error_msg += (
                    "===\n"
                    f"{ref_path} does not point to a "
                    "precomputed stats file\n===\n")
                continue
            stats_path = config['precomputed_path']
            ref_to_precompute[ref_path] = stats_path

        if len(error_msg) > 0:
            raise RuntimeError(error_msg)

        (leaf_census,
         taxonomy_tree) = run_leaf_census(
             list(ref_to_precompute.values()))

        var = read_df_from_h5ad(
            self.args['query_path'],
            df_name='var')
        query_gene_names = list(var.index.values)

        if self.args['drop_level'] is not None:
            taxonomy_tree = taxonomy_tree.drop_level(self.args['drop_level'])

        n_per_utility_override = None
        if self.args['n_per_utility_override'] is not None:
            n_per_utility_override = dict()
            for pair in self.args['n_per_utility_override']:
                if pair[0].lower() == 'none':
                    k = None
                else:
                    k = pair[0]
                n_per_utility_override[k] = pair[1]

        reference_marker_path = self.args['reference_marker_path_list'][0]

        marker_lookup = create_raw_marker_gene_lookup(
            input_cache_path=reference_marker_path,
            query_gene_names=query_gene_names,
            taxonomy_tree=taxonomy_tree,
            n_per_utility=self.args['n_per_utility'],
            n_processors=self.args['n_processors'],
            behemoth_cutoff=5000000,
            tmp_dir=self.args['tmp_dir'])

        marker_lookup['metadata'] = {'config': copy.deepcopy(self.args)}
        marker_lookup['metadata']['timestamp'] = get_timestamp()
        with open(self.args['output_path'], 'w') as dst:
            dst.write(
                json.dumps(marker_lookup, indent=2))


if __name__ == "__main__":
    runner = QueryMarkerRunner()
    runner.run()
