import argschema
import copy
import h5py
import json

from cell_type_mapper.utils.utils import (
    get_timestamp)

from cell_type_mapper.utils.anndata_utils import (
     read_df_from_h5ad)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.type_assignment.marker_cache_v2 import (
    create_raw_marker_gene_lookup)


class QueryMarkerSchema(argschema.ArgSchema):

    query_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the h5ad file containing the query "
        "dataset (used to read the list of available genes).")

    precomputed_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the precomputed stats file "
        "(used to read in the taxonomy tree).")

    reference_marker_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the HDF5 file containing the "
        "marker genes for the (reference dataset, taxonomy) "
        "pair.")

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

        var = read_df_from_h5ad(
            self.args['query_path'],
            df_name='var')
        query_gene_names = list(var.index.values)

        with h5py.File(self.args['precomputed_path'], 'r') as src:
            taxonomy_tree = TaxonomyTree(
                data=json.loads(src['taxonomy_tree'][()].decode('utf-8')))
        if self.args['drop_level'] is not None:
            taxonomy_tree = taxonomy_tree.drop_level(self.args['drop_level'])

        marker_lookup = create_raw_marker_gene_lookup(
            input_cache_path=self.args['reference_marker_path'],
            query_gene_names=query_gene_names,
            taxonomy_tree=taxonomy_tree,
            n_per_utility=self.args['n_per_utility'],
            n_processors=self.args['n_processors'],
            behemoth_cutoff=5000000,
            tmp_dir=self.args['tmp_dir'])

        marker_lookup['metadata'] = copy.deepcopy(self.args)
        marker_lookup['metadata']['timestamp'] = get_timestamp()
        with open(self.args['output_path'], 'w') as dst:
            dst.write(
                json.dumps(marker_lookup, indent=2))


if __name__ == "__main__":
    runner = QueryMarkerRunner()
    runner.run()
