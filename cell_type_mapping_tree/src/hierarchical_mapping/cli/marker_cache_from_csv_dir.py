"""
This will provide the CLI module to go from a directory full
of CSVs listing marker genes to the .JSON file containing the
reference marker lookup expected by the from_specified_markers CLI
"""

import argschema
import copy
import h5py
import json

from hierarchical_mapping.utils.utils import (
    get_timestamp)

from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from hierarchical_mapping.marker_lookup.marker_lookup import (
    marker_lookup_from_tree_and_csv)

from hierarchical_mapping.gene_id.gene_id_mapper import (
    GeneIdMapper)


class MarkerCacheInputSchema(argschema.ArgSchema):

    marker_dir = argschema.fields.InputDir(
        required=True,
        default=None,
        allow_none=False,
        description="Directory containing the lists of marker genes "
        "produced by the science team's R code")

    precomputed_file_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the HDF5 file with the precomputed stats "
        "for this cell type taxonomy. We will read the taxonomy tree "
        "from that file")

    output_path = argschema.fields.OutputFile(
        required=True,
        default=None,
        allow_none=False,
        dsecription="Path to the JSON file that will be written.")

    drop_level = argschema.fields.String(
        required=False,
        default="CCN20230504_SUPT",
        allow_none=True,
        description="If this level exists in the taxonomy, drop "
        "it before doing type assignment (this is to accommmodate "
        "the fact that the official taxonomy includes the "
        "'supertype', even though that level is not used "
        "during hierarchical type assignment")


class MarkerCacheRunner(argschema.ArgSchemaParser):

    default_schema = MarkerCacheInputSchema

    def run(self):

        with h5py.File(self.args['precomputed_file_path'], 'r') as src:
            taxonomy_tree = TaxonomyTree(
                data=json.loads(src['taxonomy_tree'][()].decode('utf-8')))

        if 'drop_level' in self.args:
            if self.args['drop_level'] is not None:
                if self.args['drop_level'] in taxonomy_tree.hierarchy:
                    taxonomy_tree = taxonomy_tree.drop_level(
                        self.args['drop_level'])

        raw_markers = marker_lookup_from_tree_and_csv(
            csv_dir=self.args['marker_dir'],
            taxonomy_tree=taxonomy_tree)

        gene_id_mapper = GeneIdMapper.from_default()
        result = dict()
        for k in raw_markers:
            new_markers = gene_id_mapper.map_gene_identifiers(
                gene_id_list=raw_markers[k],
                strict=True)

            result[k] = new_markers

        result['metadata'] = copy.deepcopy(self.args)
        result['metadata']['generated_on'] = get_timestamp()

        with open(self.args['output_path'], 'w') as dst:
            dst.write(json.dumps(result, indent=2))


if __name__ == "__main__":
    runner = MarkerCacheRunner()
    runner.run()
