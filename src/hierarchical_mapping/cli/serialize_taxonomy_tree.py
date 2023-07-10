import argschema

from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)


class TaxonomySerializationSchema(argschema.ArgSchema):
    """
    Create a taxonomy tree by bootstrapping it from columns
    in the obs dataframe of a single h5ad file. Write that
    tree to a JSON file.
    """

    h5ad_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the h5ad file from which to "
        "read the taxonomy tree.")

    column_hierarchy = argschema.fields.List(
        argschema.fields.String,
        required=True,
        default=None,
        allow_none=False,
        cli_as_single_argument=True,
        description="List of columns from the 'obs' dataframe "
        "that will be used to construct the taxonomy tree. List "
        "must be ordered from the highest level of the taxonomy "
        "down to the leaf node.")

    output_path = argschema.fields.OutputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the JSON file that will be "
        "written with the serialized taxonomy tree.")


class TaxonomySerializationRunner(argschema.ArgSchemaParser):

    default_schema = TaxonomySerializationSchema

    def run(self):
        taxonomy_tree = TaxonomyTree.from_h5ad(
            h5ad_path=self.args['h5ad_path'],
            column_hierarchy=self.args['column_hierarchy'])
        with open(self.args['output_path'], 'w') as dst:
            dst.write(taxonomy_tree.to_str())


def main():
    runner = TaxonomySerializationRunner()
    runner.run()


if __name__ == "__main__":
    main()
