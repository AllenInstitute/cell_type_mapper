"""
This module will define a CLI tool for taking a precomputed_stats
file and producing a new precomputed_stats file created by
dropping levels from the previous file's taxonomy.
"""

import argschema

from cell_type_mapper.diff_exp.truncate_precompute import (
    truncate_precomputed_stats_file)


class TruncationSchema(argschema.ArgSchema):

    input_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description=(
            "Path to the precomputed_stats file being "
            "truncated."
        ))

    output_path = argschema.fields.OutputFile(
        required=True,
        default=None,
        allow_none=False,
        description=(
            "Path to the precomputed_stats file to be written."
        ))

    new_hierarchy = argschema.fields.List(
        argschema.fields.String,
        required=True,
        default=None,
        allow_none=False,
        cli_as_single_argument=True,
        description=(
            "List of taxonomic levels to be retained in the new "
            "precomputed_stats file, ordered from most gross to "
            "most fine."
        ))


class TaxonomyTruncationRunner(argschema.ArgSchemaParser):

    default_schema = TruncationSchema

    def run(self):
        truncate_precomputed_stats_file(
            input_path=self.args['input_path'],
            output_path=self.args['output_path'],
            new_hierarchy=self.args['new_hierarchy'])


def main():
    runner = TaxonomyTruncationRunner()
    runner.run()


if __name__ == "__main__":
    main()
