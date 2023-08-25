import argschema
import h5py
import json
from marshmallow import post_load
import pathlib

from cell_type_mapper.utils.output_utils import (
    blob_to_df)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad,
    write_df_to_h5ad)

from cell_type_mapper.utils.h5_utils import (
    copy_h5_excluding_data)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)


class TranscriptionSchema(argschema.ArgSchema):

    result_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the JSON file containing the "
        "extended mapping resluts")

    h5ad_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description="Path to the h5ad file the mapping was "
        "based on. This file will be copied and the result "
        "of the cell type mapping will be added to its obs "
        "dataframe.")

    new_h5ad_path = argschema.fields.OutputFile(
        required=True,
        default=None,
        allow_none=False,
        description="File that will be written.")

    clobber = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description="If False, the code will not allow you "
        "to overwrite an existing file")

    @post_load
    def check_output_existence(self, data, **kwargs):
        if not data['clobber']:
            output_path = pathlib.Path(data['new_h5ad_path'])
            if output_path.is_file():
                raise RuntimeError(
                    f"{output_path} already exists; "
                    "run with clobber=True to ovewrite.")
        return data


class TranscribeToObsRunner(argschema.ArgSchemaParser):

    default_schema = TranscriptionSchema

    def run(self):

        with open(self.args['result_path'], 'rb') as src:
            mapping = json.load(src)

        if 'taxonomy_tree' in mapping:
            taxonomy_tree = TaxonomyTree(
                data=mapping['taxonomy_tree'])
        else:
            precomputed = mapping['config']['precomputed_stats']['path']
            with h5py.File(precomputed, 'r') as src:
                taxonomy_tree = TaxonomyTree(
                    data=json.loads(src['taxonomy_tree'][()].decode('utf-8')))

        mapping = blob_to_df(
            results_blob=mapping['results'],
            taxonomy_tree=taxonomy_tree).set_index('cell_id')

        obs = read_df_from_h5ad(
            h5ad_path=self.args['h5ad_path'],
            df_name='obs')

        msg = ''
        for column in mapping.columns:
            if column in obs.columns:
                msg += f"{column}\n"
        if len(msg) > 0:
            raise RuntimeError(
                "Cannot transcribe results to "
                f"{self.args['h5ad_path']}\n"
                "obs already contains the following columns\n"
                f"{msg}")

        copy_h5_excluding_data(
            src_path=self.args['h5ad_path'],
            dst_path=self.args['new_h5ad_path'],
            excluded_groups=['obs'],
            excluded_datasets=['obs'],
            max_elements=10*1024**3//8)

        obs = obs.join(mapping)
        write_df_to_h5ad(
            h5ad_path=self.args['new_h5ad_path'],
            df_name='obs',
            df_value=obs)


if __name__ == "__main__":
    runner = TranscribeToObsRunner()
    runner.run()
