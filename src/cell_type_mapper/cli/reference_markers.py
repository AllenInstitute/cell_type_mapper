import argschema
import h5py
import json
from marshmallow import post_load
import pathlib
import time

from cell_type_mapper.utils.utils import get_timestamp

from cell_type_mapper.diff_exp.markers import (
    find_markers_for_all_taxonomy_pairs)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)


class ReferenceMarkerSchema(argschema.ArgSchema):

    precomputed_path = argschema.fields.InputFile(
        required=True,
        default=None,
        allow_none=False,
        description=("Precomputed stats file to be used "
                     "to find markers"))

    output_path = argschema.fields.String(
        required=True,
        default=None,
        allow_none=False,
        description=("Path to HDF5 file with reference markers to "
                     "be written"))

    clobber = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description=("If False, do not allow overwrite of existing "
                     "output path"))

    drop_level = argschema.fields.String(
        required=False,
        default='CCN20230722_SUPT',
        allow_none=True,
        description=("Optional level to drop from taxonomy"))

    tmp_dir = argschema.fields.OutputDir(
        required=False,
        default=None,
        allow_none=True,
        description=("Temporary directory for writing out "
                     "scratch files"))

    n_processors = argschema.fields.Int(
        required=False,
        default=32,
        allow_none=False,
        description=("Number of independent processors to spin up."))

    exact_penetrance = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description=("If False, allow genes that technically fail "
                     "penetrance and fold-change thresholds to pass "
                     "through as reference genes."))

    @post_load
    def check_clobber(self, data, **kwargs):
        output_path = pathlib.Path(data['output_path'])
        if output_path.exists() and not data['clobber']:
            raise RuntimeError(
                f"{output_path} already exists; run with 'clobber' = True "
                "to overwrite")
        elif not output_path.exists():
            # check that we can write to the file.
            with open(output_path, "w") as dst:
                dst.write("junk")
            output_path.unlink()
        return data


class ReferenceMarkerRunner(argschema.ArgSchemaParser):

    default_schema = ReferenceMarkerSchema

    def run(self):

        metadata = {
            'config': self.args,
            'timestamp': get_timestamp()
        }
        metadata_str = json.dumps(metadata)

        precomputed_path = pathlib.Path(
            self.args['precomputed_path'])

        t0 = time.time()

        taxonomy_tree = TaxonomyTree.from_precomputed_stats(
            stats_path=precomputed_path)

        if self.args['drop_level'] is not None:
            taxonomy_tree = taxonomy_tree.drop_level(self.args['drop_level'])

        find_markers_for_all_taxonomy_pairs(
            precomputed_stats_path=precomputed_path,
            taxonomy_tree=taxonomy_tree,
            output_path=self.args['output_path'],
            tmp_dir=self.args['tmp_dir'],
            n_processors=self.args['n_processors'],
            exact_penetrance=self.args['exact_penetrance'])

        with h5py.File(self.args['output_path'], 'a') as dst:
            dst.create_dataset(
                'metadata',
                data=metadata_str.encode('utf-8'))

        dur = time.time()-t0
        print(f"completed in {dur:.2e} seconds")


if __name__ == "__main__":
    runner = ReferenceMarkerRunner()
    runner.run()
