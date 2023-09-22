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

    p_th = argschema.fields.Float(
        required=False,
        default=0.01,
        allow_none=False,
        description=("The corrected p-value that a gene's distribution "
                     "differs between two clusters must be less than this "
                     "for that gene to be considered a marker gene."))

    q1_th = argschema.fields.Float(
        required=False,
        default=0.5,
        allow_none=False,
        description=("Threshold on q1 (fraction of cells in at "
                     "least one cluster of a pair that express "
                     "a gene above 1 CPM) for a gene to be considered "
                     "a marker"))

    q1_min_th = argschema.fields.Float(
        required=False,
        default=0.1,
        allow_none=False,
        description=("If q1 less than this value, a gene "
                     "cannot be considered a marker, even if "
                     "exact_penetrance is False"))

    qdiff_th = argschema.fields.Float(
        required=False,
        default=0.7,
        allow_none=False,
        description=("Threshold on qdiff (differential penetrance) "
                     "above which a gene is considered a marker gene"))

    qdiff_min_th = argschema.fields.Float(
        required=False,
        default=0.1,
        allow_none=False,
        description=("If qdiff less than this value, a gene "
                     "cannot be considered a marker, even if "
                     "exact_penetrance is False"))

    log2_fold_th = argschema.fields.Float(
        required=False,
        default=1.0,
        allow_none=False,
        description=("The log2 fold change of a gene between two "
                     "clusters should be above this for that gene "
                     "to be considered a marker gene"))

    log2_fold_min_th = argschema.fields.Float(
        required=False,
        default=0.8,
        allow_none=False,
        description=("If the log2 fold change of a gene between two "
                     "clusters is less than this value, that gene cannot "
                     "be a marker, even if exact_penetrance is False"))

    n_valid = argschema.fields.Int(
        required=False,
        default=30,
        allow_none=False,
        description=("Try to find this many marker genes per pair. "
                     "Used only if exact_penetrance is False."))

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
            exact_penetrance=self.args['exact_penetrance'],
            p_th=self.args['p_th'],
            q1_th=self.args['q1_th'],
            q1_min_th=self.args['q1_min_th'],
            qdiff_th=self.args['qdiff_th'],
            qdiff_min_th=self.args['qdiff_min_th'],
            log2_fold_th=self.args['log2_fold_th'],
            log2_fold_min_th=self.args['log2_fold_min_th'],
            n_valid=self.args['n_valid'])

        with h5py.File(self.args['output_path'], 'a') as dst:
            dst.create_dataset(
                'metadata',
                data=metadata_str.encode('utf-8'))

        dur = time.time()-t0
        print(f"completed in {dur:.2e} seconds")


if __name__ == "__main__":
    runner = ReferenceMarkerRunner()
    runner.run()
