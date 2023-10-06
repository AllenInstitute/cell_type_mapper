import argschema
import copy
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

    precomputed_path_list = argschema.fields.List(
        argschema.fields.InputFile,
        required=True,
        default=None,
        allow_none=False,
        description=(
            "List of paths to precomputed stats files "
            "for which reference markers will be computed"))

    output_dir = argschema.fields.String(
        required=True,
        default=None,
        allow_none=False,
        description=(
            "Path to directory where refernce marker files "
            "will be written. Specific file names will be inferred "
            "from precomputed stats files."))

    clobber = argschema.fields.Boolean(
        required=False,
        default=False,
        allow_none=False,
        description=("If False, do not allow overwrite of existing "
                     "output files."))

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

        output_dir = pathlib.Path(data['output_dir'])
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        if not output_dir.is_dir():
            raise RuntimeError(
                f"output_dir: {output_dir} is not a dir")

        input_to_output = dict()
        files_to_write = set()
        salt = None
        for input_path in data['precomputed_path_list']:
            input_path = pathlib.Path(input_path)
            input_name = input_path.name
            name_params = input_name.split('.')
            old_stem = name_params[0]
            new_path = None
            while True:
                if new_path is not None:
                    if salt is None:
                        salt = 0
                    else:
                        salt += 1
                new_stem = 'reference_markers'
                if salt is not None:
                    new_stem = f'{new_stem}.{salt}'
                new_name = input_name.replace(old_stem, new_stem, 1)
                new_path = str(output_dir/new_name)
                if new_path not in files_to_write:
                    files_to_write.add(new_path)
                    break
            input_to_output[str(input_path)] = new_path

        # check that none of the output files exist (or, if they do, that
        # clobber is True)
        error_msg = ""
        for pth in input_to_output.values():
            pth = pathlib.Path(pth)
            if pth.exists():
                if not pth.is_file():
                    error_msg += f"{pth} exists and is not a file\n"
                elif not self.data['clobber']:
                    error_msg += (
                        f"{pth} already exists; to overwrite, run with "
                        "clobber=True\n")

        if len(error_msg) == 0:
            # make sure we can write to these files
            for pth in input_to_output.values():
                pth = pathlib.Path(pth)
                try:
                    with open(pth, 'wb') as dst:
                        dst.write(b'junk')
                    pth.unlink()
                except FileNotFoundError:
                    error_msg += (
                        f"cannot write to {pth}\n"
                    )

        if len(error_msg) > 0:
            error_msg += (
                 "These file names are automatically generated. "
                 "The quickest solution is to specify a new output_dir.")
            raise RuntimeError(error_msg)

        data['input_to_output_map'] = input_to_output

        return data


class ReferenceMarkerRunner(argschema.ArgSchemaParser):

    default_schema = ReferenceMarkerSchema

    def run(self):

        parent_metadata = {
            'config': self.args,
            'timestamp': get_timestamp()
        }

        taxonomy_tree = None

        t0 = time.time()

        for precomputed_path in self.args['input_to_output_map']:
            output_path = self.args['input_to_output_map'][precomputed_path]
            print(f'writing {output_path}')
            taxonomy_tree = TaxonomyTree.from_precomputed_stats(
                stats_path=precomputed_path)

            if self.args['drop_level'] is not None:
                taxonomy_tree = taxonomy_tree.drop_level(
                    self.args['drop_level'])

            find_markers_for_all_taxonomy_pairs(
                precomputed_stats_path=precomputed_path,
                taxonomy_tree=taxonomy_tree,
                output_path=output_path,
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

            metadata = copy.deepcopy(parent_metadata)
            metadata['precomputed_path'] = precomputed_path

            metadata_str = json.dumps(metadata)
            with h5py.File(output_path, 'a') as dst:
                dst.create_dataset(
                    'metadata',
                    data=metadata_str.encode('utf-8'))

        dur = time.time()-t0
        print(f"completed in {dur:.2e} seconds")


if __name__ == "__main__":
    runner = ReferenceMarkerRunner()
    runner.run()
