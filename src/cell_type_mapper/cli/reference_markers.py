import argschema
import copy
import h5py
import json
import pathlib
import time

import cell_type_mapper.utils.gene_utils as gene_utils

from cell_type_mapper.utils.cli_utils import (
    config_from_args
)

from cell_type_mapper.utils.output_utils import (
    get_execution_metadata)

from cell_type_mapper.diff_exp.markers import (
    find_markers_for_all_taxonomy_pairs)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.cli.cli_log import CommandLog

from cell_type_mapper.schemas.reference_marker_finder import (
    ReferenceMarkerFinderSchema)


class ReferenceMarkerRunner(argschema.ArgSchemaParser):

    default_schema = ReferenceMarkerFinderSchema

    def run(self):

        parent_metadata = {
            'config': config_from_args(
                        input_config=self.args,
                        cloud_safe=False)
        }

        log = CommandLog()

        input_to_output = self.create_input_to_output_map()

        parent_metadata['input_to_output_map'] = input_to_output

        parent_metadata.update(
            get_execution_metadata(
                module_file=__file__,
                t0=None))

        taxonomy_tree = None

        t0 = time.time()

        if self.args['query_path'] is not None:
            gene_list = gene_utils.get_gene_identifier_list(
                h5ad_path_list=[self.args['query_path']],
                gene_id_col=self.args['query_gene_id_col'],
                duplicate_prefix=gene_utils.invalid_precompute_prefix()
            )

            # remove any genes marked as `INVALID_MARKER`; these will
            # have been duplicate genes in the reference data
            gene_list = [
                _gene for _gene in gene_list
                if not _gene.startswith(gene_utils.invalid_precompute_prefix())
            ]

        else:
            gene_list = None

        for precomputed_path in input_to_output:
            local_t0 = time.time()
            output_path = input_to_output[precomputed_path]
            to_write = output_path
            if self.args['cloud_safe']:
                to_write = '../' + pathlib.Path(to_write).name
            print(f'writing {to_write}')
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
                n_valid=self.args['n_valid'],
                gene_list=gene_list,
                max_gb=self.args['max_gb'],
                log=log)

            log.info("REFERENCE MARKER FINDER RAN SUCCESSFULLY")

            metadata = copy.deepcopy(parent_metadata)
            metadata['precomputed_path'] = precomputed_path
            metadata['log'] = log.log
            duration = time.time()-local_t0
            metadata['duration'] = duration
            metadata_str = json.dumps(metadata)
            with h5py.File(output_path, 'a') as dst:
                dst.create_dataset(
                    'metadata',
                    data=metadata_str.encode('utf-8'))

        dur = time.time()-t0
        print(f"completed in {dur:.2e} seconds")

    def create_input_to_output_map(self):
        """
        Return dict mapping input paths to output paths
        """

        output_dir = pathlib.Path(self.args['output_dir'])

        input_to_output = dict()
        files_to_write = set()

        # salting of output file names is done in the case where
        # multiple precomputed files would result in the same
        # refrence marker path name
        salt = None
        for input_path in self.args['precomputed_path_list']:
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
                elif not self.args['clobber']:
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

        return input_to_output


if __name__ == "__main__":
    runner = ReferenceMarkerRunner()
    runner.run()
