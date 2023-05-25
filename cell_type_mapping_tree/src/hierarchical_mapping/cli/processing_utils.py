import pathlib
import shutil
import tempfile
import time

from hierarchical_mapping.utils.utils import (
    _clean_up)

from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from hierarchical_mapping.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from hierarchical_mapping.cli.utils import (
    _copy_over_file)


def create_precomputed_stats_file(
        precomputed_config,
        precomputed_path,
        precomputed_tmp,
        log,
        tmp_dir):
    """
    Create the precomputed stats file (if necessary)

    Parameters
    ----------
    precomputed_config:
        Dict containing input config for precomputed stats
    precomputed_path:
        Path to the precomputed stats file (final storage
        space)
    precomputed_tmp:
        Path to temporary file for storing precomputed stats
        file locally (before copying to precomputed_path)
    log:
        CommandLogger to log messages while running
    tmp_dir:
        Global temp dir for CLI run
    """

    log.info("creating precomputed stats file")

    reference_path = pathlib.Path(
        precomputed_config['reference_path'])

    ref_tmp = pathlib.Path(
        tempfile.mkdtemp(
            prefix='reference_data_',
            dir=tmp_dir))

    (reference_path,
     _) = _copy_over_file(file_path=reference_path,
                          tmp_dir=ref_tmp,
                          log=log)

    if 'column_hierarchy' in precomputed_config:
        column_hierarchy = precomputed_config['column_hierarchy']
        taxonomy_tree = None
    else:
        taxonomy_tree = TaxonomyTree.from_json_file(
            json_path=precomputed_config['taxonomy_tree'])
        column_hierarchy = None

    t0 = time.time()
    precompute_summary_stats_from_h5ad(
        data_path=reference_path,
        column_hierarchy=column_hierarchy,
        taxonomy_tree=taxonomy_tree,
        output_path=precomputed_tmp,
        rows_at_a_time=10000,
        normalization=precomputed_config['normalization'])
    log.benchmark(msg="precomputing stats",
                  duration=time.time()-t0)

    if precomputed_path is not None:
        log.info("copying precomputed stats from "
                 f"{precomputed_tmp} to {precomputed_path}")
        shutil.copy(
            src=precomputed_tmp,
            dst=precomputed_path)

    _clean_up(ref_tmp)
