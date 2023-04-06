import time
import json
import numpy as np
import os
import pathlib
import tempfile
import shutil

from hierarchical_mapping.utils.taxonomy_utils import (
    convert_tree_to_leaves,
    get_siblings)

from hierarchical_mapping.type_assignment.election import (
    run_type_assignment_on_h5ad)

import os

def run_test(
        bootstrap_factor=0.9,
        bootstrap_iteration=100,
        output_path=None,
        flatten=False):

    full_result = dict()

    if output_path is None:
        raise RuntimeError("must specify output_path")

    data_dir = pathlib.Path(
        '/allen/aibs/technology/danielsf/knowledge_base/validation')
    assert data_dir.is_dir()

    output_path = data_dir / output_path

    tree = json.load(open(data_dir / 'taxonomy_tree.json', 'rb'))
    if flatten:
        raw = tree
        tree = dict()
        leaf = raw['hierarchy'][-1]
        tree['hierarchy'] = [leaf]
        tree[leaf] = raw[leaf]

    full_result['taxonomy_tree'] = tree

    query_data_path = pathlib.Path(
        '/allen/programs/celltypes/workgroups/rnaseqanalysis/changkyul/CIRRO/MFISH/atlas_brain_638850.remap.4334174.updated.imputed.h5ad')
    assert query_data_path.is_file()

    full_result['query_path'] = str(query_data_path.resolve().absolute())

    tmp_dir = os.environ['TMPDIR']
    query_tmp_path = tempfile.mkstemp(dir=tmp_dir, suffix='.h5ad')[1]
    print(f"copying query_data to {query_tmp_path}")
    shutil.copy(src=query_data_path, dst=query_tmp_path)
    query_tmp_path = pathlib.Path(query_tmp_path)
    assert query_tmp_path.is_file()
    print('done copying')

    precomputed_path = data_dir / 'validation_test_precompute.h5'
    assert os.path.isfile(precomputed_path)
    full_result['precomputed_path'] = str(precomputed_path.resolve().absolute())

    precomputed_tmp_path = tempfile.mkstemp(dir=tmp_dir, suffix='.h5')[1]
    shutil.copy(src=precomputed_path, dst=precomputed_tmp_path)

    marker_cache_path = data_dir / 'validation_marker_cache.h5'
    assert marker_cache_path.is_file()
    full_result['marker_path'] = str(marker_cache_path.resolve().absolute())
    marker_tmp_path = tempfile.mkstemp(dir=tmp_dir, suffix='.h5')[1]
    shutil.copy(src=marker_cache_path, dst=marker_tmp_path)

    print('copied all data')

    t0 = time.time()
    result = run_type_assignment_on_h5ad(
        query_h5ad_path=query_tmp_path,
        precomputed_stats_path=precomputed_tmp_path,
        marker_gene_cache_path=marker_tmp_path,
        taxonomy_tree=tree,
        n_processors=32,
        chunk_size=30000,
        bootstrap_factor=bootstrap_factor,
        bootstrap_iteration=bootstrap_iteration,
        rng=np.random.default_rng(11235))

    full_result['bootstrap_factor'] = bootstrap_factor
    full_result['bootstrap_iteration'] = int(bootstrap_iteration)
    full_result['result'] = result

    duration = (time.time()-t0)/3600.0
    print(f"marker cache creation took {duration:.2e} hours")

    with open(output_path, 'w') as out_file:
        out_file.write(json.dumps(full_result, indent=2))
    duration = (time.time()-t0)/3600.0
    print(f"with writing took {duration:.2e} hours")


def main():
    run_test(
        bootstrap_factor=1.0,
        bootstrap_iteration=1,
        output_path='assignment_230406_one_election.json')

    run_test(
        bootstrap_factor=0.9,
        bootstrap_iteration=100,
        output_path='assignment_230406_full_election.json')

    run_test(
        bootstrap_factor=0.9,
        bootstrap_iteration=100,
        flatten=True,
        output_path='assignment_230406_full_election_flat.json')

    run_test(
        bootstrap_factor=1.0,
        bootstrap_iteration=1,
        flatten=True,
        output_path='assignment_230406_one_election_flat.json')


if __name__ == "__main__":
    main()
