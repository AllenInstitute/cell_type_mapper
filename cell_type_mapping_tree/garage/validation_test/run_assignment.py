import copy
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

def copy_data_over():

    data_dir = pathlib.Path(
        '/allen/aibs/technology/danielsf/knowledge_base/validation')
    assert data_dir.is_dir()

    query_path = pathlib.Path(
        '/allen/programs/celltypes/workgroups/rnaseqanalysis/changkyul/CIRRO/MFISH/atlas_brain_638850.remap.4334174.updated.imputed.h5ad')
    assert query_path.is_file()

    #precompute_path = data_dir / 'validation_test_precompute.h5'
    precompute_path = data_dir / 'ck_precompute.h5'
    assert precompute_path.is_file()

    marker_path = data_dir / 'validation_marker_cache.h5'
    assert marker_path.is_file()

    tmp_dir = pathlib.Path(os.environ['TMPDIR'])

    result = {
        'query': {'new': tmp_dir/query_path.name, 'old': query_path},
        'marker': {'new': tmp_dir/marker_path.name, 'old': marker_path},
        'precompute': {'new': tmp_dir/precompute_path.name,
                       'old': precompute_path}}

    for k in result:
        pair = result[k]
        shutil.copy(src=pair['old'], dst=pair['new'])
        print(f"copied {pair}")
    return result



def run_test(
        bootstrap_factor=0.9,
        bootstrap_iteration=100,
        output_path=None,
        flatten=False,
        data_map=None):

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

    full_result['query_path'] = str(
        data_map['query']['old'].resolve().absolute())
    full_result['marker_path'] = str(
        data_map['marker']['old'].resolve().absolute())
    full_result['precompute_path'] = str(
        data_map['precompute']['old'].resolve().absolute())
    full_result['bootstrap_factor'] = bootstrap_factor
    full_result['bootstrap_iteration'] = int(bootstrap_iteration)

    t0 = time.time()
    result = run_type_assignment_on_h5ad(
        query_h5ad_path=data_map['query']['new'],
        precomputed_stats_path=data_map['precompute']['new'],
        marker_gene_cache_path=data_map['marker']['new'],
        taxonomy_tree=tree,
        n_processors=32,
        chunk_size=30000,
        bootstrap_factor=bootstrap_factor,
        bootstrap_iteration=bootstrap_iteration,
        rng=np.random.default_rng(11235))

    full_result['result'] = result

    duration = (time.time()-t0)/3600.0
    print(f"marker cache creation took {duration:.2e} hours")

    with open(output_path, 'w') as out_file:
        out_file.write(json.dumps(full_result, indent=2))
    duration = (time.time()-t0)/3600.0
    print(f"with writing took {duration:.2e} hours")


def main():

    data_map = copy_data_over()

    run_test(
        bootstrap_factor=1.0,
        bootstrap_iteration=1,
        data_map=data_map,
        output_path='assignment_230410_one_election.json')

    run_test(
        bootstrap_factor=0.9,
        bootstrap_iteration=100,
        data_map=data_map,
        output_path='assignment_230410_full_election.json')

    run_test(
        bootstrap_factor=0.9,
        bootstrap_iteration=100,
        data_map=data_map,
        flatten=True,
        output_path='assignment_230410_full_election_flat.json')

    run_test(
        bootstrap_factor=1.0,
        bootstrap_iteration=1,
        data_map=data_map,
        flatten=True,
        output_path='assignment_230410_one_election_flat.json')


if __name__ == "__main__":
    main()
