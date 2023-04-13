import argparse
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

import tempfile
from hierarchical_mapping.utils.utils import (
    _clean_up)

from hierarchical_mapping.type_assignment.election import (
    run_type_assignment_on_h5ad)

import os

def copy_data_over(
        query_path=None,
        precompute_path=None,
        marker_path=None,
        tmp_dir=None):

    query_path = pathlib.Path(query_path)
    assert query_path.is_file()
    precompute_path = pathlib.Path(precompute_path)
    assert precompute_path.is_file()
    marker_path = pathlib.Path(marker_path)
    assert marker_path.is_file()

    if tmp_dir is None:
        tmp_dir = pathlib.Path(os.environ['TMPDIR'])
    else:
        tmp_dir = pathlib.Path(tmp_dir)

    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(dir=tmp_dir))
    assert tmp_dir.is_dir()

    result = {
        'query': {'new': tmp_dir/query_path.name, 'old': query_path},
        'marker': {'new': tmp_dir/marker_path.name, 'old': marker_path},
        'precompute': {'new': tmp_dir/precompute_path.name,
                       'old': precompute_path}}

    for k in result:
        pair = result[k]
        shutil.copy(src=pair['old'], dst=pair['new'])
        print(f"copied {pair}")
    result['tmp_dir'] = tmp_dir
    return result



def run_test(
        bootstrap_factor=0.9,
        bootstrap_iteration=100,
        output_path=None,
        flatten=False,
        data_map=None,
        n_processors=32,
        chunk_size=30000):

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
        n_processors=n_processors,
        chunk_size=chunk_size,
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
    data_dir = pathlib.Path(
        '/allen/aibs/technology/danielsf/knowledge_base/validation')

    #query_path = pathlib.Path(
    #    '/allen/programs/celltypes/workgroups/rnaseqanalysis/changkyul/CIRRO/MFISH/atlas_brain_638850.remap.4334174.updated.imputed.h5ad')
    d_query_path = (
        '/allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/Taxonomies/AIT17.0_mouse/AIT17.0.logCPM.sampled100_a.h5ad')

    #precompute_path = data_dir / 'validation_test_precompute.h5'
    d_precompute_path = data_dir / 'ck_precompute.h5'
    d_precompute_path = str(d_precompute_path.resolve().absolute())

    d_marker_path = data_dir / 'validation_marker_cache_noboot.h5'
    d_marker_path = str(d_marker_path.resolve().absolute())

    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', type=str, default=d_query_path)
    parser.add_argument('--precompute_path', type=str,
                        default=d_precompute_path)
    parser.add_argument('--marker_path', type=str, default=d_marker_path)
    parser.add_argument('--tmp_dir', type=str, default=None)
    parser.add_argument('--bootstrap_factor', type=float, default=0.9)
    parser.add_argument('--bootstrap_iteration', type=int, default=100)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--n_processors', type=int, default=32)
    parser.add_argument('--chunk_size', type=int, default=30000)
    args = parser.parse_args()

    assert args.output_path is not None

    data_map = copy_data_over(
        precompute_path=args.precompute_path,
        query_path=args.query_path,
        marker_path=args.marker_path,
        tmp_dir = args.tmp_dir)

    run_test(
        bootstrap_factor=args.bootstrap_factor,
        bootstrap_iteration=args.bootstrap_iteration,
        data_map=data_map,
        output_path=args.output_path,
        n_processors=args.n_processors,
        chunk_size=args.chunk_size)

    tmp_dir = data_map['tmp_dir']
    print(f"cleaning {tmp_dir}")
    _clean_up(tmp_dir)

if __name__ == "__main__":
    main()
