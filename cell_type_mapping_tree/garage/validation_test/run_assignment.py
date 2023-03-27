import time
import json
import numpy as np
import pathlib

from hierarchical_mapping.utils.taxonomy_utils import (
    convert_tree_to_leaves,
    get_siblings)

from hierarchical_mapping.type_assignment.election import (
    run_type_assignment_on_h5ad)

import os

def main():
    data_dir = pathlib.Path(
        '/allen/aibs/technology/danielsf/knowledge_base/validation')
    assert data_dir.is_dir()

    tree = json.load(open(data_dir / 'taxonomy_tree.json', 'rb'))

    query_data_path = pathlib.Path(
        '/allen/programs/celltypes/workgroups/rnaseqanalysis/changkyul/CIRRO/MFISH/atlas_brain_638850.remap.4334174.updated.imputed.h5ad')
    assert query_data_path.is_file()


    precomputed_path = data_dir / 'validation_test_precompute.h5'
    assert os.path.isfile(precomputed_path)


    t0 = time.time()
    marker_cache_path = data_dir / 'validation_marker_cache.h5'
    assert marker_cache_path.is_file()

    result = run_type_assignment_on_h5ad(
        query_h5ad_path=query_data_path,
        precomputed_stats_path=precomputed_path,
        marker_gene_cache_path=marker_cache_path,
        taxonomy_tree=tree,
        n_processors=32,
        chunk_size=30000,
        bootstrap_factor=0.9,
        bootstrap_iteration=100,
        rng=np.random.default_rng(11235))

    duration = (time.time()-t0)/3600.0
    print(f"marker cache creation took {duration:.2e} hours")

    output_path = data_dir / 'assignment_230327.json'
    with open(output_path, 'w') as out_file:
        out_file.write(json.dumps(result, indent=2))
    duration = (time.time()-t0)/3600.0
    print(f"with writing took {duration:.2e} hours")



if __name__ == "__main__":
    main()
