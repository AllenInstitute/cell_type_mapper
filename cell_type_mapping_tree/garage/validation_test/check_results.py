import anndata
import json
import pathlib

# read the taxonomy in from the json file you saved
# see how well it does at level_1 and level_2

def invert_tree(taxonomy_tree):
    cluster_to_l2 = dict()
    leaf_parent = taxonomy_tree['hierarchy'][-2]
    print(f"first parent {leaf_parent}")
    for parent in taxonomy_tree[leaf_parent]:
        for child in taxonomy_tree[leaf_parent][parent]:
            assert child not in cluster_to_l2
            cluster_to_l2[child] = parent

    cluster_to_l1 = dict()
    second_parent = taxonomy_tree['hierarchy'][-3]
    print(f"second parent {second_parent}")
    for parent in taxonomy_tree[second_parent]:
        for child in taxonomy_tree[second_parent][parent]:
            for leaf in taxonomy_tree[leaf_parent][child]:
                assert leaf not in cluster_to_l1
                cluster_to_l1[leaf] = parent

    return cluster_to_l2, cluster_to_l1

def main():
    query_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/changkyul/CIRRO/MFISH/atlas_brain_638850.remap.4334174.updated.imputed.h5ad'

    result_path = '/allen/aibs/technology/danielsf/knowledge_base/validation/assignment_230406_full_election.json'

    data_dir = pathlib.Path(
        '/allen/aibs/technology/danielsf/knowledge_base/validation')
    assert data_dir.is_dir()
    taxonomy_tree = json.load(open(data_dir / 'taxonomy_tree.json', 'rb'))

    cluster_to_l2, cluster_to_l1 = invert_tree(taxonomy_tree)

    results = json.load(open(result_path, 'rb'))['result']

    # get the truth
    truth_cache_path = pathlib.Path("data/truth_cache.json")
    if not truth_cache_path.is_file():
        a_data = anndata.read_h5ad(query_path, backed='r')
        obs = a_data.obs
        cell_to_truth = dict()
        for cell_id, cluster_value in zip(
                a_data.obs_names.values, obs['best.cl'].values):
           cell_to_truth[cell_id] = str(cluster_value)
        with open(truth_cache_path, "w") as out_file:
            out_file.write(json.dumps(cell_to_truth))

    cell_to_truth = json.load(open(truth_cache_path, "rb"))

    good = {'cluster': 0, 'level_2': 0, 'level_1':0}
    bad = {'cluster': 0, 'level_2': 0, 'level_1':0}
    bad_elements = []
    good_elements = []
    for element in results:
        found_cluster = element['cluster']['assignment']

        truth_cluster = cell_to_truth[element['cell_id']]
        truth_l1 = cluster_to_l1[truth_cluster]
        truth_l2 = cluster_to_l2[truth_cluster] 

        truth = {'cluster': truth_cluster,
                 'level_1': truth_l1,
                 'level_2': truth_l2}

        for k in ('cluster', 'level_1', 'level_2'):
            if element[k]['assignment'] == truth[k]:
                good[k] += 1
            else:
                bad[k] += 1

        element['true_cluster'] = truth_cluster
        element['true_level_1'] = truth_l1
        element['true_level_2'] = truth_l2

        assert type(truth_cluster) == type(found_cluster)
        if truth_cluster == found_cluster:
            good_elements.append(element)
        else:
            bad_elements.append(element)

        if (good['cluster']+bad['cluster']) % 10000 == 0:
            print('')
            for k in ('cluster', 'level_2', 'level_1'):
                print(f'{k} good {good[k]:.2e} bad {bad[k]:.2e}')

    print('')
    for k in ('cluster', 'level_2', 'level_1'):
        print(f'{k} good {good[k]:.2e} bad {bad[k]:.2e}')

    with open("good_v_bad.json", "w") as out_file:
        out_file.write(json.dumps({"good": good_elements, "bad":bad_elements}))

if __name__ == "__main__":
    main()
