from cell_type_mapper.cli.from_specified_markers import (
     FromSpecifiedMarkersRunner)

def main():

    config = dict()
    config['precomputed_stats'] = {
       'path': '/allen/aibs/technology/danielsf/knowledge_base/scratch/abc_revision_230830/abc_stats_230809.h5'}
    config['query_markers'] = {
        'serialized_lookup': '/allen/aibs/technology/danielsf/knowledge_base/scratch/abc_revision_230830/mouse_markers_230809_old_subclass.json'}
    config['type_assignment'] = {
        'rng_seed': 776622,
        'bootstrap_iteration': 100,
        'bootstrap_factor': 0.9,
        'normalization': 'log2CPM',
        'n_processors': 6,
        'chunk_size': 10000,
        'n_runners_up': 3
    }

    config['flatten'] = False
    config['max_gb'] = 40
    config['extended_result_path'] = '/allen/aibs/technology/danielsf/knowledge_base/scratch/abc_revision_230830/abc_mapping_230809.json'
    config['obsm_key'] = None
    config['query_path'] = '/allen/aibs/technology/danielsf/knowledge_base/scratch/abc_revision_230830/all.v9_locked.0803_VALIDATED_20230809135708.h5ad'
    config['drop_level'] = 'CCN20230722_SUPT'
    config['tmp_dir'] = '/local1/scott_daniel/scratch'

    runner = FromSpecifiedMarkersRunner(
        args=[],
        input_data=config)

    runner.run()

if __name__ == "__main__":
    main()
