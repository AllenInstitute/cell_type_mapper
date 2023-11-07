from cell_type_mapper.cli.map_to_on_the_fly_markers import (
    OnTheFlyMapper)

def main():
    sea_ad_dir='/allen/aibs/technology/danielsf/sea_ad'
    mapping_dir=sea_ad_dir+'/mapper'

    # these need to be re-directed to point to somewhere
    # the cloud code can write to
    json_result_path = mapping_dir + '/test_json.json'
    csv_result_path = mapping_dir + '/test_csv.csv'

    # on-prem paths
    #precompute_path=mapping_dir+'/precomputed_stats.sea_ad.h5'
    #query_path=sea_ad_dir+'/test_data/sea_ad_eg_10k_cells.h5ad'

    query_path = # path to s3://sea-ad-hierarchical-prototype/sea_ad_eg_10k_cells.h5ad

    precompute_path = # path to s3://sea-ad-hierarchical-prototype/precomputed_stats.sea_ad.h5

    config = {
        'precomputed_stats': {'path': precompute_path},
        'type_assignment': {
            'normalization': 'raw',
            'bootstrap_iteration': 100,
            'bootstrap_factor': 0.9,
            'chunk_size': 5000,
            'rng_seed': 542119,
            'n_runners_up': 5},
        'query_path': query_path,
        'extended_result_path': json_result_path,
        'csv_result_path': csv_result_path,
        'tmp_dir': None,
        'cloud_safe': True,
        'n_processors': 8,
        'flatten': False, # will be True for correlation mapping,
        'drop_level': None,
        'max_gb': 32,
        'query_markers': {'n_per_utility': 15 },
        'reference_markers': {
            'exact_penetrance': False,
            'q1_th': 0.5,
            'q1_min_th': 0.1,
            'qdiff_th': 0.7,
            'qdiff_min_th': 0.1,
            'log2_fold_th': 1.0,
            'log2_fold_min_th': 0.5,
            'n_valid': 30 }
    }

    runner = OnTheFlyMapper(
        args=[],
        input_data=config)

    runner.run()


if __name__ == "__main__":
    main()
