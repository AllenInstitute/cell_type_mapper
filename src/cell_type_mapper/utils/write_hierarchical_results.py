import json

import cell_type_mapper.utils.utils as cpm_utils
import cell_type_mapper.utils.output_utils as output_utils
import cell_type_mapper.utils.anndata_utils as anndata_utils
import cell_type_mapper.taxonomy.taxonomy_tree as tree_module


def write_mapping_to_disk(
        output,
        log,
        log_path,
        output_path,
        hdf5_output_path,
        cloud_safe):

    if log_path is not None:
        log.write_log(log_path, cloud_safe=cloud_safe)

    if output_path is not None:
        with open(output_path, "w") as out_file:
            out_file.write(
                json.dumps(
                    cpm_utils.clean_for_json(output), indent=2
                )
            )

    if hdf5_output_path is not None:
        output_utils.blob_to_hdf5(
            output_blob=output,
            dst_path=hdf5_output_path)


def write_mapping_to_csv(
        output,
        full_output_path,
        csv_output_path):

    if 'results' not in output:
        return None

    config = output['config']
    result = output['results']

    tree_for_metadata = tree_module.TaxonomyTree(data=output['taxonomy_tree'])

    if config['type_assignment']['bootstrap_iteration'] == 1 \
            or config['verbose_csv']:

        confidence_key = 'avg_correlation'
        confidence_label = 'correlation_coefficient'

    else:

        confidence_key = 'bootstrapping_probability'
        confidence_label = 'bootstrapping_probability'

    if config['verbose_csv']:
        valid_suffixes = [
            '_aggregate_probability',
            '_bootstrapping_probability',
            '_avg_correlation'
        ]
    else:
        valid_suffixes = None

    check_consistency = False
    if config['type_assignment']['bootstrap_iteration'] > 1:
        if config['flatten']:
            check_consistency = True

    output_utils.blob_to_csv(
        results_blob=result,
        taxonomy_tree=tree_for_metadata,
        output_path=csv_output_path,
        metadata_path=full_output_path,
        confidence_key=confidence_key,
        confidence_label=confidence_label,
        config=config,
        valid_suffixes=valid_suffixes,
        check_consistency=check_consistency,
        rows_at_a_time=100000)


def write_mapping_to_obsm(
       output,
       query_path,
       obsm_key,
       obsm_clobber):

    if 'results' not in output:
        return None

    tree_for_metadata = tree_module.TaxonomyTree(output['taxonomy_tree'])

    df = output_utils.blob_to_df(
        results_blob=output['results'],
        taxonomy_tree=tree_for_metadata).set_index('cell_id')

    # need to make sure that the rows are written in
    # the same order that they occur in the obs
    # dataframe

    obs = anndata_utils.read_df_from_h5ad(
        h5ad_path=query_path,
        df_name='obs')

    df = df.loc[obs.index.values]

    anndata_utils.append_to_obsm(
        h5ad_path=query_path,
        obsm_key=obsm_key,
        obsm_value=df,
        clobber=obsm_clobber)
