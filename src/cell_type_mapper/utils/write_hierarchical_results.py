import copy
import json
import pathlib

import cell_type_mapper.utils.utils as ctm_utils
import cell_type_mapper.utils.output_utils as output_utils
import cell_type_mapper.utils.anndata_utils as anndata_utils
import cell_type_mapper.taxonomy.taxonomy_tree as tree_module


def write_hierarchical_output(
        mapping_result,
        metadata,
        config,
        log,
        mapping_exception,
        write_to_disk):

    if config['extended_result_path'] is not None:
        output_path = pathlib.Path(config['extended_result_path'])
    else:
        output_path = None

    if config['hdf5_result_path'] is not None:
        hdf5_output_path = pathlib.Path(config['hdf5_result_path'])
    else:
        hdf5_output_path = None

    if config['log_path'] is not None:
        log_path = pathlib.Path(config['log_path'])
    else:
        log_path = None

    if config['summary_metadata_path'] is not None:
        write_summary_metadata_to_disk(
            summary_metadata_path=config['summary_metadata_path'],
            mapping_result=mapping_result,
            metadata=metadata,
            query_path=config['query_path']
        )

    if config['csv_result_path'] is not None:
        write_mapping_to_csv(
            mapping_result=mapping_result,
            metadata=metadata,
            csv_output_path=config['csv_result_path'],
            full_output_path=output_path
        )

    if config['obsm_key'] is not None:
        write_mapping_to_obsm(
            mapping_result=mapping_result,
            metadata=metadata,
            query_path=config['query_path'],
            obsm_key=config['obsm_key'],
            obsm_clobber=config['obsm_clobber']
        )

    if write_to_disk:
        write_mapping_to_disk(
            metadata=metadata,
            mapping_result=mapping_result,
            log=log,
            log_path=log_path,
            output_path=output_path,
            hdf5_output_path=hdf5_output_path,
            cloud_safe=config['cloud_safe']
        )
        return None

    return {
        'metadata': metadata,
        'mapping_result': mapping_result,
        'log': log,
        'log_path': log_path,
        'output_path': output_path,
        'hdf5_output_path': hdf5_output_path,
        'mapping_exception': mapping_exception
    }


def write_mapping_to_disk(
        mapping_result,
        metadata,
        log,
        log_path,
        output_path,
        hdf5_output_path,
        cloud_safe):

    output = copy.deepcopy(metadata)
    if mapping_result is not None:
        output['results'] = mapping_result

    if log_path is not None:
        log.write_log(log_path, cloud_safe=cloud_safe)

    if output_path is not None:
        with open(output_path, "w") as out_file:
            out_file.write(
                json.dumps(
                    ctm_utils.clean_for_json(output), indent=2
                )
            )

    if hdf5_output_path is not None:
        output_utils.blob_to_hdf5(
            output_blob=output,
            dst_path=hdf5_output_path)


def write_mapping_to_csv(
        mapping_result,
        metadata,
        full_output_path,
        csv_output_path):

    if mapping_result is None:
        return None

    config = metadata['config']
    result = mapping_result

    tree_for_metadata = tree_module.TaxonomyTree(
        data=metadata['taxonomy_tree']
    )

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
       mapping_result,
       metadata,
       query_path,
       obsm_key,
       obsm_clobber):

    if mapping_result is None:
        return None

    tree_for_metadata = tree_module.TaxonomyTree(metadata['taxonomy_tree'])

    df = output_utils.blob_to_df(
        results_blob=mapping_result,
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


def write_summary_metadata_to_disk(
        summary_metadata_path,
        mapping_result,
        metadata,
        query_path):
    if mapping_result is None:
        return None
    n_mapped_cells = len(mapping_result)

    n_total_genes = len(
        anndata_utils.read_df_from_h5ad(
            query_path,
            df_name='var'
        )
    )

    n_mapped_genes = (
        n_total_genes - metadata.pop('n_unmapped_genes')
    )

    with open(summary_metadata_path, 'w') as dst:
        dst.write(
            json.dumps(
                {
                 'n_mapped_cells': int(n_mapped_cells),
                 'n_mapped_genes': int(n_mapped_genes)
                },
                indent=2
            )
        )
