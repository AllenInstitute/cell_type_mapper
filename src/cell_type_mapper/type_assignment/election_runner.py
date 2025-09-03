from cell_type_mapper.type_assignment.election import (
    run_type_assignment_on_h5ad_cpu
)

from cell_type_mapper.type_assignment.utils import (
    validate_bootstrap_factor_lookup)

from cell_type_mapper.utils.torch_utils import (
    is_torch_available,
    use_torch)

from cell_type_mapper.validation.utils import (
    is_data_ge_zero)

from cell_type_mapper.utils.output_utils import (
    re_order_blob)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad
)


if is_torch_available():
    from cell_type_mapper.gpu_utils.type_assignment.election import (
        run_type_assignment_on_h5ad_gpu)


def run_type_assignment_on_h5ad(
        query_h5ad_path,
        precomputed_stats_path,
        marker_gene_cache_path,
        taxonomy_tree,
        n_processors,
        chunk_size,
        bootstrap_factor_lookup,
        bootstrap_iteration,
        rng,
        n_assignments=10,
        normalization='log2CPM',
        tmp_dir=None,
        log=None,
        max_gb=10,
        results_output_path=None,
        output_taxonomy_tree=None):

    if normalization not in ('raw', 'log2CPM'):
        error_msg = (
            f"Do not know how to handle normalization = '{normalization}'; "
            "must be either 'raw' or 'log2CPM'"
        )
        if log is not None:
            log.error(error_msg)
        else:
            raise RuntimeError(error_msg)

    obs = read_df_from_h5ad(
        query_h5ad_path,
        df_name='obs')
    obs_idx = obs.index.values
    if len(obs_idx) != len(set(obs_idx)):
        msg = (
            "obs.index.values are not unique in "
            f"h5ad file {query_h5ad_path}"
        )
        if log is not None:
            log.error(msg)
        else:
            raise RuntimeError(msg)

    if normalization == 'raw':
        # check that data is >= 0
        if log is not None:
            log.info(
                "Scanning unlabeled data to check that it is >= 0"
            )
        is_ge_zero = is_data_ge_zero(h5ad_path=query_h5ad_path, layer='X')
        if not is_ge_zero[0]:
            error_msg = (
                f"Minimum expression value is {is_ge_zero[1]}; "
                "must be >= 0 (we will be taking the logarithm "
                "in order to convert from 'raw' to 'log2CPM' data)"
            )
            if log is not None:
                log.error(error_msg)
            else:
                raise RuntimeError(error_msg)
        if log is not None:
            log.info(
                "Verified that unlabeled data is >= 0"
            )

    validate_bootstrap_factor_lookup(
        bootstrap_factor_lookup=bootstrap_factor_lookup,
        taxonomy_tree=taxonomy_tree,
        log=log)

    if use_torch():
        result = run_type_assignment_on_h5ad_gpu(
            query_h5ad_path,
            precomputed_stats_path,
            marker_gene_cache_path,
            taxonomy_tree,
            n_processors,
            chunk_size,
            bootstrap_factor_lookup,
            bootstrap_iteration,
            rng,
            n_assignments=n_assignments,
            normalization=normalization,
            tmp_dir=tmp_dir,
            log=log,
            max_gb=max_gb,
            output_taxonomy_tree=output_taxonomy_tree,
            results_output_path=results_output_path)
    else:
        result = run_type_assignment_on_h5ad_cpu(
            query_h5ad_path,
            precomputed_stats_path,
            marker_gene_cache_path,
            taxonomy_tree,
            n_processors,
            chunk_size,
            bootstrap_factor_lookup,
            bootstrap_iteration,
            rng,
            n_assignments=n_assignments,
            normalization=normalization,
            tmp_dir=tmp_dir,
            log=log,
            max_gb=max_gb,
            output_taxonomy_tree=output_taxonomy_tree,
            results_output_path=results_output_path)

    result = re_order_blob(
        results_blob=result,
        query_path=query_h5ad_path)

    return result
