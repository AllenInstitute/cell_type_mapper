from cell_type_mapper.type_assignment.election import (
    run_type_assignment_on_h5ad_cpu
)

from cell_type_mapper.type_assignment.utils import (
    validate_bootstrap_factor_lookup)

from cell_type_mapper.validation.utils import (
    is_data_ge_zero)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad
)


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
        output_taxonomy_tree=None,
        algorithm="hierarchical"):

    valid_algorithms = ("hierarchical", "hann")
    if algorithm not in valid_algorithms:
        raise ValueError(
            f"'{algorithm}' is not a valid algorithm; "
            f"only {valid_algorithms} are valid"
        )

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

    tmp_path_list = run_type_assignment_on_h5ad_cpu(
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
        results_output_path=results_output_path,
        algorithm=algorithm
    )

    return tmp_path_list
