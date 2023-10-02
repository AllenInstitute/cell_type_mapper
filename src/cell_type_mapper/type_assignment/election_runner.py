from cell_type_mapper.type_assignment.election import (
    run_type_assignment_on_h5ad_cpu
)

from cell_type_mapper.utils.torch_utils import (
    is_torch_available,
    use_torch)

from cell_type_mapper.validation.utils import (
    get_minmax_x_from_h5ad)

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
        bootstrap_factor,
        bootstrap_iteration,
        rng,
        n_assignments=10,
        normalization='log2CPM',
        tmp_dir=None,
        log=None,
        max_gb=10,
        results_output_path=None):

    if normalization not in ('raw', 'log2CPM'):
        error_msg = (
            f"Do not know how to handle normalization = '{normalization}'; "
            "must be either 'raw' or 'log2CPM'"
        )
        if log is not None:
            log.error(error_msg)
        else:
            raise RuntimeError(error_msg)

    if normalization == 'raw':
        # check that data is >= 0
        minmax = get_minmax_x_from_h5ad(query_h5ad_path)
        if minmax[0] < 0:
            error_msg = (
                f"Minimum expression value is {minmax[0]}; "
                "must be >= 0 (we will be taking the logarithm "
                "in order to convert from 'raw' to 'log2CPM' data)"
            )
            if log is not None:
                log.error(error_msg)
            else:
                raise RuntimeError(error_msg)

    if use_torch():
        return run_type_assignment_on_h5ad_gpu(
            query_h5ad_path,
            precomputed_stats_path,
            marker_gene_cache_path,
            taxonomy_tree,
            n_processors,
            chunk_size,
            bootstrap_factor,
            bootstrap_iteration,
            rng,
            n_assignments=n_assignments,
            normalization=normalization,
            tmp_dir=tmp_dir,
            log=log,
            max_gb=max_gb,
            results_output_path=results_output_path)
    return run_type_assignment_on_h5ad_cpu(
        query_h5ad_path,
        precomputed_stats_path,
        marker_gene_cache_path,
        taxonomy_tree,
        n_processors,
        chunk_size,
        bootstrap_factor,
        bootstrap_iteration,
        rng,
        n_assignments=n_assignments,
        normalization=normalization,
        tmp_dir=tmp_dir,
        log=log,
        max_gb=max_gb,
        results_output_path=results_output_path)
