from cell_type_mapper.type_assignment.election import (
    run_type_assignment_on_h5ad_cpu
)

from cell_type_mapper.utils.torch_utils import (
    is_torch_available,
    use_torch)

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
