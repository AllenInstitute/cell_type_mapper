
from hierarchical_mapping.type_assignment.election import (
    run_type_assignment_on_h5ad_cpu
)

try:
    TORCH_AVAILABLE = False
    import torch  # type: ignore
    if torch.cuda.is_available():
        TORCH_AVAILABLE = True
        NUM_GPUS = torch.cuda.device_count()
        from hierarchical_mapping.gpu_utils.type_assignment.election import (
            run_type_assignment_on_h5ad_gpu)
except ImportError:
    TORCH_AVAILABLE = False
    NUM_GPUS = None


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
        normalization='log2CPM',
        tmp_dir=None,
        log=None,
        max_gb=10,
        results_output_path=None):
    if TORCH_AVAILABLE:
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
        normalization=normalization,
        tmp_dir=tmp_dir,
        log=log,
        max_gb=max_gb,
        results_output_path=results_output_path)
