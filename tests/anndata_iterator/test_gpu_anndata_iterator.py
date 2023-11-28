import pytest

import numpy as np
import os

try:
    import torch
    from cell_type_mapper.gpu_utils.anndata_iterator.anndata_iterator import (
        Collator)
except ImportError:
    pass

from cell_type_mapper.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)

from cell_type_mapper.utils.torch_utils import (
    is_torch_available)


def test_anndata_iterator_with_torch():
    """
    Check that the torch collator correctly handles
    data being cast from a torch-invalid uint to
    a safe int value.
    """
    if not is_torch_available():
        return

    env_var = 'AIBS_BKP_USE_TORCH'
    os.environ[env_var] = 'true'

    device = torch.device('cpu')

    rng = np.random.default_rng(22131)
    n_genes = 100000
    n_cells = 117
    raw_data = rng.integers(2**16-2**15, 2**16-1,
                            (n_cells, n_genes), dtype=np.int64)

    row_sums = raw_data.sum(axis=1)

    # make sure the data is so large that we would expect
    # an overflow error unless python was doing converting
    # to appropriate datatype automatically
    assert row_sums.shape == (n_cells, )
    max_sum = row_sums.max()
    assert max_sum > 2**32
    assert max_sum < 2**33

    gene_id = [f'g_{ii}' for ii in range(n_genes)]
    raw_cell_by_gene = CellByGeneMatrix(
        data=raw_data,
        gene_identifiers=gene_id,
        normalization='raw')
    
    marker_genes = ['g_4', 'g_33', 'g_55', 'g_78', 'g_91']
    raw_cell_by_gene.to_log2CPM_in_place()
    raw_cell_by_gene.downsample_genes_in_place(marker_genes)

    coll = Collator(
       all_query_identifiers=gene_id,
       normalization='raw',
       all_query_markers=marker_genes,
       device=device)

    # pass in batch as np.uint16, which is a data type
    # that torch cannot actually support as a tensor

    batch = [(raw_data.astype(np.uint16), 0, n_cells)]
    assert raw_data.shape == (n_cells, n_genes)
    (actual, r0, r1) = coll(batch)
    np.testing.assert_allclose(
        actual.data,
        raw_cell_by_gene.data,
        atol=0.0,
        rtol=1.0e-6)

    os.environ[env_var] = ''
