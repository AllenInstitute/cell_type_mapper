import numpy as np


def assert_blobs_equal(blob0, blob1):
    """
    Assert two output blobs are equal
    """
    assert set(blob0.keys()) == set(blob1.keys())

    assert_mappings_equal(
        blob0['results'],
        blob1['results']
    )

    assert blob0['taxonomy_tree'] == blob1['taxonomy_tree']
    assert blob0['marker_genes'] == blob1['marker_genes']
    assert blob0['n_unmapped_genes'] == blob1['n_unmapped_genes']
    assert blob0['marker_genes'] == blob1['marker_genes']
    assert blob0['gene_identifier_mapping'] == \
        blob1['gene_identifier_mapping']


def assert_mappings_equal(result0, result1, eps=10e-6):
    """
    Assert that two mapping 'results' entries
    are equal.
    """

    cell_id0 = np.array([
        cell['cell_id'] for cell in result0
    ])
    cell_id1 = np.array([
        cell['cell_id'] for cell in result1
    ])
    if not np.array_equal(cell_id0, cell_id1):
        raise RuntimeError(
            "cell_id entries do not align"
        )
    for level in result0[0]:
        if level == 'cell_id':
            continue
        for key in result0[0][level]:
            _assert_element_equal(
                result0=result0,
                result1=result1,
                level=level,
                key=key,
                eps=eps
            )


def _assert_element_equal(result0, result1, level, key, eps=1.0e-6):

    if isinstance(result0[0][level][key], list):
        val0 = np.concatenate([
            cell[level][key] for cell in result0
        ])
        val1 = np.concatenate([
            cell[level][key] for cell in result1
        ])
    else:
        val0 = np.array([
            cell[level][key] for cell in result0
        ])
        val1 = np.array([
            cell[level][key] for cell in result1
        ])

    if np.issubdtype(val0.dtype, np.number):
        if not np.allclose(val0, val1, atol=0.0, rtol=eps):
            raise RuntimeError(
                f"{level}:{key} mismatch"
            )
    else:
        if not np.array_equal(val0, val1):
            raise RuntimeError(
                f"{level}:{key} mismatch"
            )
