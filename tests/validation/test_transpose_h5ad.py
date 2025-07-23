"""
Tests for the case where the columns of an h5ad file are cells and
the rows are genes.
"""
import pytest

import anndata
import itertools
import numpy as np
import pandas as pd
import scipy.sparse
import warnings

from cell_type_mapper.utils.utils import (
    mkstemp_clean
)

from cell_type_mapper.test_utils.anndata_utils import (
    write_anndata_x_to_csv
)

from cell_type_mapper.test_utils.gene_mapping.mouse_gene_id_lookup import (
    mouse_gene_id_lookup
)

from cell_type_mapper.test_utils.gene_mapping.human_gene_id_lookup import (
    human_gene_id_lookup
)

from cell_type_mapper.validation.validate_h5ad import (
    _transpose_file_if_necessary,
    validate_h5ad
)


@pytest.fixture()
def density_fixture(request):
    return request.param


@pytest.fixture()
def species_fixture(request):
    return request.param


@pytest.fixture()
def h5ad_fixture(
        tmp_dir_fixture,
        density_fixture,
        species_fixture):

    rng = np.random.default_rng(6611223)
    n_cells = 613
    n_genes = 255
    n_tot = n_cells*n_genes
    data = np.zeros(n_tot, dtype=int)
    chosen_idx = rng.choice(
        np.arange(n_tot, dtype=int),
        n_tot//5,
        replace=False)
    data[chosen_idx] = 100.0*rng.random(len(chosen_idx))
    data = data.reshape((n_cells, n_genes))

    obs = pd.DataFrame(
        [{'cell_id': f'c_{ii}', 'sq': ii**2}
         for ii in range(n_cells)]
    ).set_index('cell_id')

    if species_fixture == 'mouse':
        gene_labels = rng.choice(
            list(mouse_gene_id_lookup.keys()),
            n_genes,
            replace=False)
    elif species_fixture == 'human':
        gene_labels = rng.choice(
            list(human_gene_id_lookup.keys()),
            n_genes,
            replace=False)
    elif species_fixture == 'NA':
        gene_labels = [f'g_{ii}' for ii in range(n_genes)]
    else:
        raise RuntimeError(
            f"Do not understand species {species_fixture}"
        )

    var = pd.DataFrame(
        [{'gene_id': gene_labels[ii], 'cube': ii**3}
         for ii in range(n_genes)]
    ).set_index('gene_id')

    correct_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='correct_transposition_',
        suffix='.h5ad'
    )
    transposed_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='transposed_',
        suffix='.h5ad'
    )

    if density_fixture == 'csc':
        xx = scipy.sparse.csc_matrix(data)
        xx_t = scipy.sparse.csc_matrix(data.transpose())
    elif density_fixture == 'csr':
        xx = scipy.sparse.csr_matrix(data)
        xx_t = scipy.sparse.csr_matrix(data.transpose())
    else:
        xx = data
        xx_t = data.transpose()

    correct = anndata.AnnData(
        obs=obs,
        var=var,
        X=xx)
    correct.write_h5ad(correct_path)

    transposed = anndata.AnnData(
        obs=var,
        var=obs,
        X=xx_t
    )
    transposed.write_h5ad(transposed_path)

    return {
        'correct': correct_path,
        'transposed': transposed_path
    }


@pytest.mark.parametrize(
    'density_fixture,species_fixture',
    itertools.product(
        ('csr', 'csc', 'dense'),
        ('mouse', 'human', 'NA')
    ),
    indirect=['density_fixture', 'species_fixture']
)
def test_h5ad_transposition_from_genes(
        density_fixture,
        species_fixture,
        h5ad_fixture,
        tmp_dir_fixture):

    (new_path,
     was_transposed) = _transpose_file_if_necessary(
         src_path=h5ad_fixture['correct'],
         tmp_dir=tmp_dir_fixture,
         log=None)

    assert not was_transposed
    assert new_path == h5ad_fixture['correct']

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        (new_path,
         was_transposed) = _transpose_file_if_necessary(
             src_path=h5ad_fixture['transposed'],
             tmp_dir=tmp_dir_fixture,
             log=None)

    if species_fixture != 'NA':
        assert was_transposed
        assert new_path != h5ad_fixture['transposed']

        expected = anndata.read_h5ad(
            h5ad_fixture['correct'],
            backed='r'
        )
        actual = anndata.read_h5ad(
            new_path,
            backed='r'
        )
        pd.testing.assert_frame_equal(
            expected.obs,
            actual.obs
        )
        pd.testing.assert_frame_equal(
            expected.var,
            actual.var
        )
        expected_x = expected.X[()]
        actual_x = actual.X[()]
        if density_fixture != 'dense':
            expected_x = expected_x.toarray()
            actual_x = actual_x.toarray()
        np.testing.assert_allclose(
            expected_x,
            actual_x,
            atol=0.0,
            rtol=1.0e-7
        )
    else:
        assert not was_transposed
        assert new_path == h5ad_fixture['transposed']


@pytest.mark.parametrize(
    'density_fixture,species_fixture,round_to_int,as_csv',
    itertools.product(
        ('csr', 'csc', 'dense'),
        ('mouse',),
        (True, False),
        (True, False)
    ),
    indirect=['density_fixture', 'species_fixture']
)
def test_validation_of_transposed_h5ad_files(
        density_fixture,
        species_fixture,
        h5ad_fixture,
        round_to_int,
        as_csv,
        tmp_dir_fixture):
    """
    Test that validate_h5ad can handle transposed files
    (including transposed CSV files)
    """

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        if as_csv:
            if density_fixture != 'dense':
                return
            test_src_path = mkstemp_clean(
                dir=tmp_dir_fixture,
                prefix='transposed_file_',
                suffix='.csv'
            )
            write_anndata_x_to_csv(
                anndata_obj=anndata.read_h5ad(h5ad_fixture['transposed']),
                dst_path=test_src_path,
                cell_label_header=False,
                cell_label_type='string'
            )
        else:
            test_src_path = h5ad_fixture['transposed']

        baseline_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='validated_baseline_',
            suffix='.h5ad'
        )

        baseline_path = validate_h5ad(
            h5ad_path=h5ad_fixture['correct'],
            log=None,
            tmp_dir=tmp_dir_fixture,
            layer='X',
            round_to_int=round_to_int,
            output_dir=None,
            valid_h5ad_path=baseline_path
        )[0]

        if baseline_path is None:
            baseline_path = h5ad_fixture['correct']

        test_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='validated_test_',
            suffix='.h5ad'
        )

        validate_h5ad(
            h5ad_path=test_src_path,
            log=None,
            tmp_dir=tmp_dir_fixture,
            layer='X',
            round_to_int=round_to_int,
            output_dir=None,
            valid_h5ad_path=test_path
        )

    expected = anndata.read_h5ad(baseline_path, backed='r')
    actual = anndata.read_h5ad(test_path, backed='r')

    if not as_csv:
        pd.testing.assert_frame_equal(
            expected.obs,
            actual.obs
        )
        pd.testing.assert_frame_equal(
            expected.var,
            actual.var
        )
    else:
        np.testing.assert_array_equal(
            expected.obs.index.values,
            actual.obs.index.values
        )
        np.testing.assert_array_equal(
            expected.var.index.values,
            actual.var.index.values
        )

    expected_x = expected.X[()]
    actual_x = actual.X[()]

    if density_fixture != 'dense':
        expected_x = expected_x.toarray()
        actual_x = actual_x.toarray()
    np.testing.assert_allclose(
        expected_x,
        actual_x,
        atol=0.0,
        rtol=1.0e-7
    )
