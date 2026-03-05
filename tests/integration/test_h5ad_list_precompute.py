"""
Tests for the infrastructure to create a precomputed stats file
from a list of h5ad files and a QC CSV file
"""

import pytest

import anndata
import copy
import h5py
import itertools
import json
import numpy as np
import pandas as pd
import tempfile

import cell_type_mapper.utils.utils as basic_utils
import cell_type_mapper.cell_by_gene.utils as cbg_utils
import cell_type_mapper.taxonomy.taxonomy_tree as taxonomy_module
import cell_type_mapper.cli.precompute_stats_h5ad_list as cli


@pytest.fixture(scope='module')
def local_tmp_dir(tmp_dir_fixture):
    tmp_dir = tempfile.mkdtemp(
        dir=tmp_dir_fixture,
        prefix='precompute_from_h5ad_list_'
    )
    yield tmp_dir
    basic_utils._clean_up(tmp_dir)


@pytest.fixture(scope='function')
def h5ad_list_fixture(
        local_tmp_dir,
        request):
    """
    Return list of h5ad file, full (log normalized)
    X array, and params
    """
    # params will encode
    # layer
    # gene_id_col
    # normalization

    params = copy.deepcopy(request.param)

    n_total_cells = 100
    n_genes = 2130

    rng = np.random.default_rng(1231)
    full_xx_raw = np.zeros(n_total_cells*n_genes, dtype=int)
    chosen_idx = rng.choice(
        np.arange(n_total_cells*n_genes),
        n_total_cells*n_genes//20,
        replace=False
    )
    full_xx_raw[chosen_idx] = rng.integers(1, 50000, len(chosen_idx))
    full_xx_raw = full_xx_raw.reshape((n_total_cells, n_genes))
    full_xx_raw[:, 3] = 10   # just to make sure no zeros in log2CPM

    # engineer some cells to have gt1 != ge1
    for i_cell in range(50):
        chosen_idx = rng.choice(
            np.arange(n_genes-1),
            1000,
            replace=False
        )
        full_xx_raw[i_cell, chosen_idx] = rng.integers(
            1,
            10,
            len(chosen_idx)
        )
        tot = full_xx_raw[i_cell, :].sum()
        if tot < 1000000:
            full_xx_raw[i_cell, -1] = 1000000-tot

    full_xx_norm = cbg_utils.convert_to_cpm(
        data=full_xx_raw,
        counts_per=1.0e6
    )

    full_xx_norm = np.log2(full_xx_norm+1.0)

    chunk_size = 17
    h5ad_path_list = []
    for i0 in range(0, n_total_cells, chunk_size):
        h5ad_path = basic_utils.mkstemp_clean(
            dir=local_tmp_dir,
            suffix='.h5ad'
        )
        i1 = min(n_total_cells, i0+chunk_size)
        obs = pd.DataFrame(
            [{'cell': f'cell{ii}'}
             for ii in range(i0, i1, 1)]
        ).set_index('cell')

        if params['gene_id_col'] is None:
            var = pd.DataFrame(
                [{'gene': f'g{ii}'}
                 for ii in range(n_genes)]
            ).set_index('gene')
        else:
            var = pd.DataFrame(
                [{'alt': f'a{ii}',
                  params['gene_id_col']: f'g{ii}'}
                 for ii in range(n_genes)]
            ).set_index('alt')

        if params['normalization'] == 'raw':
            xx = full_xx_raw[i0:i1, :]
        else:
            xx = full_xx_norm[i0:i1, :]

        if params['layer'] == 'X':
            x_data = xx
            layers = None
        else:
            x_data = np.ones(xx.shape, dtype=float)
            layers = {params['layer']: xx}

        adata = anndata.AnnData(
            obs=obs,
            var=var,
            X=x_data,
            layers=layers
        )
        adata.write_h5ad(h5ad_path)
        h5ad_path_list.append(h5ad_path)

    return h5ad_path_list, full_xx_norm, params


@pytest.fixture(scope='function')
def csv_path_fixture(local_tmp_dir, request):
    """
    Path to CSV file; also params
    """
    params = copy.deepcopy(request.param)
    # params contains
    # qc_column
    # cell_label_column

    rng = np.random.default_rng(21231)

    data = []
    for ii in range(11, 100, 1):
        this = {params['cell_label_column']: f'cell{ii}'}
        this['idx'] = ii
        this['class'] = f'class{ii // 40}'
        this['subclass'] = f'subclass{ii // 20}'
        this['cluster'] = f'cluster{ii // 5:04d}'
        if ii % 11 == 0 or ii % 23 == 0 or ii % 19 == 0:
            this[params['qc_column']] = False
        else:
            this[params['qc_column']] = True
        data.append(this)

    csv_path = basic_utils.mkstemp_clean(
        dir=local_tmp_dir,
        suffix='.csv'
    )

    rng.shuffle(data)

    pd.DataFrame(data).to_csv(csv_path, index=False)

    return csv_path, params


@pytest.mark.parametrize(
    "h5ad_list_fixture, csv_path_fixture, chunk_size",
    itertools.product(
        [{"normalization": "raw",
          "gene_id_col": None,
          "layer": "X"},
         {"normalization": "log2CPM",
          "gene_id_col": None,
          "layer": "X"},
         {"normalization": "raw",
          "gene_id_col": "identification",
          "layer": "X"},
         {"normalization": "raw",
          "gene_id_col": None,
          "layer": "alt"},
         {"normalization": "log2CPM",
          "gene_id_col": None,
          "layer": "alt"},
         {"normalization": "log2CPM",
          "gene_id_col": "symbol",
          "layer": "alt"},
         {"normalization": "raw",
          "gene_id_col": "symbol",
          "layer": "alt"},
         ],
        [{"cell_label_column": "silly",
          "qc_column": "forrlz"},
         {"cell_label_column": "cell_label",
          "qc_column": "qc"}
         ],
        [100, 8]
     ),
    indirect=["h5ad_list_fixture",
              "csv_path_fixture"]
)
def test_precompute_from_h5ad_list(
        local_tmp_dir,
        h5ad_list_fixture,
        csv_path_fixture,
        chunk_size):

    h5ad_list = h5ad_list_fixture[0]
    cbg_normalized_data = h5ad_list_fixture[1]
    h5ad_params = h5ad_list_fixture[2]
    csv_path = csv_path_fixture[0]
    csv_params = csv_path_fixture[1]

    dst_path = basic_utils.mkstemp_clean(
        dir=local_tmp_dir,
        suffix='.h5'
    )

    config = {
        "output_path": str(dst_path),
        "hierarchy": ['class', 'subclass', 'cluster'],
        "h5ad_path_list": h5ad_list,
        "annotation_path": csv_path,
        "qc_column": csv_params['qc_column'],
        "cell_label_column": csv_params['cell_label_column'],
        "layer": h5ad_params['layer'],
        "n_processors": 2,
        "tmp_dir": local_tmp_dir,
        "clobber": True,
        "normalization": h5ad_params['normalization'],
        "gene_id_col": h5ad_params["gene_id_col"],
        "chunk_size": chunk_size
    }
    runner = cli.PrecomputationH5adListRunner(
        args=[],
        input_data=config
    )
    runner.run()

    taxonomy_df = pd.read_csv(csv_path)
    taxonomy_df = taxonomy_df[taxonomy_df[csv_params['qc_column']]]

    expected_tree = taxonomy_module.TaxonomyTree.from_dataframe(
        taxonomy_df,
        column_hierarchy=['class', 'subclass', 'cluster'],
        drop_rows=True
    )

    actual_tree = taxonomy_module.TaxonomyTree.from_precomputed_stats(
        dst_path
    )

    assert actual_tree.is_equal_to(expected_tree)

    cluster_list = [f'cluster{ii:04d}' for ii in range(2, 20)]
    expected_gene_names = [f'g{ii}' for ii in range(2130)]
    with h5py.File(dst_path, 'r') as actual:

        gene_names = json.loads(actual['col_names'][()].decode('utf-8'))
        np.testing.assert_array_equal(
            desired=expected_gene_names,
            actual=gene_names
        )
        cluster_to_row = {
            cl: ii
            for ii, cl in enumerate(cluster_list)
        }
        actual_rows = json.loads(actual['cluster_to_row'][()].decode('utf-8'))
        assert actual_rows == cluster_to_row

        actual_n_cells = actual['n_cells'][()]
        assert actual_n_cells.min() > 0

        expected_n_cells = [
            (taxonomy_df.cluster == cluster).sum()
            for cluster in cluster_list
        ]

        np.testing.assert_array_equal(
            desired=expected_n_cells,
            actual=actual_n_cells
        )

        for ii, cluster in enumerate(cluster_list):
            subset = taxonomy_df[taxonomy_df.cluster == cluster]
            idx_arr = np.sort(subset.idx.values)
            x_subset = cbg_normalized_data[idx_arr, :]
            expected_sum = x_subset.sum(axis=0)
            np.testing.assert_allclose(
                expected_sum,
                actual['sum'][ii, :],
                atol=0.0,
                rtol=1.0e-6
            )
            expected_sumsq = (x_subset**2).sum(axis=0)
            np.testing.assert_allclose(
                expected_sumsq,
                actual['sumsq'][ii, :],
                atol=0.0,
                rtol=1.0e-6
            )
            expected_ge1 = (x_subset >= 1).sum(axis=0)
            np.testing.assert_array_equal(
                actual=actual['ge1'][ii, :],
                desired=expected_ge1
            )
            expected_gt0 = (x_subset > 0).sum(axis=0)
            np.testing.assert_array_equal(
                actual=actual['gt0'][ii, :],
                desired=expected_gt0
            )
            expected_gt1 = (x_subset > 1).sum(axis=0)
            np.testing.assert_array_equal(
                actual=actual['gt1'][ii, :],
                desired=expected_gt1
            )


@pytest.mark.parametrize(
    "h5ad_list_fixture",
    [{"normalization": "raw",
      "gene_id_col": None,
      "layer": "X"},
     ],
    indirect=["h5ad_list_fixture"]
)
def test_no_cells_precompute_from_h5ad_list(
        local_tmp_dir,
        h5ad_list_fixture):
    """
    Test that correct failure is raised when no cells
    in the CSV align with the h5ad file
    """
    h5ad_list = h5ad_list_fixture[0]
    h5ad_params = h5ad_list_fixture[2]

    csv_path = basic_utils.mkstemp_clean(
        dir=local_tmp_dir,
        suffix='.csv'
    )
    data = [
        {'label': f'label{ii}',
         'qc': True,
         'class': f'class{ii}',
         'subclass': f'subclass{ii}',
         'cluster': f'cluster{ii}'}
        for ii in range(10)
    ]
    pd.DataFrame(data).to_csv(csv_path, index=False)

    dst_path = basic_utils.mkstemp_clean(
        dir=local_tmp_dir,
        suffix='.h5'
    )

    config = {
        "output_path": str(dst_path),
        "hierarchy": ['class', 'subclass', 'cluster'],
        "h5ad_path_list": h5ad_list,
        "annotation_path": csv_path,
        "qc_column": 'qc',
        "cell_label_column": 'label',
        "layer": h5ad_params['layer'],
        "n_processors": 2,
        "tmp_dir": local_tmp_dir,
        "clobber": True,
        "normalization": h5ad_params['normalization'],
        "gene_id_col": h5ad_params["gene_id_col"],
        "chunk_size": 100
    }

    runner = cli.PrecomputationH5adListRunner(
        args=[],
        input_data=config
    )

    msg = (
        "No data was written to disk; "
        "check to make sure that the values in "
        "the column 'label' of "
        f"'{csv_path}' correspond with "
        "values in the index of your h5ad "
        "files."
    )

    with pytest.raises(RuntimeError, match=msg):
        runner.run()
