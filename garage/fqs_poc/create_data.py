import multiprocessing
import numpy as np
import pandas as pd
import pathlib

from cell_type_mapper.utils.anndata_utils import (
    amalgamate_h5ad,
    read_df_from_h5ad)

def create_data(
        n_cells,
        output_path,
        salt,
        rng,
        chunk_size=100000,
        tmp_dir=None):

    lydia_dir = pathlib.Path(
        '/allen/programs/celltypes/workgroups/rnaseqanalysis/lydian')
    _assert_dir(lydia_dir)
    abc_dir = lydia_dir / 'ABC_handoff'
    _assert_dir(abc_dir)
    data_dir = abc_dir / 'expression_matrices/WMB-10Xv3/20230630'
    _assert_dir(data_dir)

    file_path_list = [n for n in data_dir.rglob('*raw.h5ad')]
    file_path_list.sort()
    file_arr =[]
    row_arr = []
    for ii in range(len(file_path_list)):
        obs = read_df_from_h5ad(file_path_list[ii], df_name='obs')
        n_obs = len(obs)
        file_arr += [ii]*n_obs
        row_arr += list(range(n_obs))
    file_arr = np.array(file_arr)
    row_arr = np.array(row_arr)

    print(f'{len(file_arr)} cells ({len(row_arr)})')

    chosen = rng.choice(np.arange(len(file_arr)), n_cells, replace=False)
    chosen = np.sort(chosen)
    file_arr = file_arr[chosen]
    row_arr = row_arr[chosen]

    delta = np.diff(file_arr)
    assert delta.min() >= 0
    #delta = np.diff(row_arr)
    #assert delta.min() >= 0

    obs_records = []
    for i_file in np.unique(file_arr):
        file_path = file_path_list[i_file]
        obs = read_df_from_h5ad(file_path, df_name='obs')
        these_records = obs.reset_index().to_dict(orient='records')
        valid = np.where(file_arr==i_file)[0]
        obs_records += [these_records[ir] for ir in valid]

    obs = pd.DataFrame(obs_records).set_index('cell_label')

    var = read_df_from_h5ad(
        file_path_list[file_arr[0]],
        df_name='var')

    src_rows = []
    i0 = 0
    same = None
    same_i = None
    while i0 < len(chosen):
        this_file = file_arr[i0]
        if same is None or this_file != same_i:
            same = np.where(file_arr==this_file)[0]
            same_i = this_file
        i1 = min(i0+chunk_size, same.max())
        np.testing.assert_array_equal(
            file_arr[i0:i1], this_file*np.ones(i1-i0))
        entry = {
            'path': file_path_list[file_arr[i0]],
            'layer': 'X',
            'rows': row_arr[i0:i1]
        }
        src_rows.append(entry)

    amalgamate_h5ad(
        src_rows=src_rows,
        dst_path=output_path,
        dst_obs=obs,
        dst_var=var,
        dst_sparse=True,
        tmp_dir=tmp_dir,
        compression=True)


def _assert_dir(dir_path):
    if not dir_path.is_dir():
        raise RuntimeError(
            f'{dir_path}\nis not a dir')

def main():
    rng = np.random.default_rng(22313)
    data_dir = '/allen/scratch/aibstemp/danielsf/fqs_poc'
    #ncells = [100000, 100000, 500000, 500000, 1000000, 1000000]
    ncells= [1000, 2000, 3000]
    salt_list = ['a', 'b', 'a', 'b', 'a', 'b']
    process_list = []
    chunk_size = 100
    for n, salt in zip(ncells, salt_list):
        output_path = f'{data_dir}/cells_{n//1000}k_{salt}.h5ad'
        config = {
            'n_cells': n,
            'output_path': output_path,
            'salt': salt,
            'rng': np.default_rng(rng.integers(8,2**32)),
            'tmp_dir': '/local1/scott_daniel/scratch',
            'chunk_size': chunk_size
        }
        p = multiprocessing.Process(
            target='create_data',
            kwargs=config)
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()
    print('=======ALL DONE=======')

if __name__ == "__main__":
    main()
        

