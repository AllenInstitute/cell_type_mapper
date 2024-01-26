import numpy as np
import pandas as pd

import anndata

import scipy.sparse

def create_data(
        n_cells,
        output_path,
        salt,
        rng):

    data_dtype = np.float32
    indices_dtype = np.int32
    n_genes = 32000
    ntot = n_cells*n_genes
    var = pd.DataFrame(
        [{'gene_id': f'g_{ii}'}
         for ii in range(n_genes)]).set_index('gene_id')

    obs = pd.DataFrame(
        [{'cell_id': f'{salt}_{ii}'}
         for ii in range(n_cells)]).set_index('cell_id')

    n_non_zero = 0
    cell_to_n = dict()
    nmin = np.round(0.05*n_genes).astype(int)
    nmax = np.round(0.35*n_genes).astype(int)
    for i_cell in range(n_cells):
        n = rng.integers(nmin, nmax)
        n_non_zero += n
        cell_to_n[i_cell] = n
    print(f'{n_non_zero:.2e} non_zero {n_non_zero/ntot}')
    data = rng.random(n_non_zero).astype(data_dtype)+0.01

    indices = np.zeros(n_non_zero, dtype=indices_dtype)
    indptr = np.zeros(n_cells+1, dtype=indices_dtype)
    i0 = 0
    for i_cell in range(n_cells):
        indptr[i_cell] = i0
        this = rng.choice(np.arange(n_genes),
                          cell_to_n[i_cell],
                          replace=False)
        this = np.sort(this)
        indices[i0:i0+len(this)] = this
        i0 += len(this)

    indptr[-1] = n_non_zero

    data = scipy.sparse.csr_matrix(
        (data, indices, indptr),
        shape=(n_cells, n_genes))

    print('created sparse matrix')
    a = anndata.AnnData(X=data, obs=obs, var=var)
    a.write_h5ad(output_path,
                 compression='gzip',
                 compression_opts=4)
    print(f'wrote {output_path}')



def main():
    rng = np.random.default_rng(22313)

    ncells = [100000, 100000, 500000, 500000, 1000000, 1000000]
    salt_list = ['a', 'b', 'a', 'b', 'a', 'b']
    for n, salt in zip(ncells, salt_list):
        output_path = f'data/cells_{n//1000}k_{salt}.h5ad'
        create_data(
            n_cells=n,
            output_path=output_path,
            salt=salt,
            rng=rng)

if __name__ == "__main__":
    main()
        

