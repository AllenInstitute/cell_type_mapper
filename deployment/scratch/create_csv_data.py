import anndata
import numpy as np
import pandas as pd
import scipy.sparse

from cell_type_mapper.test_utils.anndata_utils import (
    write_anndata_x_to_csv,
    create_h5ad_without_encoding_type
)

from cell_type_mapper.utils.anndata_utils import (
    transpose_h5ad_file
)


def main():
    rng = np.random.default_rng(771231)

    small_src_path = 'csvs/small_cartoon_mouse_baseline.h5ad'

    src = get_src()
    small_src = src[:200, :]
    small_src.write_h5ad(
        small_src_path,
        compression='gzip',
        compression_opts=4)

    src.write_h5ad(
        'csvs/cartoon_mouse_baseline.h5ad',
        compression='gzip',
        compression_opts=4
    )

    write_anndata_x_to_csv(
        anndata_obj=src,
        dst_path='csvs/cartoon_mouse_csv.csv',
        cell_label_header=False,
        cell_label_type='string',
        as_int=True
    )

    # create a copy of the big dataset with degenerate cell labels
    new_obs = pd.DataFrame(
        [{'cell_id': f'c_{rng.integers(0, 15)}'}
         for ii in range(len(src.obs))]
    ).set_index('cell_id')
    write_anndata_x_to_csv(
        anndata_obj=anndata.AnnData(
            obs=new_obs,
            var=src.var,
            X=src.X
        ),
        dst_path='csvs/cartoon_mouse_degenerate_labels.csv',
        cell_label_header=False,
        cell_label_type='string',
        as_int=True
    )


    csv_config_list = [
        {'path': 'csvs/small_cartoon_mouse_csv.csv',
         'cell_label_header': False,
         'cell_label_type': 'string'},
        {'path': 'csvs/small_cartoon_mouse_gzipped.csv.gz',
         'cell_label_header': True,
         'cell_label_type': 'string'},
        {'path': 'csvs/small_cartoon_mouse_numerical.csv',
         'cell_label_header': True,
         'cell_label_type': 'numerical'},
        {'path': 'csvs/small_cartoon_mouse_big_numerical.csv',
         'cell_label_header': True,
         'cell_label_type': 'big_numerical'},
        {'path': 'csvs/small_cartoon_mouse_no_label.csv',
         'cell_label_header': False,
         'cell_label_type': None}
    ]
    for csv_config in csv_config_list:
        write_anndata_x_to_csv(
            anndata_obj=small_src,
            dst_path=csv_config['path'],
            cell_label_header=csv_config['cell_label_header'],
            cell_label_type=csv_config['cell_label_type'],
            as_int=True
        )

    # write a copy of the small cartoon file without the proper
    # encoding
    create_h5ad_without_encoding_type(
        src_path=small_src_path,
        dst_path='csvs/small_cartoon_mouse_no_encoding.h5ad'
    )

    # write the data out in CSC and CSR configuration
    small_x = small_src.X
    assert isinstance(small_x, np.ndarray)
    for tag in ('csc', 'csr'):
        if tag == 'csc':
            x = scipy.sparse.csc_matrix(small_x)
        elif tag == 'csr':
            x = scipy.sparse.csr_matrix(small_x)
        else:
            raise RuntimeError(f"what is {tag}?")
        new_obj = anndata.AnnData(
            obs=small_src.obs,
            var=small_src.var,
            X=x
        )
        path = f'csvs/small_cartoon_mouse_{tag}.h5ad'
        new_obj.write_h5ad(
            path,
            compression='gzip',
            compression_opts=4)

    # finally, write out a transposed version of the file
    transpose_h5ad_file(
        src_path=small_src_path,
        dst_path='csvs/small_cartoon_mouse_transposed.h5ad'
    )

    nan_x = np.copy(small_x)
    nan_x[15, 23] = np.nan
    with_nan = anndata.AnnData(
        obs=small_src.obs,
        var=small_src.var,
        X=nan_x
    )
    with_nan.write_h5ad(
        'csvs/corrupted_has_nan.h5ad'
    )


def get_src():
    src = anndata.read_h5ad('cartoon_mouse_baseline.h5ad', backed='r')
    dst = anndata.AnnData(
        obs=src.obs,
        var=src.var,
        X=src.X[()].toarray().astype(np.float32)
    )
    return dst


if __name__ == "__main__":
    main()
