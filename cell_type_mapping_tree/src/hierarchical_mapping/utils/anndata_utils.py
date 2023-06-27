import anndata
from anndata._io.specs import read_elem
from anndata._io.specs import write_elem
import h5py


def read_df_from_h5ad(h5ad_path, df_name):
    """
    Read the dataframe df_name (probably 'obs' or 'var')
    from the h5ad file at h5ad_path
    """
    with h5py.File(h5ad_path, 'r') as src:
        return read_elem(src[df_name])


def write_df_to_h5ad(h5ad_path, df_name, df_value):
    """
    Write the data in df_value to the element df_name in the
    specified h5ad_path
    """
    with h5py.File(h5ad_path, 'a') as dst:
        try:
            write_elem(dst, key=df_name, val=df_value)
        except TypeError:
            write_elem(dst, k=df_name, elem=df_value)


def copy_layer_to_x(
        original_h5ad_path,
        new_h5ad_path,
        layer):
    """
    Copy the data in original_h5ad_path over to new_h5ad_path.

    Copy over only obs, var and the specified layer, moving the
    layer into 'X'
    """
    layer_key = f'layers/{layer}'
    obs = read_df_from_h5ad(original_h5ad_path, 'obs')
    var = read_df_from_h5ad(original_h5ad_path, 'var')
    output = anndata.AnnData(obs=obs, var=var)
    output.write_h5ad(new_h5ad_path)
    with h5py.File(original_h5ad_path, 'r') as src:
        attrs = dict(src[layer_key].attrs)
    encoding_type = attrs['encoding-type']
    if encoding_type == 'array':
        _copy_layer_to_x_dense(
            original_h5ad_path=original_h5ad_path,
            new_h5ad_path=new_h5ad_path,
            layer=layer)
    elif 'csr' in encoding_type or 'csc' in encoding_type:
        _copy_layer_to_x_sparse(
            original_h5ad_path=original_h5ad_path,
            new_h5ad_path=new_h5ad_path,
            layer=layer)
    else:
        raise RuntimeError(
            "unclear how to copy layer with attrs "
            f"{attrs}")


def _copy_layer_to_x_dense(
        original_h5ad_path,
        new_h5ad_path,
        layer):
    layer_key = f'layers/{layer}'
    with h5py.File(original_h5ad_path) as src:
        data = src[layer_key]
        attrs = dict(src[layer_key].attrs)
        chunks = data.chunks
        with h5py.File(new_h5ad_path, 'a') as dst:
            if 'X' in dst:
                del dst['X']

            dst_dataset = dst.create_dataset(
                'X',
                shape=data.shape,
                chunks=chunks,
                dtype=data.dtype)

            for k in attrs:
                dst_dataset.attrs.create(
                    name=k,
                    data=attrs[k])

            if chunks is None:
                chunks = data.shape
            for r0 in range(0, data.shape[0], chunks[0]):
                r1 = min(data.shape[0], r0+chunks[0])
                for c0 in range(0, data.shape[1], chunks[1]):
                    c1 = min(data.shape[1], c0+chunks[1])
                    dst_dataset[r0:r1, c0:c1] = data[r0:r1, c0:c1]


def _copy_layer_to_x_sparse(
        original_h5ad_path,
        new_h5ad_path,
        layer):
    layer_key = f'layers/{layer}'
    with h5py.File(original_h5ad_path) as src:
        src_grp = src[layer_key]
        attrs = dict(src_grp.attrs)
        with h5py.File(new_h5ad_path, 'a') as dst:
            if 'X' in dst:
                del dst['X']

            dst_grp = dst.create_group('X')

            for k in attrs:
                dst_grp.attrs.create(
                    name=k,
                    data=attrs[k])

            for el in ('indptr', 'indices', 'data'):
                src_dataset = src_grp[el]
                dtype = src_dataset.dtype
                chunks = src_dataset.chunks
                dst_grp.create_dataset(
                    el,
                    shape=src_dataset.shape,
                    chunks=chunks,
                    dtype=dtype)
                if chunks is None:
                    dst_grp[el] = src_dataset[()]
                else:
                    for i0 in range(0, src_dataset.shape[0], chunks[0]):
                        i1 = min(src_dataset.shape[0], i0+chunks[0])
                        dst_grp[el][i0:i1] = src_dataset[i0:i1]
