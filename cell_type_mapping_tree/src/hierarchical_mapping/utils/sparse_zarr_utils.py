import zarr

def _create_empty_zarr(
        data_shape,
        indptr_shape,
        output_path,
        data_dtype,
        chunks):

    with zarr.open(output_path, 'w') as output_zarr:
        output_zarr.create(
                    name='data',
                    shape=data_shape,
                    dtype=data_dtype,
                    chunks=chunks,
                    compressor=None)
        output_zarr.create(
                    name='indices',
                    shape=data_shape,
                    dtype=int,
                    chunks=chunks,
                    compressor=None)
        output_zarr.create(
                    name='indptr',
                    shape=indptr_shape,
                    dtype=int,
                    chunks=chunks,
                    compressor=None)
