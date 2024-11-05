import h5py
import numpy as np

def h5_match(obj0, obj1):
    """
    Recursively check identity of datasets and groups in
    HDF5 handles pointed to by obj0 and obj1
    """
    if isinstance(obj0, h5py.Dataset):
        d0 = obj0[()]
        d1 = obj1[()]
        if isinstance(d0, np.ndarray):
            np.testing.assert_allclose(
                d0,
                d1,
                atol=0.0,
                rtol=1.0e-7
            )
        else:
            assert d0 == d1
    else:
        for k in obj0.keys():
            if k == 'metadata':
                continue
            h5_match(obj0[k], obj1[k])
