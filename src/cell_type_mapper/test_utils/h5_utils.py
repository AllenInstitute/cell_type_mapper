import h5py
import numpy as np


def h5_match(obj0, obj1, this_key=None, to_skip=None):
    """
    Recursively check identity of datasets and groups in
    HDF5 handles pointed to by obj0 and obj1
    """

    if to_skip is None:
        to_skip = []

    if isinstance(obj0, h5py.Dataset):
        d0 = obj0[()]
        d1 = obj1[()]
        if isinstance(d0, np.ndarray):
            if np.issubdtype(d0.dtype, np.number):
                np.testing.assert_allclose(
                    d0,
                    d1,
                    atol=0.0,
                    rtol=1.0e-7
                )
            else:
                np.testing.assert_array_equal(d0, d1)
        else:
            if not d0 == d1:
                raise RuntimeError(
                    f"Mismatch on {this_key}"
                )
    else:
        for k in obj0.keys():
            if k == 'metadata' or k in to_skip:
                continue
            h5_match(obj0[k], obj1[k], this_key=k)
