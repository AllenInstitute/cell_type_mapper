import h5py
import numpy as np


def h5_match(obj0, obj1, this_key=None, to_skip=None):
    """
    Recursively check identity of datasets and groups in
    HDF5 handles pointed to by obj0 and obj1

    this_key is a string indicating which element you are
    currently testing.

    to_skip is an optional list of keys (like 'uns') to ignore
    """
    if this_key is not None:
        err_msg = f"Failed on {this_key}"
    else:
        err_msg = ""

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
                    rtol=1.0e-7,
                    err_msg=err_msg
                )
            else:
                np.testing.assert_array_equal(
                    d0,
                    d1,
                    err_msg)
        else:
            if not d0 == d1:
                raise RuntimeError(
                    err_msg
                )
    else:
        for k in obj0.keys():
            if k == 'metadata' or k in to_skip:
                continue
            if this_key is not None:
                new_key = f"{this_key}/{k}"
            else:
                new_key = k
            h5_match(obj0[k], obj1[k], this_key=new_key)
