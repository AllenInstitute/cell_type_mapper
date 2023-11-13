import h5py
import numpy as np


def _grab_indices_from_chunk(
        indptr,
        indptr_0,
        indices_chunk,
        indices_minmax,
        indices_position,
        output_dict):
    """
    Parameters
    ----------
    inpdtr:
        chunk of original indptr array that is relevant
    indptr_0:
        offset in indptr_idx (in case indptr is not the raw
        indptr)
    indices_chunk:
        the chunk of indices to be processed now
    indices_minmax:
        tuple indicating min, max of indices values being
        considered by this worker
    indices_position:
        tuple indicating (i0, i1) of indices_chunk (i.e. indices
        chunk is global_indices[i0:i1])
    output_dict:
        dict mapping value in original indices to array of original
        indptr values

    Notes
    -----
    This function will scan through the chunk of indices, grabbing
    the desired indices and populating a dict that maps indices
    to indptr (since, in the transposed array, indices<->indptr)

    output_dict[ii] will be a list of arrays indicating the
    "column" values for that index.
    """
    indices_idx = np.arange(indices_position[0], indices_position[1], dtype=int)
    valid = np.ones(indices_chunk.shape, dtype=bool)
    valid[indices_chunk<indices_minmax[0]] = False
    valid[indices_chunk>=indices_minmax[1]] = False
    indices_idx = indices_idx[valid]
    indices_chunk = indices_chunk[valid]

    indptr_idx = np.zeros(indices_idx.shape, dtype=int)
    for ii in range(len(indptr)-1):
        row = ii+indptr_0
        idx = indptr[ii]
        valid = np.logical_and(indices_idx>=idx, indices_idx<indptr[ii+1])
        indptr_idx[valid] = row

    for unq in np.unique(indices_chunk):
        valid = (indices_chunk==unq)
        this = indptr_idx[valid]
        if unq not in output_dict:
            output_dict[unq] = []
        output_dict[unq].append(this)
