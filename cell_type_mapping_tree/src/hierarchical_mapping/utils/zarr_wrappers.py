import zarr
import numpy as np
import json

from hierarchical_mapping.utils.utils import (
    merge_index_list)


class ZarrCacheWrapper(object):

    def __init__(self, zarr_path):
        self._zarr_handle = zarr.open(zarr_path, 'r')
        self._loaded_chunk = None
        self._loaded_chunk_bounds = (-1, -1)
        self._metadata = json.load(open(f'{zarr_path}/.zarray', 'rb'))
        if len(self.shape) > 1:
            raise NotImplementedError(
                "ZarrCacheWrapper does not yet support "
                "N-D arrays; this is {len(self.shape)}-D")

    def __getitem__(self, key):
        if isinstance(key, int):
            key_list = [key]
        elif isinstance(key, slice):
            if key.step is not None and key.step != 1:
                raise RuntimeError("slice must have stride 1 or None")
            key_list = list(range(key.start, key.stop, 1))
        else:
            raise RuntimeError(
                "key must be int or slice; this is "
                f"{type(key)}")

        return self._get_items_from_list(key_list)

    @property
    def dtype(self):
        return self.zarr_handle.dtype

    @property
    def chunks(self):
        return self._metadata['chunks']

    @property
    def shape(self):
        return self._metadata['shape']

    @property
    def zarr_handle(self):
        return self._zarr_handle

    def _load_chunk(self, index_bounds):
        """
        index_bounds is a tuple specifying
        (i_min, i_max) of the chunk to be loaded
        from self.zarr_handle
        """
        self._loaded_chunk = self.zarr_handle[
                                index_bounds[0][0]:index_bounds[0][1]]
        self._loaded_chunk_bounds = index_bounds

    def _get_items_from_list(self, index_list):
        """
        index_list is a list of integers denoting the elements
        to load from the array
        """
        (chunk_mapping) = separate_into_chunks(
                                    index_list,
                                    self.chunks)

        results = np.zeros(len(index_list), dtype=self.dtype)

        for chunk in chunk_mapping:
            if self._loaded_chunk_bounds != chunk['chunk_spec']:
                self._load_chunk(chunk['chunk_spec'])
            output_loc = chunk['output_loc']
            chunk_loc = chunk['chunk_loc']
            results[
                output_loc[0]:
                output_loc[1]] = self._loaded_chunk[
                                      chunk_loc[0]:
                                      chunk_loc[1]]

        return results
        

# test this
def separate_into_chunks(
        index_list,
        chunk_shape):
    """
    Take index_list;

    Return a dict mapping chunk specification to
    (chunk_slice, output_slice)

    ** assumes index_list is contiguous ***
    """
    index_list = np.array(index_list)
    if len(index_list) > 1:
        d = np.diff(index_list)
        if d.max() > 1:
            raise RuntimeError("Index list is not contiguous")
        if d.min() < 0:
            raise RuntimeError("Index list is not ascending")

    assigned = np.zeros(len(index_list), dtype=bool)

    chunk_lookup = []
    while assigned.sum() < len(index_list):
        min_key = index_list[np.logical_not(assigned)].min()
        min_chunk = chunk_shape[0] * (min_key // chunk_shape[0])
        chunk_spec = ((min_chunk, min_chunk+chunk_shape[0]),)
        in_this_chunk = np.logical_and(
                            index_list>=chunk_spec[0][0],
                            index_list<chunk_spec[0][1])

        in_this_chunk = np.where(in_this_chunk)[0]

        output_loc = (in_this_chunk.min(), in_this_chunk.max()+1)
        chunk_loc = (index_list[output_loc[0]]-chunk_spec[0][0],
                     index_list[output_loc[1]-1]-chunk_spec[0][0]+1)



        chunk_lookup.append({'chunk_spec': chunk_spec,
                             'chunk_loc': chunk_loc,
                             'output_loc': output_loc})

        assigned[in_this_chunk] = True

    return chunk_lookup
