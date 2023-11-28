import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from cell_type_mapper.anndata_iterator.anndata_iterator import (
    AnnDataRowIterator)
from cell_type_mapper.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)


class AnnDataRowDataset(Dataset):
    def __init__(self, query_h5ad_path, chunk_size=1, max_gb=10, tmp_dir=None):
        self.iterator = AnnDataRowIterator(
            h5ad_path=query_h5ad_path,
            row_chunk_size=chunk_size,
            tmp_dir=tmp_dir,
            log=None,
            max_gb=max_gb,
            keep_open=False)

    def __len__(self):
        return self.iterator.n_rows

    def __getitem__(self, idx):
        return self.iterator[idx]


class Collator():
    def __init__(self,
                 all_query_identifiers,
                 normalization,
                 all_query_markers,
                 device):
        self.all_query_identifiers = all_query_identifiers
        self.normalization = normalization
        self.all_query_markers = all_query_markers
        self.device = device

    def __call__(self, batch):

        data = np.concatenate([b[0] for b in batch])

        try:
            data = torch.from_numpy(data)  # .type(torch.HalfTensor)
        except TypeError:
            alt_dtype = _choose_alternate_dtype(data.dtype)
            data = data.astype(alt_dtype)
            data = torch.from_numpy(data)

        data = data.to(device=self.device, non_blocking=True)

        r0 = batch[0][1]
        r1 = batch[-1][-1]

        data = CellByGeneMatrix(
            data=data,
            gene_identifiers=self.all_query_identifiers,
            normalization=self.normalization)

        if data.normalization != 'log2CPM':
            data.to_log2CPM_in_place()

        # downsample to just include marker genes
        # to limit memory footprint
        data.downsample_genes_in_place(self.all_query_markers)

        return data, r0, r1


def get_torch_dataloader(query_h5ad_path,
                         chunk_size,
                         all_query_identifiers,
                         normalization,
                         all_query_markers,
                         device,
                         num_workers=0,
                         max_gb=10,
                         tmp_dir=None):
    dataset = AnnDataRowDataset(query_h5ad_path,
                                chunk_size=1,
                                max_gb=max_gb,
                                tmp_dir=tmp_dir)
    collator = Collator(all_query_identifiers,
                        normalization,
                        all_query_markers,
                        device)

    sampler = BatchSampler(
            SequentialSampler(dataset),
            batch_size=chunk_size,
            drop_last=False
        )

    dataloader = DataLoader(dataset,
                            batch_size=1,  # batch size given by sampler
                            shuffle=False,
                            num_workers=num_workers,
                            sampler=sampler,
                            collate_fn=collator,
                            pin_memory=True)
    return dataloader


def _choose_alternate_dtype(original_dtype):
    """
    Choose a signed integer type to replace original_dtype, which is
    presumably an unsigned integer type which torch cannot successfully
    convert into a tensor
    """
    if not np.issubdtype(original_dtype, np.integer):
        raise RuntimeError(
            f"{original_dtype} is not an integer; unclear how to "
            "convert it to a torch compatible dtype")
    orig_iinfo = np.iinfo(original_dtype)
    for candidate in (np.int16, np.int32, np.int64):
        this_iinfo = np.iinfo(candidate)
        if this_iinfo.max >= orig_iinfo.max:
            return candidate
    raise RuntimeError(
        "Could not find a torch-compatible dtype "
        f"equivalent to {original_dtype}")
