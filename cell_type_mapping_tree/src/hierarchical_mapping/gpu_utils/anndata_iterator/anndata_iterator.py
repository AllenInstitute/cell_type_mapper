import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from hierarchical_mapping.anndata_iterator.anndata_iterator import (
    AnnDataRowIterator)
from hierarchical_mapping.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)


class AnnDataRowDataset(Dataset):
    def __init__(self, query_h5ad_path, chunk_size=1):
        self.iterator = AnnDataRowIterator(
            h5ad_path=query_h5ad_path,
            row_chunk_size=chunk_size,
            tmp_dir=None,
            log=None,
            max_gb=None,
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

        data = torch.from_numpy(data)  # .type(torch.HalfTensor)
        data = data.to(device=self.device)

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
                         num_workers=0):
    dataset = AnnDataRowDataset(query_h5ad_path, chunk_size=1)
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
