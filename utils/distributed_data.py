from typing import Any
import math
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
import torch.multiprocessing as mp


class DGraphDataset(Dataset):
    def __init__(self, glist) -> None:
        super().__init__()
        self.glist = glist

    def __len__(self):
        return len(self.glist)

    def __getitem__(self, idx: Any) -> Any:
        return self.glist[idx]


class Partition(object):
    """Dataset partitioning helper"""

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    def __init__(self, data, num_part=4, seed=1234):
        self.data = data
        self.partitions = []
        data_len = len(data)
        part_len = math.ceil(data_len / num_part)
        indexes = [x for x in range(0, data_len)]

        for part in range(num_part):
            start_idx = part_len * part
            end_idx = start_idx + part_len
            self.partitions.append(indexes[start_idx:end_idx])

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_DGraph_dataset(glist, num_part, batch_size):
    dataset = DGraphDataset(glist)
    partition = DataPartitioner(dataset, num_part)
    partition = partition.use(dist.get_rank())
    return torch.utils.data.DataLoader(partition, batch_size=batch_size, shuffle=True)
