import dgl
from dgl.data.utils import load_graphs

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

from model.model import HTGNN, NodePredictor
from utils.pytorchtools import EarlyStopping
from utils.data import load_dgraph_data
from utils.distributed_data import (
    DataPartitioner,
    DGraphDataset,
    partition_DGraph_dataset,
)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

dgl.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

# device = torch.device("cuda")
time_window = 7
input_root = "../HRGCN/dataset/dgl_format_1"

batch_size = 1

# torch.multiprocessing.set_sharing_strategy('file_system')


# %%
def init_process(rank, size, fn, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def read_DGraph_data(name, part):
    with open(f"{input_root}/dgraph_{name}_dgl_num_nodes_dict.json", "r") as fin:
        num_node_dict = json.loads(fin.read())

    node_labels = torch.load(f"{input_root}/dgraph_{name}_dgl_node_labels.pt")

    glist, _ = load_graphs(f"{input_root}/dgraph_{name}_dgl.bin.{part}")

    feat = load_dgraph_data(glist, time_window, num_node_dict)
    return feat, node_labels, num_node_dict


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def run(rank, size):
    print(f"[rank {rank}]: Loading Data..")
    
    train_feat, train_node_labels, train_num_node_dict = read_DGraph_data(
        "train", part=100
    )

    train_dataset = DGraphDataset(train_feat)
    train_set = partition_DGraph_dataset(
        train_dataset, num_part=size, batch_size=batch_size
    )
    graph_atom = train_dataset[0]
    device = torch.device(f"cuda:{rank}")
    
    print(f"[rank {rank}]: Define Model..")
    htgnn = HTGNN(
        graph=graph_atom,
        n_inp=16,
        n_hid=32,
        n_layers=2,
        n_heads=1,
        time_window=time_window,
        norm=False,
        device=device,
    )
    predictor = NodePredictor(n_inp=32, n_classes=1)
    model = nn.Sequential(htgnn, predictor).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    
    for G_feat in train_set:
        # G_label = train_node_labels
        print(f"[rank {rank}]: getting h..")
        h = model[0](G_feat.to(device), "vtype_2")
        print(f"[rank {rank}]: getting pred..")
        pred = model[1](h)
        print(f"[rank {rank}]: done.")
        break

#         TODO


if __name__ == "__main__":
    size = 4
    processes = []

#     train_feat, train_node_labels, train_num_node_dict = read_DGraph_data(
#         "train", part=100
#     )
#     train_dataset = DGraphDataset(train_feat)

    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
