from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops, degree

import numpy as np
import math


class HRGCNConv(MessagePassing):
    def __init__(self, n_inp, n_hid, device, timeframe, n_layer=1) -> None:
        super().__init__()

        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layer = n_layer
        self.device = device
        self.timeframe = timeframe

        src_ntypes = ["A", "B", "C"]
        dst_ntypes = ["A", "B", "C"]
        etypes = [f"{i}" for i in range(12)]

        # intra reltion aggregation modules
        intra_dict = {}
        for srcn in src_ntypes:
            for dstn in dst_ntypes:
                for e in etypes:
                    intra_dict[(srcn, e, dstn)] = torch.nn.Linear(
                        n_inp, n_hid, bias=True
                    ).to(device)
        self.intra_rel_agg = torch.nn.ModuleDict(intra_dict)

        self.cross_time_agg = torch.nn.ModuleDict(
            {
                ntype: TemporalAgg(n_hid, n_hid, len(timeframe), device)
                for ntype in set(src_ntypes + dst_ntypes)
            }
        )

        self.relu = torch.nn.LeakyReLU()

    def reset_parameters(self):
        pass

    def forward(self, graph, node_features):
        # TODO: Replace both intra and inter layer from HTGNN
        # Input: whole graph, node features (with different types)
        # Output: output[ntype][ttype] after HRGCN
        all_etype_t = sorted(
            list(set([etype.split("_")[-1] for _, etype, _ in graph.canonical_etypes]))
        )

        intra_features = dict({ttype: {} for ttype in self.timeframe})
        t_mapping = {v: f"t{i}" for i, v in enumerate(all_etype_t)}

        for idx, (stype, etype, dtype) in enumerate(graph.canonical_etypes):
            rel_graph = graph[stype, etype, dtype].to(self.device)
            reltype = etype.split("_")[0]
            ttype = t_mapping[etype.split("_")[-1]]

            src_node_feat = node_features[stype][ttype].to(self.device)
            dst_node_feat = node_features[dtype][ttype].to(self.device)

            intra_node_features = torch.cat([src_node_feat, dst_node_feat], dim=0)

            src_node_id, dst_node_id = rel_graph.edges("uv")
            dst_node_id = dst_node_id + graph.nodes(stype).shape[0]
            intra_edge_index = torch.stack([src_node_id, dst_node_id])

            intra_edge_index, _ = self._norm(
                intra_edge_index,
                size=intra_node_features.shape[0],
                # edge_weight=het_edge_weight,
            )

            content_h = self.intra_rel_agg[(stype, reltype, dtype)](intra_node_features)
            content_h = self.propagate(
                intra_edge_index,
                x=content_h,
                # edge_weight=het_edge_weight
            )
            content_h = self.relu(content_h)
            dst_feat = content_h[src_node_feat.shape[0] :]

            intra_features[ttype][(stype, etype, dtype)] = dst_feat.squeeze()

        # different types aggregation
        # inter_features, dict, {'ntype': {ttype: features}}
        inter_features = dict({ntype: {} for ntype in graph.ntypes})

        # TODO: return inter_features
        for ttype in intra_features.keys():
            for ntype in graph.ntypes:
                types_features = []
                for stype, etype, dtype in intra_features[ttype]:
                    if ntype == dtype:
                        types_features.append(
                            intra_features[ttype][(stype, etype, dtype)]
                        )

                types_features = torch.stack(types_features, dim=1)
                out_feat = self.inter_rel_agg[ttype](types_features)
                inter_features[ntype][ttype] = out_feat

    def _norm(self, edge_index, size, edge_weight=None, flow="source_to_target"):
        assert flow in ["source_to_target", "target_to_source"]

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)

        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_attr=edge_weight, num_nodes=size
        )

        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

        row, col = edge_index
        if flow == "source_to_target":
            deg = scatter_add(edge_weight, col, dim=0, dim_size=size)
        else:
            deg = scatter_add(edge_weight, row, dim=0, dim_size=size)

        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)

        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return edge_index, edge_weight

    def message(self, x_j, edge_weight):
        # x_j has shape [num_edges, out_channels]
        return edge_weight.view(-1, 1) * x_j

    def update(self, inputs):
        # aggr_out has shape [num_nodes, out_channels]
        return inputs


class TemporalAgg(torch.nn.Module):
    def __init__(self, n_inp: int, n_hid: int, time_window: int, device: torch.device):
        """

        :param n_inp      : int         , input dimension
        :param n_hid      : int         , hidden dimension
        :param time_window: int         , the number of timestamps
        :param device     : torch.device, gpu
        """
        super(TemporalAgg, self).__init__()

        self.proj = torch.nn.Linear(n_inp, n_hid)
        self.q_w = torch.nn.Linear(n_hid, n_hid, bias=False)
        self.k_w = torch.nn.Linear(n_hid, n_hid, bias=False)
        self.v_w = torch.nn.Linear(n_hid, n_hid, bias=False)
        self.fc = torch.nn.Linear(n_hid, n_hid)
        self.pe = (
            torch.tensor(self.generate_positional_encoding(n_hid, time_window))
            .float()
            .to(device)
        )

    def generate_positional_encoding(self, d_model, max_len):
        pe = np.zeros((max_len, d_model))
        for i in range(max_len):
            for k in range(0, d_model, 2):
                div_term = math.exp(k * -math.log(100000.0) / d_model)
                pe[i][k] = math.sin((i + 1) * div_term)
                try:
                    pe[i][k + 1] = math.cos((i + 1) * div_term)
                except:
                    continue
        return pe

    def forward(self, x):
        x = x.permute(1, 0, 2)
        h = self.proj(x)
        h = h + self.pe
        q = self.q_w(h)
        k = self.k_w(h)
        v = self.v_w(h)

        qk = torch.matmul(q, k.permute(0, 2, 1))
        score = F.softmax(qk, dim=-1)

        h_ = torch.matmul(score, v)
        h_ = F.relu(self.fc(h_))

        return h_


class RelationAgg(torch.nn.Module):
    def __init__(self, n_inp: int, n_hid: int):
        """

        :param n_inp: int, input dimension
        :param n_hid: int, hidden dimension
        """
        super(RelationAgg, self).__init__()

        self.project = torch.nn.Sequential(
            torch.nn.Linear(n_inp, n_hid),
            torch.nn.Tanh(),
            torch.nn.Linear(n_hid, 1, bias=False),
        )

    def forward(self, h):
        w = self.project(h).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((h.shape[0],) + beta.shape)

        return (beta * h).sum(1)
