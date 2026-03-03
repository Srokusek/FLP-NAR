import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, HeteroConv, aggr, GATConv, SAGEConv
from torch_geometric.data import HeteroData



class primal_dual_primal(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        #aggregation from primal to dual variables
        self.primal_to_dual = HeteroConv({
            ("x", "to", "alpha"): SAGEConv(hidden_dim, hidden_dim),
            ("y", "to", "beta"): SAGEConv(hidden_dim, hidden_dim),
        }, aggr="sum")

        self.dual_to_primal = HeteroConv({
            ("alpha", "to", "x"): SAGEConv(hidden_dim, hidden_dim),
            ("beta", "to", "y"): SAGEConv(hidden_dim, hidden_dim),
        }, aggr="sum")

        self.dual_to_dual = HeteroConv({
            ("alpha", "to", "beta"): GATConv(hidden_dim, hidden_dim, edge_dim=hidden_dim, add_self_loops=False), #using GATConv for now as that enables edge_attributes
            ("beta", "to", "alpha"): GATConv(hidden_dim, hidden_dim, edge_dim=hidden_dim, add_self_loops=False),
        }, aggr="sum")

        self.dual_pool = aggr.MinAggregation()

    def forward(self, graph: HeteroData):
        x_dict = graph.x_dict

        #pass information from primal to dual
        out1 = self.primal_to_dual(x_dict, graph.edge_index_dict)
        x_dict["alpha"] = out1["alpha"]
        x_dict["beta"] = out1["beta"]

        #pass information between duals
        out2 = self.dual_to_dual(x_dict, graph.edge_index_dict, graph.edge_attr_dict)
        x_dict["alpha"] = out2["alpha"]
        x_dict["beta"] = out2["beta"]

        #pass information from duals to primal
        out3 = self.dual_to_primal(x_dict, graph.edge_index_dict)
        x_dict["x"] = out3["x"]
        x_dict["y"] = out3["y"]

        #use min aggregation for alphas and betas to get information related to the delta
        duals_combined = torch.cat([x_dict["alpha"], x_dict["beta"]], dim=0)
        batch_idx = torch.cat([graph["alpha"].batch, graph["beta"].batch], dim=0)
        delta = self.dual_pool(duals_combined, index=batch_idx)
        graph.delta = delta

        for key, val in x_dict.items():
            graph[key].x = val

        return graph

class primal_dual_dual_primal(nn.Module):
    def __init__(self, hidden_dim, mixing: bool = True):
        super().__init__()

        self.mixing = mixing

        #x,y residuals to alpha,beta prediction
        self.primal_to_dual = HeteroConv({
            ("x", "to", "alpha"): SAGEConv(hidden_dim, hidden_dim, aggr="sum"),
            ("y", "to", "beta"): SAGEConv(hidden_dim, hidden_dim, aggr="sum"),
        })

        #dual mixing
        self.dual_to_dual = HeteroConv({
            ("alpha", "to", "beta"): SAGEConv(hidden_dim, hidden_dim, aggr="sum"),
            ("beta", "to", "alpha"): SAGEConv(hidden_dim, hidden_dim, aggr="sum")
        })

        #dual+primal to out
        self.out = HeteroConv({
            ("alpha", "to", "x"): SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr="sum"),
            ("beta", "to", "y"): SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr="sum"),
        })

        self.dual_pool = aggr.MinAggregation()

    def forward(self, graph: HeteroData, masks: dict):
        x_dict = graph.x_dict
        edge_index_dict = graph.edge_index_dict

        if masks is not None:
            #mask out connections from served clients -> these are no longer contributing
            edge_index_dict[("x", "to", "alpha")] = edge_index_dict[("x", "to", "alpha")][:,masks["active_clients"]]
            edge_index_dict[("alpha", "to", "beta")] = edge_index_dict[("alpha", "to", "beta")][:, masks["active_clients"]]
            #also reverse connections
            edge_index_dict[("alpha", "to", "x")] = edge_index_dict[("alpha", "to", "x")][:,masks["active_clients"]]
            edge_index_dict[("beta", "to", "alpha")] = edge_index_dict[("beta", "to", "alpha")][:, masks["active_clients"]]

        out1 = self.primal_to_dual(x_dict, edge_index_dict)
        x_dict["alpha"] = out1["alpha"]
        x_dict["beta"] = out1["beta"]

        if self.mixing:
            out2 = self.dual_to_dual(x_dict, edge_index_dict)
            x_dict["alpha"] = out2["alpha"]
            x_dict["beta"] = out2["beta"]

        #use min aggregation for alphas and betas to get information related to the delta
        duals_combined = torch.cat([x_dict["alpha"], x_dict["beta"]], dim=0)
        batch_idx = torch.cat([graph["alpha"].batch, graph["beta"].batch], dim=0)
        delta = self.dual_pool(duals_combined, index=batch_idx)
        graph.delta = delta

        out3 = self.out(x_dict, edge_index_dict)
        x_dict["x"] = out3["x"]
        x_dict["y"] = out3["y"]

        for key, val in x_dict.items():
            graph[key].x = val

        return graph

class processor(nn.Module):
    def __init__(self, hidden_dim, res_only: bool = False, mixing: bool = True):
        super().__init__()
        self.res_only = res_only
        if res_only:
            self.processor = primal_dual_dual_primal(hidden_dim, mixing)
        else:
            self.processor = primal_dual_primal(hidden_dim)

    def forward(self, graph, masks: dict = None):
        return self.processor(graph, masks) if self.res_only else self.processor(graph)