import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, HeteroConv, aggr, GATConv, SAGEConv
from torch_geometric.data import HeteroData

class combined_processor(nn.Module):
    def __init__(self, hidden_dim, mixing: bool = True):
        super().__init__()

        self.mixing = mixing

        #step 1: x_res to alpha
        self.x_to_alpha = HeteroConv({
            ("x", "to", "alpha"): SAGEConv(hidden_dim, hidden_dim, aggr="sum"),
        })

        #step 2: alpha + y to beta
        self.to_beta = HeteroConv({
            ("alpha", "to", "beta"): SAGEConv(hidden_dim, hidden_dim, aggr="sum"),
            ("y", "to", "beta"): SAGEConv(hidden_dim, hidden_dim, aggr="sum"),
        })

        #optional dual mixing
        self.dual_to_dual = HeteroConv({
            ("alpha", "to", "beta"): SAGEConv(hidden_dim, hidden_dim, aggr="sum"),
            ("beta", "to", "alpha"): SAGEConv(hidden_dim, hidden_dim, aggr="sum")
        })

        #step 3: alpha + beta to x, beta to y
        self.out = HeteroConv({
            ("alpha", "to", "x"): SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr="sum"),
            ("beta", "to", "x"): SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr="sum"),
            ("beta", "to", "y"): SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr="sum"),
        })

        self.dual_pool = aggr.MinAggregation()

        self.norm_alpha = nn.LayerNorm(hidden_dim)
        self.norm_beta = nn.LayerNorm(hidden_dim)
        self.norm_x = nn.LayerNorm(hidden_dim)
        self.norm_y = nn.LayerNorm(hidden_dim)

    def forward(self, graph: HeteroData, masks: dict):
        x_dict = graph.x_dict
        edge_index_dict = graph.edge_index_dict

        if masks is not None:
            #mask out connections from served clients -> these are no longer contributing
            edge_index_dict[("x", "to", "alpha")] = edge_index_dict[("x", "to", "alpha")][:,masks["active_clients"]]
            edge_index_dict[("alpha", "to", "beta")] = edge_index_dict[("alpha", "to", "beta")][:, masks["active_clients"]]
            edge_index_dict[("alpha", "to", "x")] = edge_index_dict[("alpha", "to", "x")][:,masks["active_clients"]]
            edge_index_dict[("beta", "to", "alpha")] = edge_index_dict[("beta", "to", "alpha")][:, masks["active_clients"]]
            edge_index_dict[("beta", "to", "x")] = edge_index_dict[("beta", "to", "x")][:,masks["active_clients"]]
            edge_index_dict[("alpha", "to", "y")] = edge_index_dict[("alpha", "to", "y")][:,masks["active_clients"]]

        #step 1
        out1 = self.x_to_alpha(x_dict, edge_index_dict)
        x_dict["alpha"] = self.norm_alpha(x_dict["alpha"] + out1["alpha"])

        #step 2
        out2 = self.to_beta(x_dict, edge_index_dict)
        x_dict["beta"] = self.norm_beta(x_dict["beta"] + out2["beta"])

        #optional dual mixing
        if self.mixing:
            out_mix = self.dual_to_dual(x_dict, edge_index_dict)
            x_dict["alpha"] = self.norm_alpha(x_dict["alpha"] + out_mix["alpha"])
            x_dict["beta"] = self.norm_beta(x_dict["beta"] + out_mix["beta"])

        #use min aggregation for alphas and betas to get information related to the delta
        duals_combined = torch.cat([x_dict["alpha"], x_dict["beta"]], dim=0)
        batch_idx = torch.cat([graph["alpha"].batch, graph["beta"].batch], dim=0)
        delta = self.dual_pool(duals_combined, index=batch_idx)
        graph.delta = delta

        #step 3
        out3 = self.out(x_dict, edge_index_dict)
        x_dict["x"] = self.norm_x(x_dict["x"] + out3["x"])
        x_dict["y"] = self.norm_y(x_dict["y"] + out3["y"])

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
        x_dict["alpha"] = x_dict["alpha"] + out1["alpha"]
        x_dict["beta"] = x_dict["beta"] + out1["beta"]

        if self.mixing:
            out2 = self.dual_to_dual(x_dict, edge_index_dict)
            x_dict["alpha"] = x_dict["alpha"] + out2["alpha"]
            x_dict["beta"] = x_dict["beta"] + out2["beta"]

        #use min aggregation for alphas and betas to get information related to the delta
        duals_combined = torch.cat([x_dict["alpha"], x_dict["beta"]], dim=0)
        batch_idx = torch.cat([graph["alpha"].batch, graph["beta"].batch], dim=0)
        delta = self.dual_pool(duals_combined, index=batch_idx)
        graph.delta = delta

        out3 = self.out(x_dict, edge_index_dict)
        x_dict["x"] = x_dict["x"] + out3["x"]
        x_dict["y"] = x_dict["y"] + out3["y"]

        for key, val in x_dict.items():
            graph[key].x = val

        return graph

class processor(nn.Module):
    def __init__(self, hidden_dim, mixing: bool = True):
        super().__init__()
        self.processor = combined_processor(hidden_dim, mixing)

    def forward(self, graph, masks: dict = None):
        return self.processor(graph, masks)