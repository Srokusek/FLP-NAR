import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, HeteroConv, aggr, GATConv, SAGEConv
from torch_geometric.data import HeteroData

class combined_processor(nn.Module):
    def __init__(self, hidden_dim, mixing: bool = True):
        super().__init__()

        self.mixing = mixing
        aggregation = aggr.MultiAggregation(["min", "max"])

        #step 1: x_res to alpha
        self.x_to_alpha = HeteroConv({
            ("x", "to", "alpha"): SAGEConv(hidden_dim, hidden_dim, aggr=aggr.MultiAggregation(["min", "sum"])),
        })

        #step 2: alpha + y to beta
        self.to_beta = HeteroConv({
            ("alpha", "to", "beta"): SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
            ("y", "to", "beta"): SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
        })

        #optional dual mixing
        self.dual_to_dual = HeteroConv({
            ("alpha", "to", "beta"): SAGEConv(hidden_dim, hidden_dim, aggr=aggregation),
            ("beta", "to", "alpha"): SAGEConv(hidden_dim, hidden_dim, aggr=aggregation)
        })

        #step 3: alpha + beta to x, beta to y
        self.out = HeteroConv({
            ("alpha", "to", "x"): SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr="mean"),
            ("beta", "to", "x"): SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr=aggr.MultiAggregation(["min", "sum"])),
            ("beta", "to", "y"): SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr=aggr.MultiAggregation(["min", "sum"])),
        })

        self.dual_pool = aggr.MinAggregation()

        self.norm_alpha = nn.LayerNorm(hidden_dim)
        self.norm_beta = nn.LayerNorm(hidden_dim)
        self.norm_x = nn.LayerNorm(hidden_dim)
        self.norm_y = nn.LayerNorm(hidden_dim)

        self.gate_alpha = nn.Sequential(
            # nn.Linear(hidden_dim * 2, hidden_dim * 2),
            # nn.Sigmoid(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.gate_beta = nn.Sequential(
            # nn.Linear(hidden_dim * 2, hidden_dim * 2),
            # nn.Sigmoid(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.gate_x = nn.Sequential(
            # nn.Linear(hidden_dim * 2, hidden_dim * 2),
            # nn.Sigmoid(),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid()
        )

        self.gate_y = nn.Sequential(
            # nn.Linear(hidden_dim * 2, hidden_dim * 2),
            # nn.Sigmoid(),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid()
        )

        def dummy(_in):
            return _in

        self.post_alpha = dummy
        self.post_beta = dummy
        self.post_y = dummy
        self.post_x = dummy

        # self.post_alpha = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.PReLU()
        # )

        # self.post_beta = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.PReLU()
        # )

        # self.post_x = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.PReLU()
        # )

        # self.post_y = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.PReLU()
        # )

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
        msg_alpha = self.post_alpha(out1["alpha"])
        gate_alpha = self.gate_alpha(torch.cat([x_dict["alpha"], msg_alpha], dim=-1))
        x_dict["alpha"] = self.norm_alpha(x_dict["alpha"] + gate_alpha * msg_alpha)

        #step 2
        out2 = self.to_beta(x_dict, edge_index_dict)
        msg_beta = self.post_beta(out2["beta"])
        gate_beta = self.gate_beta(torch.cat([x_dict["beta"], msg_beta], dim=-1))
        x_dict["beta"] = self.norm_beta(x_dict["beta"] + gate_beta * msg_beta)

        #optional dual mixing
        if self.mixing:
            out_mix = self.dual_to_dual(x_dict, edge_index_dict)
            msg_alpha = out_mix["alpha"]
            gate_alpha = self.gate_alpha(torch.cat([x_dict["alpha"], msg_alpha], dim=-1))
            x_dict["alpha"] = self.norm_alpha(x_dict["alpha"] + gate_alpha * msg_alpha)

            msg_beta = out_mix["beta"]
            gate_beta = self.gate_beta(torch.cat([x_dict["beta"], msg_beta], dim=-1))
            x_dict["beta"] = self.norm_beta(x_dict["beta"] + gate_beta * msg_beta)

        #use min aggregation for alphas and betas to get information related to the delta
        duals_combined = torch.cat([x_dict["alpha"], x_dict["beta"]], dim=0)
        batch_idx = torch.cat([graph["alpha"].batch, graph["beta"].batch], dim=0)
        delta = self.dual_pool(duals_combined, index=batch_idx)
        graph.delta = delta

        #step 3
        out3 = self.out(x_dict, edge_index_dict)

        msg_x = self.post_x(out3["x"])
        gate_x = self.gate_x(torch.cat([x_dict["x"], msg_x, delta.repeat_interleave(msg_x.shape[0] // delta.shape[0], dim=0)], dim=-1))
        x_dict["x"] = self.norm_x(x_dict["x"] + gate_x * msg_x)

        msg_y = self.post_y(out3["y"])
        gate_y = self.gate_y(torch.cat([x_dict["y"], msg_y, delta.repeat_interleave(msg_y.shape[0] // delta.shape[0], dim=0)], dim=-1))
        x_dict["y"] = self.norm_y(x_dict["y"] + gate_y * msg_y)

        for key, val in x_dict.items():
            graph[key].x = val

        return graph

class simple_processor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        #step 1: x_res to y_res
        self.x_to_y = HeteroConv({
            ("x", "to", "y"): SAGEConv(hidden_dim, hidden_dim, aggr=aggr.MultiAggregation(["min", "sum"])),
        })

        #step 2: y_res to x_res
        self.y_to_x = HeteroConv({
            ("y", "to", "x"): SAGEConv(hidden_dim, hidden_dim, aggr="sum") #aggregation is not as impartant since the connection is always one-to-one
        })

        #delta pooling
        self.delta_pool = aggr.MinAggregation

        self.gate_x = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.gate_y = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, graph: HeteroData, masks: dict):
        x_dict = graph.x_dict
        edge_index_dict = graph.edge_index_dict

        

class processor(nn.Module):
    def __init__(self, hidden_dim, mixing: bool = True):
        super().__init__()
        self.processor = combined_processor(hidden_dim, mixing)

    def forward(self, graph, masks: dict = None):
        return self.processor(graph, masks)