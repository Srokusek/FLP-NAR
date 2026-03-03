import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter

from .processors import processor

class Reasoner(nn.Module):
    def __init__(self, hidden_dim, tf_prob: float = 0.5, res_only: bool = False):
        super().__init__()
        self.tf_prob = tf_prob
        self.res_only = res_only
        self.eps = 1e-8

        #encoders
        if not self.res_only:
            self.alpha_encoder = nn.Sequential(
                nn.Linear(1, hidden_dim, bias=True),
                nn.ReLU()
            )

            self.beta_encoder = nn.Sequential(
                nn.Linear(1, hidden_dim, bias=True),
                nn.ReLU()
            )

            self.dist_encoder = nn.Sequential(
                nn.Linear(1, hidden_dim, bias=True),
                nn.ReLU()
            )

            self.x_encoder = nn.Sequential(
                nn.Linear(2, hidden_dim, bias=True), #input: [binary state | static demand]
                nn.ReLU()
            )

            self.y_encoder = nn.Sequential(
                nn.Linear(2, hidden_dim, bias=True), #input: [binary state | static facility costs]
                nn.ReLU()
            )

        if self.res_only:
            self.x_res_encoder = nn.Sequential(
                nn.Linear(1, hidden_dim, bias=True),
                nn.ReLU()
            )

            self.y_res_encoder = nn.Sequential(
                nn.Linear(1, hidden_dim, bias=True),
                nn.ReLU()
            )

            self.alpha_init = nn.Parameter(torch.randn(1,hidden_dim) * 0.01)
            self.beta_init = nn.Parameter(torch.randn(1,hidden_dim) * 0.01)

        #processor
        self.processor = processor(hidden_dim, self.res_only)

        #decoders
        self.x_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True) #keep logits for BCE loss
        )

        self.y_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True) #keep logits for BCE loss
        )

        self.alpha_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
            nn.Softplus()
        )

        self.beta_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
            nn.Softplus()
        )

        self.delta_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
            nn.Softplus()
        )

        self.x_res_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
            nn.Softplus()
        )

        self.y_res_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
            nn.Softplus()
        )

    def _encode_static(self, batch: HeteroData):
        #encode the time invariant parameters
        h_dist = self.dist_encoder(batch["alpha", "to", "beta"].edge_attr)
        batch["alpha", "to", "beta"].edge_attr = h_dist
        batch["beta", "to", "alpha"].edge_attr = h_dist
        return batch
    
    def _encode_node_features(self, alpha_in, beta_in, x_in, y_in, demands, f_costs):
        #encode the node features for all node types
        h_alpha = self.alpha_encoder(alpha_in)
        h_beta = self.beta_encoder(beta_in)
        h_x = self.x_encoder(torch.cat([x_in, demands], dim=-1))
        h_y = self.y_encoder(torch.cat([y_in, f_costs], dim=-1))
        return h_alpha, h_beta, h_x, h_y
    
    def _encode_residual_features(self, x_res_in, y_res_in):
        h_x_res = self.x_res_encoder(x_res_in)
        h_y_res = self.y_res_encoder(y_res_in)
        return h_x_res, h_y_res
    
    def _set_residual_features(self, batch, h_x, h_y):
        batch["x"].x = h_x
        batch["y"].x = h_y
        batch["alpha"].x = self.alpha_init.expand(batch["alpha"].x.shape[0], -1)
        batch["beta"].x = self.beta_init.expand(batch["beta"].x.shape[0], -1)
        return batch
        
    def _set_node_features(self, batch, h_alpha, h_beta, h_x, h_y):
        #update the features in the HeteroData object
        batch["alpha"].x = h_alpha
        batch["beta"].x = h_beta
        batch["x"].x = h_x
        batch["y"].x = h_y
        return batch
    
    def _decode_step(self, batch):
        #decode node features into algorithm step predictions
        return {
            "alpha": self.alpha_decoder(batch["alpha"].x), 
            "beta": self.beta_decoder(batch["beta"].x),
            "x": self.x_decoder(batch["x"].x),
            "y": self.y_decoder(batch["y"].x),
            "delta": self.delta_decoder(batch.delta)
        }
    
    def _decode_residual_step(self, batch):
        return {
            "alpha": self.alpha_decoder(batch["alpha"].x),
            "beta": self.beta_decoder(batch["beta"].x),
            "x": self.x_decoder(batch["x"].x),
            "y": self.y_decoder(batch["y"].x),
            "x_res": self.x_res_decoder(batch["x"].x),
            "y_res": self.y_res_decoder(batch["y"].x),
            "delta": self.delta_decoder(batch.delta),
        }

    @staticmethod
    def _mse(pred, target, mask=None):
        #ensure shapes match: squeeze pred if [N,1] vs target [N]
        if pred.dim() > target.dim():
            pred = pred.squeeze(-1)
        elif target.dim() > pred.dim():
            target = target.squeeze(-1)
        if mask is not None and mask.sum() > 0:
            return nn.functional.mse_loss(pred[mask], target[mask])
        return nn.functional.mse_loss(pred, target)
    
    @staticmethod
    def _bce(pred, target, mask=None):
        #ensure shapes match: squeeze if needed
        if pred.dim() > target.dim():
            pred = pred.squeeze(-1)
        elif target.dim() > pred.dim():
            target = target.squeeze(-1)
        if mask is not None and mask.sum() > 0:
            return nn.functional.binary_cross_entropy_with_logits(pred[mask], target[mask])    
        return nn.functional.binary_cross_entropy_with_logits(pred, target)

    def convert_to_solution(self, batch: HeteroData, x_pred: torch.Tensor, y_pred: torch.Tensor):
        ##Convert predictions into total cost per sample.
        #Assumes all samples in the batch have the same n_cli and n_fac.
        y_prob = torch.sigmoid(y_pred).detach()  # [total_fac, 1]
        y_binary = (y_prob > 0.5).float()

        f_costs = batch["y"].f_costs  #[total_fac, 1]
        demands = batch["x"].demands  #[total_x, 1]
        dist = batch["alpha", "to", "beta"].edge_attr if not self.res_only else batch["x"].dist
        dist_w = dist * demands  #[total_x, 1] where total_x = B * n_cli * n_fac

        has_batch = hasattr(batch["y"], "batch")

        if has_batch:
            n_samples = batch["y"].batch.max().item() + 1
            n_fac = (batch["y"].batch == 0).sum().item()
            n_cli = (batch["alpha"].batch == 0).sum().item()
        else:
            n_samples = 1
            n_fac = batch["y"].x.shape[0]
            n_cli = batch["alpha"].x.shape[0]

        #reshape to [B, n_fac] and [B, n_cli, n_fac]
        y_binary_per_sample = y_binary.squeeze(-1).view(n_samples, n_fac)  #[B, n_fac]
        f_costs_per_sample = f_costs.squeeze(-1).view(n_samples, n_fac)  #[B, n_fac]
        dist_w_per_sample = dist_w.squeeze(-1).view(n_samples, n_cli, n_fac)  #[B, n_cli, n_fac]

        #facility cost
        facility_cost = (y_binary_per_sample * f_costs_per_sample).sum(dim=1)  #[B]

        #client assignment: for each client, find min distance to an OPEN facility
        #mask closed facilities with inf
        open_mask = y_binary_per_sample.unsqueeze(1)  #[B, 1, n_fac]
        masked_dist = dist_w_per_sample.clone()
        masked_dist[open_mask.expand_as(masked_dist) == 0] = float("inf")

        #min distance per client
        min_dist_per_client = masked_dist.min(dim=2).values  # [B, n_cli]

        #handle case where no facility is open - use max distance as penalty
        max_dist_per_client = dist_w_per_sample.max(dim=2).values  # [B, n_cli]
        min_dist_per_client = torch.where(
            min_dist_per_client.isinf(),
            max_dist_per_client,
            min_dist_per_client
        )

        client_cost = min_dist_per_client.sum(dim=1)  # [B]

        total_cost = facility_cost + client_cost
        return total_cost.unsqueeze(-1)  # [B, 1]

    def teacher_forcing(self, batch: HeteroData) -> dict:
        ###do T-1 steps with fuzzy teacher input

        #support batched data with variable time steps
        T = batch["alpha"].x.shape[1]  #max_T after padding
        n_cli = batch["alpha"].x.shape[0]  #total clients across all samples in batch
        n_fac = batch["y"].x.shape[0]  #total facilities across all samples in batch
        
        #get per-sample time lengths for masking
        t_lengths = getattr(batch, 't_lengths', None)
        has_variable_T = t_lengths is not None and len(t_lengths) > 1

        if not self.res_only:
            demands = batch["x"].demands
            f_costs = batch["y"].f_costs
            dist_ab = batch["alpha", "to", "beta"].edge_attr
            dist_ba = batch["beta", "to", "alpha"].edge_attr

            #store the ground truth traces
            alpha_trace = batch["alpha"].x
            beta_trace = batch["beta"].x
            x_trace = batch["x"].x.float()
            y_trace = batch["y"].x.float()
            delta_trace = batch.delta
        else:
            x_res_trace = batch["x"].x
            y_res_trace = batch["y"].x
            x_trace = batch["x"].trace_sol.float()
            y_trace = batch["y"].trace_sol.float()
            alpha_trace = batch["alpha"].x
            beta_trace = batch["beta"].x
            delta_trace = batch.delta

        #encode the static features once
        batch = self._encode_static(batch) if not self.res_only else batch

        #initialize "previous step" -> the inputs that will be given to the NAR system
        prev_alpha = alpha_trace[:, 0, :]
        prev_beta = beta_trace[:, 0, :]
        prev_x = x_trace[:, 0, :]
        prev_y = y_trace[:, 0, :]
        if self.res_only:
            prev_x_res = x_res_trace[:, 0, :]
            prev_y_res = y_res_trace[:, 0, :]

        all_alpha_loss = []
        all_beta_loss = []
        all_x_loss = []
        all_y_loss = []
        all_delta_loss = []
        optimum_loss = 0.0
        if self.res_only:
            all_x_res_loss = []
            all_y_res_loss = []

        #init masks for tight facilities/clients
        masks = None
        if self.res_only:
            masks = {
                "active_clients": torch.ones(prev_x_res.shape[0], dtype=torch.bool, device=prev_x_res.device)
            }
        
        #pre-compute batch indices and offsets for frozen client masking (if batched)
        if self.res_only and has_variable_T:
            x_batch_idx = batch["x"].batch
            y_batch_idx = batch["y"].batch
            alpha_batch_idx = batch["alpha"].batch
            n_samples = x_batch_idx.max().item() + 1
            n_fac_per_sample = (y_batch_idx == 0).sum().item()
            n_cli_per_sample = (alpha_batch_idx == 0).sum().item()
            
            #compute local indices and mappings once
            x_local_idx = torch.zeros_like(x_batch_idx)
            y_offsets = torch.zeros(n_samples, dtype=torch.long, device=x_batch_idx.device)
            alpha_offsets = torch.zeros(n_samples, dtype=torch.long, device=x_batch_idx.device)
            for b in range(n_samples):
                x_mask = (x_batch_idx == b)
                x_local_idx[x_mask] = torch.arange(x_mask.sum(), device=x_batch_idx.device)
                if b > 0:
                    y_offsets[b] = y_offsets[b-1] + (y_batch_idx == b-1).sum()
                    alpha_offsets[b] = alpha_offsets[b-1] + (alpha_batch_idx == b-1).sum()
            
            fac_idx_per_x = x_local_idx % n_fac_per_sample
            cli_idx_per_x = x_local_idx // n_fac_per_sample
            y_global_idx = y_offsets[x_batch_idx] + fac_idx_per_x
            alpha_global_idx = alpha_offsets[x_batch_idx] + cli_idx_per_x

        for t in range(1,T):
            #fuzzy teacher mixing
            if self.training and self.tf_prob < 1.0:
                def _mix(ground_truth, pred):
                    mask = (torch.rand_like(ground_truth) < self.tf_prob)
                    return torch.where(mask, ground_truth, pred) #return combination of ground_truth and predictions based on the selected probability

                alpha_in = _mix(alpha_trace[:, t-1,:], #groud truth
                                prev_alpha #previous prediction
                                )
                beta_in = _mix(beta_trace[:, t-1, :], prev_beta)
                x_in = _mix(x_trace[:, t-1, :], prev_x)
                y_in = _mix(y_trace[:, t-1, :], prev_y)
                x_loss_mask = None
                y_loss_mask = None
                if self.res_only:
                    x_res_in = _mix(x_res_trace[:, t-1, :], prev_x_res)
                    y_res_in = _mix(y_res_trace[:, t-1, :], prev_y_res)
                    #frozen client masking
                    if has_variable_T:
                        y_res_for_x = y_res_in.squeeze(-1)[y_global_idx]  #[total_x]
                        x_tight = (x_res_in.squeeze(-1) < self.eps) & (y_res_for_x < self.eps)  #= client variable is tight and facility is tentatively open
                        #for each client, check if ANY connection is tight
                        client_has_tight = scatter(x_tight.float(), alpha_global_idx, dim=0, reduce='max') > 0  #[total_alpha]
                        masks["active_clients"] = ~client_has_tight[alpha_global_idx]
                    else:
                        tight_connections = ((x_res_in < self.eps) * (y_res_in < self.eps).repeat(n_cli, 1)).view(n_cli, n_fac)
                        frozen_clients = torch.sum(tight_connections, dim=1) > 0
                        masks["active_clients"] = ~frozen_clients.repeat_interleave(n_fac)

            elif self.tf_prob == 1.0: #use only ground truth, used when tf_prob >= 1
                #when not trying to use any labels at all, use forward method instead
                alpha_in = alpha_trace[:, t-1, :]
                beta_in = beta_trace[:, t-1, :]
                x_in = x_trace[:, t-1, :]
                y_in = y_trace[:, t-1, :]
                x_loss_mask = x_in.bool()
                y_loss_mask = y_in.bool()
                if self.res_only:
                    x_res_in = x_res_trace[:, t-1, :]
                    y_res_in = y_res_trace[:, t-1, :]
                    x_loss_mask = None 
                    y_loss_mask = None
                    #frozen client masking
                    if has_variable_T:
                        y_res_for_x = y_res_in.squeeze(-1)[y_global_idx]
                        x_tight = (x_res_in.squeeze(-1) < self.eps) & (y_res_for_x < self.eps)
                        client_has_tight = scatter(x_tight.float(), alpha_global_idx, dim=0, reduce='max') > 0
                        masks["active_clients"] = ~client_has_tight[alpha_global_idx]
                    else:
                        tight_connections = ((x_res_in < self.eps) * (y_res_in < self.eps).repeat(n_cli, 1)).view(n_cli, n_fac)
                        frozen_clients = torch.sum(tight_connections, dim=1) > 0
                        masks["active_clients"] = ~frozen_clients.repeat_interleave(n_fac)
            
            elif not self.training and self.tf_prob < 1.0:
                #complete trace independent eval/test
                alpha_in = prev_alpha
                beta_in = prev_beta
                x_in = prev_x
                y_in = prev_y
                x_loss_mask = x_in.bool()
                y_loss_mask = y_in.bool()
                if self.res_only:
                    x_res_in = prev_x_res
                    y_res_in = prev_y_res
                    x_loss_mask = None 
                    y_loss_mask = None
                    #frozen client masking
                    if has_variable_T:
                        y_res_for_x = y_res_in.squeeze(-1)[y_global_idx]
                        x_tight = (x_res_in.squeeze(-1) < self.eps) & (y_res_for_x < self.eps)
                        client_has_tight = scatter(x_tight.float(), alpha_global_idx, dim=0, reduce='max') > 0
                        masks["active_clients"] = ~client_has_tight[alpha_global_idx]
                    else:
                        tight_connections = ((x_res_in < self.eps) * (y_res_in < self.eps).repeat(n_cli, 1)).view(n_cli, n_fac)
                        frozen_clients = torch.sum(tight_connections, dim=1) > 0
                        masks["active_clients"] = ~frozen_clients.repeat_interleave(n_fac)

            #encode the node features
            if self.res_only:
                h_x_res, h_y_res = self._encode_residual_features(x_res_in, y_res_in)
                batch = self._set_residual_features(batch, h_x_res, h_y_res)
            else:
                h_alpha, h_beta, h_x, h_y = self._encode_node_features(alpha_in, beta_in, x_in, y_in, demands, f_costs)
                batch = self._set_node_features(batch, h_alpha, h_beta, h_x, h_y)

            #do the processing step
            batch = self.processor(batch, masks) if self.res_only else self.processor(batch)

            #decode into algorithm steps
            preds = self._decode_step(batch) if not self.res_only else self._decode_residual_step(batch)

            #save losses with time-step masking for batched variable-T data
            if has_variable_T:
                #create masks for nodes belonging to samples with valid timestep t
                alpha_batch_idx = batch["alpha"].batch
                beta_batch_idx = batch["beta"].batch
                x_batch_idx = batch["x"].batch
                y_batch_idx = batch["y"].batch
                
                #valid_samples: t < t_length
                valid_samples = (t_lengths > t)  #c[B]
                
                #create node-level masks
                alpha_mask = valid_samples[alpha_batch_idx] #[total_alpha_nodes]
                beta_mask = valid_samples[beta_batch_idx] #[total_beta_nodes]
                x_mask = valid_samples[x_batch_idx] #[total_x_nodes]
                y_mask = valid_samples[y_batch_idx] #[total_y_nodes]
                
                #compute masked losses (only on nodes from samples with valid t)
                if alpha_mask.sum() > 0:
                    all_alpha_loss.append(self._mse(preds["alpha"][alpha_mask], alpha_trace[:, t, :][alpha_mask]))
                    all_beta_loss.append(self._mse(preds["beta"][beta_mask], beta_trace[:, t, :][beta_mask]))
                
                #delta: delta_trace is [max_T, B, 1] after batching with colalte_fn
                delta_target = delta_trace[t]  #[B, 1]
                delta_valid_mask = valid_samples
                if delta_valid_mask.sum() > 0:
                    all_delta_loss.append(self._mse(preds["delta"][delta_valid_mask], delta_target[delta_valid_mask]))
                
                if x_mask.sum() > 0:
                    x_m = x_mask if x_loss_mask is None else (x_mask & x_loss_mask.squeeze(-1))
                    all_x_loss.append(self._bce(preds["x"][x_m], x_trace[:, t, :][x_m]))
                    y_m = y_mask if y_loss_mask is None else (y_mask & y_loss_mask.squeeze(-1))
                    all_y_loss.append(self._bce(preds["y"][y_m], y_trace[:, t, :][y_m]))
                
                if self.res_only and x_mask.sum() > 0:
                    all_x_res_loss.append(self._mse(preds["x_res"][x_mask], x_res_trace[:, t, :][x_mask]))
                    all_y_res_loss.append(self._mse(preds["y_res"][y_mask], y_res_trace[:, t, :][y_mask]))
            else:
                #single-sample
                all_alpha_loss.append(self._mse(preds["alpha"], alpha_trace[:, t, :]))
                all_beta_loss.append(self._mse(preds["beta"], beta_trace[:, t, :]))
                all_delta_loss.append(self._mse(preds["delta"], delta_trace[t].unsqueeze(-1)))
                all_x_loss.append(self._bce(preds["x"], x_trace[:, t, :], mask=x_loss_mask))
                all_y_loss.append(self._bce(preds["y"], y_trace[:, t, :], mask=y_loss_mask))
                if self.res_only:
                    all_x_res_loss.append(self._mse(preds["x_res"], x_res_trace[:, t, :]))
                    all_y_res_loss.append(self._mse(preds["y_res"], y_res_trace[:, t, :]))

            #store the predictions for next step
            prev_alpha = preds["alpha"].detach()
            prev_beta = preds["beta"].detach()
            prev_x = torch.sigmoid(preds["x"]).detach()
            prev_y = torch.sigmoid(preds["y"]).detach()
            if self.res_only:
                prev_x_res = preds["x_res"].detach()
                prev_y_res = preds["y_res"].detach()

        #restore original state of data
        batch["alpha"].x = alpha_trace
        batch["beta"].x = beta_trace
        batch.delta = delta_trace
        
        if self.res_only:
            batch["x"].x = x_res_trace
            batch["y"].x = y_res_trace
            batch["x"].trace_sol = x_trace
            batch["y"].trace_sol = y_trace
        else:
            batch["x"].x = x_trace
            batch["y"].x = y_trace
            batch["alpha", "to", "beta"].edge_attr = dist_ab
            batch["beta", "to", "alpha"].edge_attr = dist_ba

        #use mean loss aggregation
        alpha_loss = torch.stack(all_alpha_loss).mean()
        beta_loss = torch.stack(all_beta_loss).mean()
        x_loss = torch.stack(all_x_loss).mean()
        y_loss = torch.stack(all_y_loss).mean()
        delta_loss = torch.stack(all_delta_loss).mean()
        if self.res_only:
            x_res_loss = torch.stack(all_x_res_loss).mean()
            y_res_loss = torch.stack(all_y_res_loss).mean()

        #compute the optimum gap at the last prediction
        total_cost = self.convert_to_solution(batch, preds["x"], preds["y"])
        optimum_diff = (total_cost/batch.optimum).mean()  #mean across batch
        dual_diff = (total_cost / batch.dual_solution).mean()  #mean across batch

        optimum_loss = self._mse(total_cost, batch.optimum) #it's okay to use mse here, since the cost will never be lower than the optimum

        return {
            "alpha_loss": alpha_loss,
            "beta_loss": beta_loss,
            "x_loss": x_loss,
            "y_loss": y_loss,
            "delta_loss": delta_loss,
            "optimum_loss": optimum_loss,
            "optimum_diff": optimum_diff,
            "dual_diff": dual_diff,
            "x_res_loss": x_res_loss if self.res_only else 0.0,
            "y_res_loss": y_res_loss if self.res_only else 0.0,
        }