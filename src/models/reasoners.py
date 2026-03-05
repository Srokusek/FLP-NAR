import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter

from .processors import processor

class Reasoner(nn.Module):
    def __init__(self, hidden_dim, tf_prob: float = 0.5):
        super().__init__()
        self.tf_prob = tf_prob
        self.eps = 1e-8

        #encoders
        self.x_res_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.y_res_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.alpha_init = nn.Parameter(torch.randn(1,hidden_dim) * 0.01)
        self.beta_init = nn.Parameter(torch.randn(1,hidden_dim) * 0.01)

        #processor
        self.processor = processor(hidden_dim)

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
    
    def _encode_node_features(self, x_res_in, y_res_in):
        h_x_res = self.x_res_encoder(x_res_in)
        h_y_res = self.y_res_encoder(y_res_in)
        return h_x_res, h_y_res
    
    def _set_node_features(self, batch, h_x, h_y):
        batch["x"].x = h_x
        batch["y"].x = h_y
        if not batch.duals_initialized: #initialize during the first step, then keep from last step
            batch["alpha"].x = self.alpha_init.expand(batch["alpha"].x.shape[0], -1) #learned parametetr
            batch["beta"].x = self.beta_init.expand(batch["beta"].x.shape[0], -1) #learned parameter
        return batch
    
    def _decode_step(self, batch):
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

    def mask_to_solution(self, batch: HeteroData, masks):
        #use the mask that was created throughout the algorithm run, not the last prediction (message passing is limited at later stages)
        open_facilites = masks["opened_facilities"] #[n_fac * B, 1]

        f_costs = batch["y"].f_costs #[n_fac * B, 1]
        demands = batch["x"].demands #[n_cli * B, 1]
        dist = batch["x"].dist
        dist_w = dist * demands #[n_cli * n_fac * B, 1]

        has_batch = hasattr(batch["y"], "batch")

        if has_batch:
            n_samples = batch["y"].batch.max().item() + 1
            n_fac = (batch["y"].batch == 0).sum().item()
            n_cli = (batch["alpha"].batch == 0).sum().item()
        else:
            n_samples = 1
            n_fac = batch["y"].x.shape[0]
            n_cli = batch["alpha"].x.shape[0]

        opened_facilites_per_sample = open_facilites.squeeze(-1).view(n_samples, n_fac) #[B, n_fac]
        f_costs_per_sample = f_costs.squeeze(-1).view(n_samples, n_fac) #[B, n_fac]
        dist_w_per_sample = dist_w.squeeze(-1).view(n_samples, n_cli, n_fac) #[B, n_cli]

        facility_cost = (opened_facilites_per_sample * f_costs_per_sample).sum(dim=1) #[B] -> cost for opened faciliteis for each sample

        open_mask = opened_facilites_per_sample.unsqueeze(1) #[B, 1, n_fac]
        masked_dist = dist_w_per_sample.clone()
        masked_dist[open_mask.expand_as(masked_dist) == 0] = float("inf") #set distance as inf for unopened facilities

        #get the minimum distance per client
        min_dist_per_client = masked_dist.min(dim=2).values #[B, n_cli]

        #whenever no facilites are open -> use max distance
        #TODO: this is porbably not the best way, as the facility costs would be low and selecting maximum client cost might not be enough to offset thath
        max_dist_per_client = dist_w_per_sample.max(dim=2).values  # [B, n_cli]
        min_dist_per_client = torch.where(
            min_dist_per_client.isinf(),
            max_dist_per_client,
            min_dist_per_client
        )

        client_cost = min_dist_per_client.sum(dim=1)  # [B]

        total_cost = facility_cost + client_cost
        return total_cost.unsqueeze(-1)  # [B, 1]


    def convert_to_solution(self, batch: HeteroData, x_pred: torch.Tensor, y_pred: torch.Tensor):
        ##Convert predictions into total cost per sample.
        #Assumes all samples in the batch have the same n_cli and n_fac.
        y_prob = torch.sigmoid(y_pred).detach()  # [n_fac * B, 1]
        y_binary = (y_prob > 0.5).float()

        f_costs = batch["y"].f_costs  #[n_fac * B, 1]
        demands = batch["x"].demands  #[n_cli * B, 1]
        dist = batch["x"].dist
        dist_w = dist * demands  #[n_fac * n_cli * B, 1]

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
        batch.duals_initialized = False
        
        #get per-sample time lengths for masking
        t_lengths = getattr(batch, 't_lengths', None)
        has_variable_T = t_lengths is not None and len(t_lengths) > 1

        x_res_trace = batch["x"].x
        y_res_trace = batch["y"].x
        x_trace = batch["x"].trace_sol.float()
        y_trace = batch["y"].trace_sol.float()
        alpha_trace = batch["alpha"].x
        beta_trace = batch["beta"].x
        delta_trace = batch.delta

        #initialize "previous step" -> the inputs that will be given to the NAR system
        prev_x_res = x_res_trace[:, 0, :]
        prev_y_res = y_res_trace[:, 0, :]

        all_alpha_loss = []
        all_beta_loss = []
        all_x_loss = []
        all_y_loss = []
        all_delta_loss = []
        optimum_loss = 0.0
        all_x_res_loss = []
        all_y_res_loss = []

        #init masks for tight facilities/clients
        masks = {
            "active_clients": torch.ones(prev_x_res.shape[0], dtype=torch.bool, device=prev_x_res.device),
            "opened_facilities": torch.zeros(prev_y_res.shape[0], dtype=torch.bool, device=prev_y_res.device),
        }
        
        #pre-compute batch indices, knowing that all samples within batch have same n_fac, n_cli
        if has_variable_T:
            """
            batching indices logic, the logic is complicated (at least for me), so I am documenting it in more detail here

            take for example that we have 2 samples in a batch, both with n_cli=3 and n_fac=2

            x_batch_idx = [0 0 0 0 0 0 0 1 1 1 1 1 1] ->which batch each of the x_ij belongs to
            y_batch_idx = [0 0 1 1 ] -> same but for y_j
            n_samples = 2 (obvious)
            n_fac_per_sample = 2 (n_fac) -> same for n_cli_per...  and n_x_per...

            x_local_idx = [0 1 2 3 4 5 0 1 2 3 4 5] -> the index of the x variables in terms of individual samples

            fac_idx_per_x = [0 1 0 1 0 1 0 1 0 1 0 1] -> which (local) facility does the x variable connect to, local j index
            cli_idx_per_x = [0 0 1 1 2 2 0 0 1 1 2 2] -> which (local) client does the x varaibel connect to, local i index

            now we turn these into global indices, such that we can index across the entire batch
            y_global_idx = [0 1 0 1 0 1 2 3 2 3 2 3] -> this is similar to fac_idx_per_x but now refers to the facility index in the global (batch) tensor
            alpha_global_idx = [0 0 1 1 2 2 3 3 4 4 5 5] -> same as y_global_idx, but in this case with clients/cli_idx_per_x  
            """
            x_batch_idx = batch["x"].batch #batch index for different nodes
            y_batch_idx = batch["y"].batch
            alpha_batch_idx = batch["alpha"].batch
            n_samples = x_batch_idx.max().item() + 1
            n_fac_per_sample = (y_batch_idx == 0).sum().item()
            n_cli_per_sample = (alpha_batch_idx == 0).sum().item()
            n_x_per_sample = n_fac_per_sample * n_cli_per_sample #x_ij
            
            x_local_idx = torch.arange(x_batch_idx.shape[0], device=x_batch_idx.device) % n_x_per_sample 
            
            fac_idx_per_x = x_local_idx % n_fac_per_sample

            cli_idx_per_x = x_local_idx // n_fac_per_sample

            y_global_idx = x_batch_idx * n_fac_per_sample + fac_idx_per_x
            alpha_global_idx = x_batch_idx * n_cli_per_sample + cli_idx_per_x

        for t in range(1,T):
            #fuzzy teacher mixing
            if self.training and self.tf_prob < 1.0:
                def _mix(ground_truth, pred):
                    mask = (torch.rand_like(ground_truth) < self.tf_prob)
                    return torch.where(mask, ground_truth, pred) #return combination of ground_truth and predictions based on the selected probability

                x_res_in = _mix(x_res_trace[:, t-1, :], prev_x_res)
                y_res_in = _mix(y_res_trace[:, t-1, :], prev_y_res)
                #frozen client masking - the x_ij nodes get disconnected once the relevant client (i index) has been assigned to a open facility
                if has_variable_T:
                    """
                    contuning with the example where we have 2 samples in batch, each with n_cli=3 and n_fac=2

                    y_res_for_x = [r00 r01 r00 r01 r00 r01 r10 r11 r10 r11 r10 r11] where r_sj (s is sample index, j is facility index) -> the related r_j for each x_ij with global indexing

                    x_tight = [...] -> binary mask where both res_x_ij and related res_y_j are tight, shape [n_x_per_sample * n_samples]

                    client_has_tight = [...] check for each of the x_ij, whether it has tight connection to any tight (open) facility -> client should be assigned to facility and frozen
                    mask["active_clietns"] = ~client_has_tight[alpha_global_idx] -> mask out any connections where clients are frozen, [alpha_global_idx] maps this to the batched tensor
                    """
                    y_res_for_x = y_res_in.squeeze(-1)[y_global_idx]  #[total_x]
                    x_tight = (x_res_in.squeeze(-1) < self.eps) & (y_res_for_x < self.eps)  #= client variable is tight and facility is tentatively open
                    client_has_tight = scatter(x_tight.float(), alpha_global_idx, dim=0, reduce='max') > 0  #[total_alpha]
                    masks["active_clients"] = ~client_has_tight[alpha_global_idx]
                    masks["opened_facilities"] = y_res_in < self.eps
                else:
                    #simplified version for when the data is not batched, no need for global index mapping etc.
                    tight_connections = ((x_res_in < self.eps) * (y_res_in < self.eps).repeat(n_cli, 1)).view(n_cli, n_fac)
                    frozen_clients = torch.sum(tight_connections, dim=1) > 0
                    masks["active_clients"] = ~frozen_clients.repeat_interleave(n_fac)
                    masks["opened_facilities"] = y_res_in < self.eps

            elif self.tf_prob == 1.0: #use only ground truth, used when tf_prob >= 1
                x_res_in = x_res_trace[:, t-1, :]
                y_res_in = y_res_trace[:, t-1, :]
                #frozen client masking
                if has_variable_T:
                    y_res_for_x = y_res_in.squeeze(-1)[y_global_idx]
                    x_tight = (x_res_in.squeeze(-1) < self.eps) & (y_res_for_x < self.eps)
                    client_has_tight = scatter(x_tight.float(), alpha_global_idx, dim=0, reduce='max') > 0
                    masks["active_clients"] = ~client_has_tight[alpha_global_idx]
                    masks["opened_facilities"] = y_res_in < self.eps
                else:
                    tight_connections = ((x_res_in < self.eps) * (y_res_in < self.eps).repeat(n_cli, 1)).view(n_cli, n_fac)
                    frozen_clients = torch.sum(tight_connections, dim=1) > 0
                    masks["active_clients"] = ~frozen_clients.repeat_interleave(n_fac)
                    masks["opened_facilities"] = y_res_in < self.eps
            elif not self.training and self.tf_prob < 1.0:
                #complete trace independent eval/test
                x_res_in = prev_x_res
                y_res_in = prev_y_res
                #frozen client masking
                if has_variable_T:
                    y_res_for_x = y_res_in.squeeze(-1)[y_global_idx]
                    x_tight = (x_res_in.squeeze(-1) < self.eps) & (y_res_for_x < self.eps)
                    client_has_tight = scatter(x_tight.float(), alpha_global_idx, dim=0, reduce='max') > 0
                    masks["active_clients"] = ~client_has_tight[alpha_global_idx]
                    masks["opened_facilities"] = y_res_in < self.eps
                else:
                    tight_connections = ((x_res_in < self.eps) * (y_res_in < self.eps).repeat(n_cli, 1)).view(n_cli, n_fac)
                    frozen_clients = torch.sum(tight_connections, dim=1) > 0
                    masks["active_clients"] = ~frozen_clients.repeat_interleave(n_fac)
                    masks["opened_facilities"] = y_res_in < self.eps

            #encode the node features
            h_x_res, h_y_res = self._encode_node_features(x_res_in, y_res_in)
            batch = self._set_node_features(batch, h_x_res, h_y_res)

            #the duals are initialized after the first pass
            batch.duals_initialized = True

            #do the processing step, maskingn out edges for frozen clients
            batch = self.processor(batch, masks)

            #decode into algorithm steps
            preds = self._decode_step(batch)

            #save losses with time-step masking for batched variable-T data
            if has_variable_T:
                #create masks for nodes belonging to samples with valid timestep t
                alpha_batch_idx = batch["alpha"].batch
                beta_batch_idx = batch["beta"].batch
                x_batch_idx = batch["x"].batch
                y_batch_idx = batch["y"].batch
                
                #valid_samples: t < t_length
                valid_samples = (t_lengths > t)  #[B]
                
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
                    all_x_loss.append(self._bce(preds["x"][x_mask], x_trace[:, t, :][x_mask]))
                    all_y_loss.append(self._bce(preds["y"][y_mask], y_trace[:, t, :][y_mask]))
                    all_x_res_loss.append(self._mse(preds["x_res"][x_mask], x_res_trace[:, t, :][x_mask]))
                    all_y_res_loss.append(self._mse(preds["y_res"][y_mask], y_res_trace[:, t, :][y_mask]))

                prev_x_res = preds["x_res"].detach()
                prev_y_res = preds["y_res"].detach()
                
            else:
                #single-sample
                all_alpha_loss.append(self._mse(preds["alpha"], alpha_trace[:, t, :]))
                all_beta_loss.append(self._mse(preds["beta"], beta_trace[:, t, :]))
                all_delta_loss.append(self._mse(preds["delta"], delta_trace[t].unsqueeze(-1)))
                all_x_loss.append(self._bce(preds["x"], x_trace[:, t, :]))
                all_y_loss.append(self._bce(preds["y"], y_trace[:, t, :]))
                all_x_res_loss.append(self._mse(preds["x_res"], x_res_trace[:, t, :]))
                all_y_res_loss.append(self._mse(preds["y_res"], y_res_trace[:, t, :]))

                prev_x_res = preds["x_res"].detach()
                prev_y_res = preds["y_res"].detach()

        #restore original state of data
        batch["alpha"].x = alpha_trace
        batch["beta"].x = beta_trace
        batch.delta = delta_trace
        batch["x"].x = x_res_trace
        batch["y"].x = y_res_trace
        batch["x"].trace_sol = x_trace
        batch["y"].trace_sol = y_trace

        #use mean loss aggregation
        alpha_loss = torch.stack(all_alpha_loss).mean()
        beta_loss = torch.stack(all_beta_loss).mean()
        x_loss = torch.stack(all_x_loss).mean()
        y_loss = torch.stack(all_y_loss).mean()
        delta_loss = torch.stack(all_delta_loss).mean()
        x_res_loss = torch.stack(all_x_res_loss).mean()
        y_res_loss = torch.stack(all_y_res_loss).mean()

        #compute the optimum gap at the last prediction
        #total_cost = self.convert_to_solution(batch, preds["x"], preds["y"])
        total_cost = self.mask_to_solution(batch, masks)
        optimum_diff = (total_cost / batch.optimum).mean()  #mean across batch
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
            "x_res_loss": x_res_loss,
            "y_res_loss": y_res_loss,
        }
    
    @torch.no_grad()
    def inference(self, batch: HeteroData):
        #pure inference step, only made to work with non-batched data for simplicity

        T = batch["alpha"].x.shape[1]
        n_fac = batch["y"].x.shape[0]
        n_cli = batch["alpha"].x.shape[0]
        batch.duals_initialized = False

        #collect algorithm traces
        x_res_trace = batch["x"].x
        y_res_trace = batch["y"].x
        x_trace_sol = batch["x"].trace_sol.float()
        y_trace_sol = batch["y"].trace_sol.float()

        #initial residual state
        prev_x_res = x_res_trace[:, 0, :]
        prev_y_res = y_res_trace[:, 0, :]

        masks = {
            "active_clients": torch.ones(prev_x_res.shape[0], dtype=torch.bool, device=prev_x_res.device),
            "opened_facilities": torch.zeros(prev_y_res.shape[0], dtype=torch.bool, device=prev_y_res.device),
        }

        #main loop
        for t in range(1, T):
            x_res_in = prev_x_res
            y_res_in = prev_y_res

            tight_connections = ((x_res_in < self.eps) * (y_res_in < self.eps).repeat(n_cli, 1)).view(n_cli, n_fac)
            frozen_clients = torch.sum(tight_connections, dim=1) > 0
            masks["active_clients"] = ~frozen_clients.repeat_interleave(n_fac)
            masks["opened_facilities"] = y_res_in < self.eps

            #encode
            h_x_res, h_y_res = self._encode_node_features(x_res_in, y_res_in)
            batch = self._set_node_features(batch, h_x_res, h_y_res)
            batch.duals_initialized = True

            #process
            batch = self.processor(batch, masks)

            #decode
            preds = self._decode_step(batch)

            #use previous prediction as input in next loop iteration
            prev_x_res = preds["x_res"]
            prev_y_res = preds["y_res"]

        #use opened facilities mask to extract solution cost
        total_cost = self.mask_to_solution(batch, masks)

        #collect results
        results = {
            "y_res": prev_y_res.squeeze(-1).cpu().numpy(),
            "x_res": prev_x_res.squeeze(-1).cpu().numpy(),
            "y_target": y_trace_sol[:, -1, :].squeeze(-1).cpu().numpy(), #last step in trace
            "x_target": x_trace_sol[:, -1, :].squeeze(-1).cpu().numpy(),
            "opened": masks["opened_facilities"].cpu().numpy(),
            "pred_cost": total_cost.squeeze().item(),
            "optimum": batch.optimum.squeeze().item(),
            "dual_bound": batch.dual_solution.squeeze().item(),
            "opt_ratio": (total_cost/batch.optimum).squeeze().item(),
            "dual_ratio": (total_cost/batch.dual_solution).squeeze().item(),
            "n_fac": n_fac,
            "n_cli": n_cli,
        }

        #restore sample into original form
        batch["x"].x = x_res_trace
        batch["y"].x = y_res_trace
        batch["x"].trace_sol = x_trace_sol
        batch["y"].trace_sol = y_trace_sol

        return [results]