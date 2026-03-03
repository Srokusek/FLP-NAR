import copy
import os
import pickle
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset
from .data import UncapGeneratorConfig, TrainingSample
from .generators import generate_uncap_instance
from ..solvers import solve_uncap_exact, solve_dual_exact
from ..traces import solve_uncap_jv

import torch
from torch_geometric.data import HeteroData
import numpy as np

class GenerateDataset(Dataset):
    def __init__(self, generator_config:UncapGeneratorConfig, n_samples: int, cache_dir: Path, res_only: bool):
        super().__init__()
        #basic generator
        self.generator = generate_uncap_instance

        #exact solvers
        self.primal_solver = solve_uncap_exact
        self.dual_solver = solve_dual_exact

        #approximation algorithm for generating traces
        self.traces_solver = solve_uncap_jv

        self.n_samples = n_samples
        self.cache_dir = cache_dir
        self.generator_config = generator_config
        self.res_only = res_only #select graph type based on what variation of the processor we want to use
        
    def prepare_data(self):
        cache_file = os.path.join(self.cache_dir, f"{self.n_samples}_{self.generator_config.n_cli}_{self.generator_config.n_fac}.pkl")

        os.makedirs(self.cache_dir, exist_ok=True)

        if os.path.exists(cache_file):
            print("loading instances from cache")
            with open(cache_file, "rb") as f:
                self.data = pickle.load(f)
        else:
            print("generating instances")
            samples = self.generate_samples()
            self.data = [self.process_sample(sample) for sample in samples]
            with open(cache_file, "wb") as f:
                pickle.dump(self.data, f)

    def generate_samples(self):
        samples = []
        for n in range(self.n_samples):
            config = copy.copy(self.generator_config)
            config.seed = config.seed + n

            #generate the instance
            instance = self.generator(config)

            #get the primal and dual solution (exact)
            primal = self.primal_solver(instance)
            dual = self.dual_solver(instance)

            #get the traces
            traces = self.traces_solver(instance)

            #save as single training sample
            sample = TrainingSample(
                instance=instance,
                primal=primal,
                dual=dual,
                traces=traces
            )
            samples.append(sample)
        return samples

    def process_sample(self, sample: TrainingSample):
        #method for formatting a dataset that takes as input both primals and duals
        n_cli, n_fac = sample.instance.dist_matrix.shape
        t_steps = sample.traces.deltas.shape[0]
        ij_idx = torch.arange(n_cli * n_fac)
        i_idx = torch.arange(n_cli).repeat_interleave(n_fac)
        j_idx = torch.arange(n_fac).repeat(n_cli)

        #alpha: [t_steps, n_cli_nodes] -> [n_cli_nodes, t_steps, 1(features)]
        alpha = torch.from_numpy(sample.traces.alpha).transpose(0, 1).unsqueeze(-1) .float()

        #beta: [t_steps, n_cli, n_fac] -> [n_cli_nodes * n_fac_nodes, t_steps, 1(features)]
        beta = torch.from_numpy(sample.traces.beta).permute(1, 2, 0).reshape(n_cli*n_fac, t_steps).unsqueeze(-1).float()

        #deltas: [t_steps,] -> [t_steps, 1]
        deltas = torch.from_numpy(sample.traces.deltas).unsqueeze(-1) .float()
        
        #x: [n_cli, t_steps] -> [n_cli*n_fac, t_steps, 1]
        x = np.zeros((n_cli,n_fac,t_steps))
        x[np.tile(np.arange(n_cli), t_steps), sample.traces.assignments.flatten(), np.repeat(np.arange(t_steps), n_cli)] = 1
        x = torch.from_numpy(x).reshape(n_cli*n_fac, t_steps).unsqueeze(-1).long() 

        #y:[t_steps, n_fac] -> [n_fac, t_steps, 1]
        y = torch.from_numpy(sample.traces.open_facilities).permute(1,0).unsqueeze(-1).long()

        ##static features:
        #dist features: [n_cli,n_fac] -> [n_cli*n_fac, 1]
        dist = torch.from_numpy(sample.instance.dist_matrix).reshape(n_cli*n_fac).unsqueeze(-1).float()
        #demand: [n_cli,] -> [n_cli*n_fac, 1]  (pre-expanded to match x nodes)
        demands = torch.from_numpy(sample.instance.demands).repeat_interleave(n_fac).unsqueeze(-1).float()
        #facility costs: [n_fac,] -> [n_fac, 1]
        f_costs = torch.from_numpy(sample.instance.facility_costs).unsqueeze(-1).float()
        #optimal total cost based on exact solver: scalar -> [1]
        optimum = torch.tensor([sample.primal.total_cost], dtype=torch.float32)
        dual_solution = torch.tensor(sample.traces.final_solution.total_cost, dtype=torch.float32)

        sample = HeteroData()

        #node features
        sample.delta = deltas
        sample.optimum = optimum
        sample.dual_solution = dual_solution
        sample.t_length = x.shape[1] #x: [n_cli * n_fac, t_steps, 1]

        #connect nodes
        sample["x", "to", "alpha"].edge_index = torch.stack([ij_idx, i_idx])
        sample["alpha", "to", "beta"].edge_index = torch.stack([i_idx, ij_idx])
        sample["beta", "to", "y"].edge_index = torch.stack([ij_idx, j_idx])

        #->reverse direction
        sample["y", "to", "beta"].edge_index = torch.stack([j_idx, ij_idx])
        sample["beta", "to", "alpha"].edge_index = torch.stack([ij_idx, i_idx])
        sample["alpha", "to", "x"].edge_index = torch.stack([i_idx, ij_idx])

        sample["alpha"].x = alpha
        sample["beta"].x = beta

        if not self.res_only:
            sample["x"].x = x
            sample["y"].x = y

            #add edge attributes
            sample["alpha", "to", "beta"].edge_attr = dist
            sample["beta", "to", "alpha"].edge_attr = dist

            #store demands and f_costs for encoding
            sample["x"].demands = demands
            sample["y"].f_costs = f_costs

            return sample
        
        else:
            sample["x"].trace_sol = x
            sample["y"].trace_sol = y
            
            #store the demand weighted distance and f_costs as initial residuals -> in the dual this is the budget of each client/facility
            dist_w = dist * demands #[n_cli*n_fac, 1] * [n_cli*n_fac, 1] -> [n_cli*n_fac, 1]
            sample["x"].x = dist_w.unsqueeze(-1) - alpha.repeat((n_fac, 1, 1))
            sample["y"].x = f_costs.unsqueeze(-1) - torch.sum(beta.view((n_cli, n_fac, t_steps, 1)), dim=0)

            sample["x"].dist = dist
            sample["x"].demands = demands
            sample["y"].f_costs = f_costs

            #in this variation, no edge_attributes are necessary as distance is stored within the x node

            #TODO: possible future improvement would be to add some other features, 
            #like degree to inform the network about between how many possible connections to split the residual budget

            return sample
    
    def process_sample_res_inupts(self, sample: TrainingSample):
        #method for creating a dataset that is suited to have the residual (not the primals) as inputs and takes no duals as input

        #collect the instance parameters
        n_cli, n_fac = sample.instance.dist_matrix.shape
        t_steps = sample.traces.deltas.shape[0]

        #create indices for easy edge_index generation
        ij_idx = torch.arange(n_cli * n_fac)
        i_idx = torch.arange(n_cli).repeat_interleave(n_fac)
        j_idx = torch.arange(n_fac).repeat(n_cli)



    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]