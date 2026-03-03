from torch_geometric.data import Batch, HeteroData
import torch

def collate_fn(batch):
    #collate function for dataloaders for HeteroData,
    #-> is necessary, because since the number of steps of different traces is non-constant

    #get the number of algorithm steps for each sample in batch
    t_lengths = torch.tensor([int(sample.t_length) for sample in batch], dtype=torch.long)
    max_T = int(t_lengths.max().item()) #we will pad all other samples to the maximum length

    #prepare to collect deltas
    deltas_list = []

    #pad the temporal dimension of the sample tensors
    padded_samples = []
    for i, sample in enumerate(batch):
        T = int(sample.t_length)
        pad_size = max_T - T #amount of padding = difference with longest sample

        #initiate new heterodata object
        padded = HeteroData()
        for edge_type in sample.edge_types:
            padded[edge_type].edge_index = sample[edge_type].edge_index.clone()
            if hasattr(sample[edge_type], "edge_attr") and sample[edge_type].edge_attr is not None:
                padded[edge_type].edge_attr = sample[edge_type].edge_attr.clone()

        #helper function for temporal padding
        def pad_temporal(tensor, pad_size):
            if pad_size > 0:
                return torch.cat([
                    tensor,
                    tensor[:, -1:, :].expand(-1, pad_size, -1)
                ], dim=1)
            return tensor.clone() #return original tensor if no padding necessary
        
        #pad the node features
        padded["alpha"].x = pad_temporal(sample["alpha"].x, pad_size)
        padded["beta"].x = pad_temporal(sample["beta"].x, pad_size)
        padded["x"].x = pad_temporal(sample["x"].x, pad_size)
        padded["y"].x = pad_temporal(sample["y"].x, pad_size)
        if hasattr(sample["x"], "trace_sol"):
            padded["x"].trace_sol = pad_temporal(sample["x"].trace_sol, pad_size)
        if hasattr(sample["y"], "trace_sol"):
            padded["y"].trace_sol = pad_temporal(sample["y"].trace_sol, pad_size)

        #copy the static features
        if hasattr(sample["x"], "demands"):
            padded["x"].demands = sample["x"].demands.clone()
        if hasattr(sample["y"], "f_costs"):
            padded["y"].f_costs = sample["y"].f_costs.clone()
        if hasattr(sample["x"], "dist"):
            padded["x"].dist = sample["x"].dist.clone()

        #pad delta, (different ndims, can't use the helper funciton)
        if pad_size > 0:
            sample_delta_padded = torch.cat([
                sample.delta,
                sample.delta[-1:, :].expand(pad_size, -1)
            ], dim=0)
        else:
            sample_delta_padded = sample.delta.clone()
        
        deltas_list.append(sample_delta_padded) #[max_T, 1]

        padded.optimum = sample.optimum.clone() if isinstance(sample.optimum, torch.Tensor) else torch.tensor(sample.optimum)
        padded.dual_solution = sample.dual_solution.clone() if isinstance(sample.dual_solution, torch.Tensor) else torch.tensor(sample.dual_solution)
        padded.t_length = max_T #all padded to same length

        padded_samples.append(padded)
    
    batched = Batch.from_data_list(padded_samples)

    batched.delta = torch.stack(deltas_list, dim=1) #[max_t, B, 1]

    #keep also the original length for loss masking
    batched.t_lengths = t_lengths
    batched.max_T = max_T

    return batched 