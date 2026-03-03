import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import random
from torch.utils.data import Dataset
from .data import UncapInstance, UncapGeneratorConfig, GENERATOR_FACTORIES

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def _pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))

def generate_uncap_instance(config: UncapGeneratorConfig) -> UncapInstance:
    rng = np.random.default_rng(config.seed)

    coords_gen = GENERATOR_FACTORIES[type(config.coords_config)](config.coords_config)
    demand_gen = GENERATOR_FACTORIES[type(config.demand_config)](config.demand_config)
    cost_gen = GENERATOR_FACTORIES[type(config.facility_cost_config)](config.facility_cost_config)

    facility_coords = coords_gen(config.n_fac * 2, rng).reshape(config.n_fac, 2)
    client_coords = coords_gen(config.n_cli * 2, rng).reshape(config.n_cli, 2)

    demands = demand_gen(config.n_cli, rng)
    facility_costs = cost_gen(config.n_fac, rng)
    
    dist_matrix = _pairwise_dist(client_coords, facility_coords).astype(np.float32)

    return UncapInstance(
        facility_coords=facility_coords,
        client_coords=client_coords,
        demands=demands,
        dist_matrix=dist_matrix,
        facility_costs=facility_costs,
        seed=config.seed,
    )