from dataclasses import dataclass
import numpy as np
from typing import Tuple, Callable, Dict
import torch



@dataclass
class DualSolution:
    alpha: np.ndarray #[n_cli,]
    beta: np.ndarray #[n_cli, n_fac]
    objective: float

@dataclass
class UncapInstance:
    facility_coords: np.ndarray #[n_fac, 2] float
    client_coords: np.ndarray #[n_cli, 2] float
    demands: np.ndarray #[n_cli, ] 
    dist_matrix: np.ndarray #[n_cli, n_fac] float
    facility_costs: np.ndarray #[n_fac, ] float
    seed: int

@dataclass
class UncapSolution:
    open_facilities: np.ndarray[bool] #[n_fac, ]
    client_assignment: np.ndarray[bool] #[n_fac, ]
    opening_costs: float
    assignment_cost: float
    total_cost: float
    solve_time: float

@dataclass
class JVTrace:
    alpha: torch.Tensor #[n_steps, n_cli]
    beta: np.ndarray #[n_steps, n_cli, n_fac]
    assignments: np.ndarray #[n_steps, n_cli]
    open_facilities: np.ndarray #[n_steps, n_fac]
    deltas: np.ndarray #[n_steps, 1]
    client_served: np.ndarray #[n_steps, n_cli]
    final_solution: UncapSolution

@dataclass
class TrainingSample:
    instance: UncapInstance
    primal: UncapSolution
    dual: DualSolution
    traces: JVTrace

@dataclass
class DistributionConfig:
    """base class for distribution configs"""

@dataclass
class UncapGeneratorConfig:
    n_fac: int
    n_cli: int
    seed: int

    #distribution for the different randomized parameters
    demand_config: DistributionConfig
    facility_cost_config: DistributionConfig
    coords_config: DistributionConfig

@dataclass
class UniformConfig(DistributionConfig):
    min: float
    max: float

@dataclass
class NormalConfig(DistributionConfig):
    mean: float
    std: float

@dataclass
class ExponentialConfig(DistributionConfig):
    lambda_: float

DistributionGenerator = Callable[[int, np.random.Generator], np.ndarray]

#functions for generating the different distributions
def _create_uniform_generator(config: UniformConfig) -> DistributionGenerator:
    min_val = config.min
    max_val = config.max

    def generator(size: int, rng: np.random.Generator, precision=np.float32) -> np.ndarray:
        return rng.uniform(min_val, max_val, size).astype(precision)
    
    return generator

def _create_normal_generator(config: NormalConfig) -> DistributionConfig:
    mean = config.mean
    std = config.std

    def generator(size: int, rng: np.random.Generator, precision=np.float32) -> np.ndarray:
        values = rng.normal(mean, std, size)
        return values.astype(precision)
    
    return generator

GENERATOR_FACTORIES: Dict[type, Callable[[DistributionConfig], DistributionGenerator]] = {
    UniformConfig: _create_uniform_generator,
    NormalConfig: _create_normal_generator,
}