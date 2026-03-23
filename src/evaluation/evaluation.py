import copy
import os
import pickle
from pathlib import Path
from typing import Optional
import pandas as pd

from torch.utils.data import Dataset
from ..data.data import UncapGeneratorConfig, TrainingSample, ConstantConfig, NormalConfig, UniformConfig
from ..data.generators import generate_uncap_instance
from ..solvers import solve_uncap_exact, solve_dual_exact
from ..traces import solve_uncap_jv
from ..data.prepare_data import GenerateDataset
from ..utils.collate_fn import test_collate_fn
from ..training.training import Trainer 

import torch
from torch_geometric.data import HeteroData
from torch.utils.data import DataLoader
import numpy as np

def create_test_datasets(
        sizes: list[tuple[int, int]], #list of tuples in the form (n_cli, n_fac)
        n_samples: int, #number of samples to generate for each of the sample size
        type: str, #allows different distributions of data created (specifically by changing the distance calculation)
        exact: bool = True, #set to False if generating large instance -> only use the approximation Heuristic
        base_seed = 9999,
):
    test_datasets = {}

    #set the GeneratorConfig and path depending on the type of data distribution
    if type == "metric":
        demand_config = ConstantConfig(1)
        path = Path(__file__).resolve().parent.parent / "datasets" / "metric"
        distance_calculation = "euclidean"
    elif type == "weighted":
        demand_config = UniformConfig(min=0, max=1)
        path = Path(__file__).resolve().parent.parent / "datasets" / "weighted"
        distance_calculation = "euclidean"
    elif type == "random":
        demand_config = UniformConfig(min=0, max=1)
        path = Path(__file__).resolve().parent.parent / "datasets" / "random"
        distance_calculation = "random"
    else:
        raise ValueError(f"The selected instance typr ({type}) is not implemented, use one of [metric, weighted, random]")

    for i, (n_cli, n_fac) in enumerate(sizes):

        config = UncapGeneratorConfig(
            n_fac=n_fac,
            n_cli=n_cli,
            seed=base_seed + i * 1000,
            demand_config=demand_config,
            facility_cost_config=UniformConfig(min=0,max=1),
            coords_config=NormalConfig(mean=0.5, std=0.5),
            distance_calculation=distance_calculation
        )

        dataset = GenerateDataset(
            generator_config = config,
            n_samples=n_samples,
            cache_dir= path / f"test_{n_cli}_{n_fac}"
        )
        dataset.prepare_test_data(exact=exact)

        loader = DataLoader(
            dataset,
            batch_size=1, #inference implementation is currecntly only compatible with single sample batches
            shuffle=False,
            collate_fn=test_collate_fn,
        )

        test_datasets[(n_cli, n_fac)] = {"dataset":dataset, "loader":loader}

    return test_datasets

def run_inference(
        datasets: dict, #dictionary of to run inference on
        trainer: Trainer, #path to the pretrained model
        repair: bool = True #repair final solution? i.e. close any proposed facilities which did not get assigned any clients
):
    all_results = []
    for (n_cli, n_fac), data in datasets.items():
        print(f"evaluating {n_cli}x{n_fac}")
        for batch in data["loader"]:
            batch = batch.to(trainer.device)
            all_results.extend(trainer.model.inference(batch, repair=repair))

    #collect all resuts in a dataframe
    results_df = pd.DataFrame([
        {
            "size": f"{r['n_fac']}x{r['n_cli']}",
            "n_fac": r["n_fac"],
            "n_cli": r["n_cli"],
            "predicted": r["pred_cost"],
            "optimum": r["optimum"],
            "dual_bound": r["dual_bound"],
            "opt_ratio": r["opt_ratio"],
            "dual_ratio": r["dual_ratio"],
            "opt_gap_pct": (r["opt_ratio"] - 1.0) * 100,
            "n_fac_opened": int(r["repaired_opened"].sum()),
            "n_fac_target": int((r["y_target"] > 0.5).sum()),
        }
        for r in all_results
    ])

    return results_df
