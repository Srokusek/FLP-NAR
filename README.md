# FLP-NAR

Neural Algorithmic Reasoning (NAR) for the uncapacitated facility location
problem (UFLP). The model learns to imitate intermediate steps of the
Jain–Vazirani primal-dual approximation algorithm and produces feasible
facility-opening and client-assignment decisions.

The accompanying [paper](<Neural_Algorithmic_Reasoning_for_solving_the_uncapacitated_facility_location_problem (4).pdf>)
describes the method and experiments in detail.

## Overview

Given facility opening costs, client demands, and client-to-facility connection
costs, UFLP minimizes the sum of opening and assignment costs while assigning
every client to one open facility. This repository contains:

- a Pyomo/GLPK exact solver;
- a Jain–Vazirani teacher with execution-trace generation;
- a graph encoder–processor–decoder implemented with PyTorch Geometric;
- training with trace supervision, teacher forcing, and exact-cost supervision;
- evaluation on metric, demand-weighted, and random-cost instances.

The processor follows the teacher's dependency structure through three message-
passing phases (`g_alpha -> g_beta -> g_xy`). A pretrained 128-dimensional model
is included at `notebooks/checkpoints/pretrained_model.pt`.

## Setup

Python 3.12 is recommended. Exact solving and dataset generation also require the
GLPK `glpsol` executable.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Debian/Ubuntu; use the equivalent package for your system
sudo apt install glpk-utils
```

Install JupyterLab separately if needed:

```bash
pip install jupyterlab
```

## Usage

The quickest entry point is the showcase notebook. Launch it from the notebook
directory so its checkpoint and result paths resolve correctly:

```bash
cd notebooks
jupyter lab showcase.ipynb
```

`showcase.ipynb` demonstrates data generation, optional training, pretrained
checkpoint loading, inference, CSV export, and plots. Set the test `type` to
`"metric"`, `"weighted"`, or `"random"`; set `exact=False` for large instances
where only comparison with the Jain–Vazirani solution is required. The current
inference implementation uses a batch size of one.

`collab.ipynb` provides the same end-to-end workflow in a Colab-oriented format.
Generated datasets are cached under `src/datasets/`, while the paper's exported
experiment tables are included in `notebooks/results/`.

## Reported results

The paper reports that a model trained on instances no larger than `(10, 10)`:

- maintains comparatively stable solution quality on metric instances through
  `(500, 500)` and scales similarly to its teacher through `(1000, 1000)`;
- remains robust under the tested demand-weighted distribution shift and
  outperforms the teacher there;
- degrades when connection costs are fully random, showing that generalization
  still depends on training-distribution structure;
- becomes faster than both the implemented teacher and exact optimization on
  sufficiently large instances.

These are empirical trade-offs, not optimality guarantees. The Jain–Vazirani
3-approximation guarantee applies to metric UFLP; the learned model itself has no
formal approximation guarantee.

## Repository structure

```text
src/data/        instance generation and graph dataset preparation
src/solvers/     exact Pyomo formulations
src/traces/      Jain–Vazirani solver and trace generation
src/models/      NAR reasoner and graph processors
src/training/    training loop and checkpoint handling
src/evaluation/  test generation and inference summaries
src/utils/       batching and visualization helpers
notebooks/       showcase, Colab workflow, checkpoint, and results
```
