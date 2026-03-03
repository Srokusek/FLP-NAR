from ..data.data import UncapSolution, UncapInstance, DualSolution

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

def solve_uncap_exact(problem: UncapInstance, solver: str = "glpk") -> UncapSolution:
    dist = np.asarray(problem.dist_matrix, dtype=np.float64)
    demands = np.asarray(problem.demands, dtype=np.float64)
    n_cli, n_fac = dist.shape
    facility_costs = np.asarray(problem.facility_costs)

    #convert dist to dict-like format
    dist_dict = {
        (i, j): dist[i, j] for i in range(n_cli) for j in range(n_fac)
    }

    demands_dict = {
        i: demands[i] for i in range(n_cli)
    }

    facility_costs_dict = {
        j: facility_costs[j] for j in range(n_fac)
    }

    model = pyo.ConcreteModel()

    #init the sets
    model.I = pyo.Set(initialize=range(n_cli)) #client set
    model.J = pyo.Set(initialize=range(n_fac)) #facility set
    model.M = pyo.Param(initialize=1e6, mutable=True) #big-m

    #init the parameters
    model.c = pyo.Param(model.J, initialize=facility_costs_dict)
    model.d = pyo.Param(model.I, model.J, initialize=dist_dict)
    model.demand = pyo.Param(model.I, initialize=demands_dict)

    #initialize the variables
    model.x = pyo.Var(model.I, model.J, domain=pyo.Binary)
    model.y = pyo.Var(model.J, domain=pyo.Binary)

    #each client gets assigned
    def assigned_rule(model, i):
        return sum(model.x[i,:]) == 1
    model.assign = pyo.Constraint(model.I, rule=assigned_rule)

    #can only assign to facilities that are actually opened
    def opened_rule(model, i, j):
        return model.x[i, j] <= model.y[j]
    model.open = pyo.Constraint(model.I, model.J, rule=opened_rule)

    model.obj = pyo.Objective(
        expr=sum(model.d[i, j] * model.demand[i] * model.x[i, j] for i in model.I for j in model.J) + sum(model.c[j] * model.y[j] for j in model.J),
        sense = pyo.minimize
    )

    solver_model = pyo.SolverFactory(solver)
    result = solver_model.solve(model)

    #extract and pack the found solution
    x_solution = np.asarray([[pyo.value(model.x[i, j]) for j in model.J] for i in model.I], dtype=np.float32)
    open_facilities = np.asarray([pyo.value(model.y[j]) for j in model.J], dtype=np.float32)
    assignments = np.argmax(x_solution, axis=1)
    sol_facility_costs = np.sum(facility_costs * open_facilities)
    sol_assignment_cost = np.sum((dist * x_solution) * demands[:, None])


    return UncapSolution(
        open_facilities=open_facilities,
        client_assignment=assignments,
        opening_costs=sol_facility_costs,
        assignment_cost=sol_assignment_cost,
        total_cost=sol_facility_costs + sol_assignment_cost,
        solve_time=result["Solver"][0]["Time"]
    )

def solve_dual_exact(problem: UncapInstance, solver: str = "glpk") -> UncapSolution:
    dist = np.asarray(problem.dist_matrix, dtype=np.float64)
    demands = np.asarray(problem.demands, dtype=np.float64)
    n_cli, n_fac = dist.shape
    facility_costs = np.asarray(problem.facility_costs)

    #convert to dict-like format
    dist_dict = {
        (i, j): dist[i, j] for i in range(n_cli) for j in range(n_fac)
    }

    demands_dict = {
        i: demands[i] for i in range(n_cli)
    }

    facility_costs_dict = {
        j: facility_costs[j] for j in range(n_fac)
    }

    model = pyo.ConcreteModel()

    #initialize the sets
    model.I = pyo.Set(initialize=range(n_cli)) 
    model.J = pyo.Set(initialize=range(n_fac))
    
    #init the parameters
    model.c = pyo.Param(model.J, initialize=facility_costs_dict)
    model.d = pyo.Param(model.I, model.J, initialize=dist_dict)
    model.demand = pyo.Param(model.I, initialize=demands_dict)

    #initialize the variables
    model.alpha = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    model.beta = pyo.Var(model.I, model.J, domain=pyo.NonNegativeReals)

    ###Constraints
    #charge bumps for each facility is no more than its total cost
    def bumps_rule(model, j):
        return sum(model.beta[:,j]) <= model.c[j]
    model.bumps = pyo.Constraint(model.J, rule=bumps_rule)

    #tightness constraint
    def tightness_rule(model, i, j):
        return  model.alpha[i] - model.beta[i, j] <= model.d[i, j] * model.demand[i]
    model.tight = pyo.Constraint(model.I, model.J, rule=tightness_rule)

    ###Objective
    model.obj = pyo.Objective(
        expr=sum(model.alpha[i] for i in model.I),
        sense=pyo.maximize
    )

    solver_model = pyo.SolverFactory(solver)
    result = solver_model.solve(model)

    alpha_solution = np.asarray([pyo.value(model.alpha[i]) for i in model.I], dtype=np.float32)
    beta_solution = np.asarray([[pyo.value(model.beta[i,j]) for j in model.J] for i in model.I], dtype=np.float32)
    objective_value = pyo.value(model.obj)

    return DualSolution(
        alpha=alpha_solution,
        beta=beta_solution,
        objective=objective_value,
    )