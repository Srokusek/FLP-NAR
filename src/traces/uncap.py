from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import time

from ..data.data import UncapInstance, UncapSolution, JVTrace

def _sum_contrib_at(distances: np.ndarray, t: float) -> float:
    if distances.size == 0:
        return 0.0
    d = np.sort(distances)
    k = np.searchsorted(d, t, side="right")
    if k == 0:
        return 0.0
    prefix = np.cumsum(d)
    return float(k * t - prefix[k - 1])

def _time_to_tight(distances: np.ndarray, cost: float, current_t: float) -> float:
    if distances.size == 0:
        return float("inf")
    if cost <= 0.0:
        return current_t
    
    d = np.sort(distances)
    return _time_to_tight_sorted(d, cost, current_t)

def _time_to_tight_sorted(sorted_distances: np.ndarray, cost: float, current_t: float) -> float:
    if sorted_distances.size == 0:
        return float("inf")
    if cost <= 0.0:
        return current_t

    d = sorted_distances
    prefix = np.cumsum(d)
    k = int(np.searchsorted(d, current_t, side="right"))
    if k == 0:
        sum_val = 0.0
    else:
        sum_val = float(k * current_t - prefix[k - 1])
    if sum_val >= cost:
        return current_t
    
    t = current_t
    while k < d.size:
        next_break = float(d[k])
        slope = float(k)
        if slope > 0.0:
            t_candidate = t + (cost - sum_val) / slope
            if t_candidate <= next_break:
                return t_candidate
        sum_val += slope * (next_break - t)
        t = next_break
        k += 1

    slope = float(d.size)
    return t + (cost - sum_val) / slope

def _build_conflict_graph(neighborhoods: List[np.ndarray]) -> List[List[int]]:
    num = len(neighborhoods) #number of neighborhoods
    conflicts: List[List[int]] = [[] for _ in range(num)]
    for i in range(num):
        ni = neighborhoods[i]
        for j in range(i + 1, num):
            if np.intersect1d(ni, neighborhoods[j], assume_unique=False).size > 0:
                conflicts[i].append(j)
                conflicts[j].append(i)
    return conflicts

# def solve_uncap_jv(instance: UncapInstance) -> JVTrace:
#     dist = np.asarray(instance.dist_matrix, dtype=np.float64)
#     facility_costs = np.asarray(instance.facility_costs)

#     n_cli, n_fac = dist.shape
#     demands = np.asarray(instance.demands)
#     dist = dist * demands[:, None] #demand weighted distance

#     active = np.ones(n_cli, dtype=bool) #active => client is contributing to growing the duals
#     dual_y = np.zeros(n_cli, dtype=np.float64) #alpha_i => client budget
#     tight_facilities = np.zeros(n_fac, dtype=bool) #start with no facilities set to open
#     tight_times = np.full(n_fac, np.nan, dtype=np.float64) #time step at which a given facility becomes tight

#     alpha = np.zeros(n_cli, dtype=np.float64)
#     beta = np.zeros((n_cli, n_fac), dtype=np.float64)

#     #trace storage
#     alpha_traces = []
#     beta_traces = []
#     assignment_traces = []
#     opened_traces = []
#     steps = []

#     t = 0.0
#     eps = 1e-9

#     #start timing
#     t0 = time.time()

#     sorted_idx = np.argsort(dist, axis=0)
#     sorted_dist = np.take_along_axis(dist, sorted_idx, axis=0)

#     while np.any(active): #while there are clients contributing (== not considered as served)
#         t_candidates = np.full(n_fac, np.inf, dtype=np.float64)
#         for j in range(n_fac):
#             if tight_facilities[j]:
#                 continue
#             active_sorted = active[sorted_idx[:, j]]
#             distances = sorted_dist[active_sorted, j]
#             t_candidates[j] = _time_to_tight_sorted(distances, facility_costs[j], t)

#         t_next = float(np.min(t_candidates))
#         if not np.isfinite(t_next): #no clients to become inactive/served
#             break
#         t = t_next #skip to the time step where the next dual constraint becomes tight

#         newly_tight = np.where(t_candidates <= t + eps)[0]
#         if newly_tight.size == 0:
#             break

#         if not np.any(~tight_facilities[newly_tight]):
#             break

#         for j in newly_tight:
#             if tight_facilities[j]: #already tight
#                 continue
#             tight_facilities[j] = True
#             tight_times[j] = t
#             newly_frozen = active & (dist[:, j] <= t + eps)
#             dual_y[newly_frozen] = t
#             active[newly_frozen] = False

#         #save snapshot
#         alpha[:] = dual_y
#         beta = np.maximum(0.0, alpha[:, None] - dist)

#         #save to trace history, BEFORE conflict resolution
#         current_open = np.where(tight_facilities)[0]
#         assignemnts = np.full(n_cli, -1, dtype=int)
#         if current_open.size > 0:
#             assignemnts[:] = np.argmin(dist[:, current_open], axis=1)
#             assignemnts = current_open[assignemnts]

#         alpha_traces.append(alpha.copy())
#         beta_traces.append(beta.copy())
#         assignment_traces.append(assignemnts.copy())
#         opened_traces.append(tight_facilities.copy())
#         steps.append(t)

#     if np.any(active):
#         dual_y[active] = t
#         active[:] = False

#     tight_indices = np.where(tight_facilities)[0]
#     if tight_indices.size == 0:
#         open_facilities = np.zeros(n_fac, dtype=bool)
#         open_facilities[int(np.argmin(facility_costs))] = True
#     else: #conflict resolution
#         neighborhoods: List[np.ndarray] = []
#         for j in tight_indices:
#             neighborhoods.append(np.where(dual_y >= dist[:, j] - eps)[0])

#         conflicts = _build_conflict_graph(neighborhoods)

#         order = np.argsort(tight_times[tight_indices])
#         selected = np.zeros(len(tight_indices), dtype=bool)
#         blocked = np.zeros(len(tight_indices), dtype=bool)
#         for idx in order:
#             if blocked[idx]:
#                 continue
#             selected[idx] = True
#             for nbr in conflicts[idx]: #block others
#                 blocked[nbr] = True

#         open_facilities = np.zeros(n_fac, dtype=bool)
#         open_facilities[tight_indices[selected]] = True

#     if not np.any(open_facilities):
#         open_facilities[int(np.argmin(facility_costs))] = True

#     open_indices = np.where(open_facilities)[0]
#     assignments = np.argmin(dist[:, open_indices], axis=1)
#     assignments = open_indices[assignments]

#     connection_cost = float(np.sum(dist[np.arange(n_cli), assignments] * demands[:, None]))
#     facility_cost = float(np.sum(facility_costs[open_facilities]))
#     steps = np.array(steps)
#     deltas = np.diff(steps, prepend=0)

#     solve_time = time.time() - t0

#     solution = UncapSolution(
#         open_facilities=open_facilities,
#         client_assignment=assignments,
#         total_cost=connection_cost + facility_cost,
#         opening_costs=facility_cost,
#         assignment_cost=connection_cost,
#         solve_time=solve_time,
#     )

#     return JVTrace(
#         alpha=np.array(alpha_traces),
#         beta=np.array(beta_traces),
#         assignments=np.array(assignment_traces),
#         open_facilities=np.array(opened_traces),
#         deltas=deltas,
#         final_solution=solution,
#     )

def solve_uncap_jv(instance: UncapInstance) -> JVTrace:
    demands = np.asarray(instance.demands, dtype=np.float64) #[n_cli]
    facility_costs = np.asarray(instance.facility_costs, dtype=np.float64) #[n_fac]
    dist = np.asarray(instance.dist_matrix, dtype=np.float64) #[n_cli, n_fac]
    n_cli, n_fac = dist.shape

    dist_w = dist * demands[:, None] #demand weighted distance

    #initiate dual variables
    alpha = np.zeros(n_cli, dtype=np.float64)
    beta = np.zeros((n_cli, n_fac), dtype=np.float64)

    #initiate primal variables
    y = np.zeros(n_fac, dtype=np.bool)
    x_assignment = np.zeros(n_cli, dtype=int)

    #keep track of algorithm state
    client_served = np.zeros(n_cli, dtype=bool)
    y_tight_time = np.full(n_fac, np.nan)
    t = 0.0
    eps = 1e-9

    #presort demand weighted distances for efficient tight-time computation
    sorted_idx = np.argsort(dist_w, axis=0)
    sorted_dist = np.take_along_axis(dist_w, sorted_idx, axis=0)

    #initiate traces
    alpha_traces = [alpha.copy()]
    beta_traces = [beta.copy()]
    x_assignment_traces = [x_assignment.copy()]
    y_traces = [y.copy()]
    client_served_trace = [client_served.copy()]
    t_steps = [0.0]

    #start timing
    t0 = time.time()

    #main loop -> iterate until all clients are assigned
    while np.any(~client_served):
        t_until_tight = np.full(n_fac, np.inf, dtype=np.float64)
        #for each not-tight facility:
        for j in range(n_fac):
            if y[j]:
                continue
            #add the contribution from frozen clients that are assigned to the facility
            frozen_contrib = np.sum(np.maximum(0.0, alpha - dist_w[:, j]) * client_served)
            remaining_cost = facility_costs[j] - frozen_contrib

            #calculate the closest t when facility j would become tight and save it
            unserved_sorted = (~client_served)[sorted_idx[:, j]]
            distances = sorted_dist[unserved_sorted, j]
            t_until_tight[j] = _time_to_tight_sorted(distances, remaining_cost, t)
        
        #move to the nearest time when a facility would become tight
        t_next = float(np.min(t_until_tight))
        if not np.isfinite(t_next):
            break

        t = t_next

        newly_tight_j = np.where(t_until_tight <= t + eps)[0] #index of facility that became tight
        if newly_tight_j.size == 0 or np.all(y[newly_tight_j]):
            break

        #open the newly opened facility(ies) and freeze clients that are now served
        for j in newly_tight_j:
            if y[j]:
                continue
            y[j] = True #set to open
            y_tight_time[j] = t
            newly_served = (~client_served) & (dist_w[:, j] <= t + eps) #newly served clients are ones that were not served before and their budget is enough to connect to newly opened facility
            alpha[newly_served] = t 
            client_served[newly_served] = True #freeze alpha of newly served clients

        #grow the alpha of the unserved customers
        alpha[~client_served] = t

        #update beta
        beta = np.maximum(0.0, alpha[:, None] - dist_w)

        #get the current best x_ij by assigning each client to nearest open facility
        open_j = np.where(y)[0]
        if open_j.size > 0:
            nearest = np.argmin(dist_w[:,open_j], axis=1)
            x_assignment = open_j[nearest] #[n_cli] facility indices
        
        #save snapshot into traces
        alpha_traces.append(alpha.copy())
        beta_traces.append(beta.copy())
        x_assignment_traces.append(x_assignment.copy())
        y_traces.append(y.copy())
        t_steps.append(t)
        client_served_trace.append(client_served.copy())

    #freeze any clients that are still unserved
    alpha[~client_served] = t
    client_served[:] = True
    ###How could this be modeled better for the NAR system?

    #confict resolution
    tight_indices = np.where(y)[0]
    if tight_indices.size == 0:
        y[int(np.argmin(facility_costs))] = True
    else:
        neighborhoods = [np.where(alpha >= dist_w[:, j] - eps)[0] for j in tight_indices]
        conflicts = _build_conflict_graph(neighborhoods)

        order = np.argsort(y_tight_time[tight_indices]) #prefer earlier tight facilities
        selected = np.zeros(len(tight_indices), dtype=bool)
        blocked = np.zeros(len(tight_indices), dtype=bool)
        for idx in order:
            if blocked[idx]:
                continue
            selected[idx] = True
            for nbr in conflicts[idx]:
                blocked[nbr] = True
        
        y[:] = False
        y[tight_indices[selected]] = True

    if not np.any(y):
        y[int(np.argmin(facility_costs))] = True

    #final x_ij after conflict resolution
    open_indices = np.where(y)[0]
    nearest = np.argmin(dist_w[:, open_indices], axis=1)
    final_assignments = open_indices[nearest]

    connection_cost = float(np.sum(dist_w[np.arange(n_cli), final_assignments]))
    opening_cost = float(np.sum(facility_costs[y]))

    t_steps_array = np.array(t_steps)
    deltas = np.diff(t_steps_array, prepend=0.0)

    solution = UncapSolution(
        open_facilities=y,
        client_assignment=final_assignments,
        total_cost=connection_cost + opening_cost,
        opening_costs=opening_cost,
        assignment_cost=connection_cost,
        solve_time=time.time() - t0,
    )

    return JVTrace(
        alpha=np.array(alpha_traces), #[T, n_cli]
        beta=np.array(beta_traces), #[T, n_cli, n_fac]
        assignments=np.array(x_assignment_traces), #[T, n_cli]
        open_facilities=np.array(y_traces), #[T, n_fac]
        deltas=deltas, #[T,]
        client_served=np.array(client_served_trace), #[T, n_cli]
        final_solution=solution,
    )
        

# def solve_uncap_jv_nar(instance: UncapInstance) -> JVTrace:
#     demands        = np.asarray(instance.demands, dtype=np.float64)         # [n_cli]
#     facility_costs = np.asarray(instance.facility_costs, dtype=np.float64)  # [n_fac]
#     dist           = np.asarray(instance.dist_matrix, dtype=np.float64)     # [n_cli, n_fac]
#     n_cli, n_fac   = dist.shape

#     # Demand-weighted distances, matching the LP: min Σ d_ij * dem_i * x_ij + Σ c_j * y_j
#     dist_w = dist * demands[:, None]  # [n_cli, n_fac]

#     # ------------------------------------------------------------------
#     # Dual variables (grown by the algorithm)
#     # ------------------------------------------------------------------
#     alpha = np.zeros(n_cli, dtype=np.float64)           # α_i: client budget, grows until client served
#     beta  = np.zeros((n_cli, n_fac), dtype=np.float64)  # β_ij = max(0, α_i - d_ij * dem_i)

#     # ------------------------------------------------------------------
#     # Primal variables (set when facilities go tight)
#     # ------------------------------------------------------------------
#     y            = np.zeros(n_fac, dtype=bool)   # y_j: facility open indicator
#     x_assignment = np.zeros(n_cli, dtype=int)    # x encoded as index: nearest open facility per client

#     # ------------------------------------------------------------------
#     # Algorithm state
#     # ------------------------------------------------------------------
#     client_served = np.zeros(n_cli, dtype=bool)  # True once α_i is frozen (client assigned)
#     y_tight_time  = np.full(n_fac, np.nan)        # time y_j went tight (needed for conflict resolution)
#     t   = 0.0
#     eps = 1e-9

#     # Pre-sort demand-weighted distances per facility for efficient tight-time computation
#     sorted_idx  = np.argsort(dist_w, axis=0)
#     sorted_dist = np.take_along_axis(dist_w, sorted_idx, axis=0)

#     # ------------------------------------------------------------------
#     # Trace storage — includes initial state at t=0
#     # ------------------------------------------------------------------
#     alpha_traces       = [alpha.copy()]        # [T, n_cli]
#     beta_traces        = [beta.copy()]         # [T, n_cli, n_fac]
#     x_assignment_traces = [x_assignment.copy()]  # [T, n_cli]  — facility index per client
#     y_traces           = [y.copy()]            # [T, n_fac]
#     t_steps            = [0.0]                 # [T,] — absolute time at each snapshot

#     t0 = time.time()

#     # ------------------------------------------------------------------
#     # Main loop: grow α_i uniformly for all unserved clients
#     # ------------------------------------------------------------------
#     while np.any(~client_served):

#         # For each non-tight facility j, compute time until it goes tight.
#         # Frozen clients already contribute max(0, α_i - d_ij) permanently,
#         # so subtract their contribution from the remaining facility cost.
#         t_until_tight = np.full(n_fac, np.inf, dtype=np.float64)
#         for j in range(n_fac):
#             if y[j]:
#                 continue
#             frozen_contrib = np.sum(np.maximum(0.0, alpha - dist_w[:, j]) * client_served)
#             remaining_cost = facility_costs[j] - frozen_contrib

#             unserved_sorted = (~client_served)[sorted_idx[:, j]]
#             distances       = sorted_dist[unserved_sorted, j]
#             t_until_tight[j] = _time_to_tight_sorted(distances, remaining_cost, t)

#         # Advance time to the next event (next facility going tight)
#         t_next = float(np.min(t_until_tight))
#         if not np.isfinite(t_next):
#             break

#         t = t_next  # δ = t_next - t_prev is the step size for this iteration

#         newly_tight_j = np.where(t_until_tight <= t + eps)[0]
#         if newly_tight_j.size == 0 or np.all(y[newly_tight_j]):
#             break

#         # Open newly tight facilities and freeze clients that are now served
#         for j in newly_tight_j:
#             if y[j]:
#                 continue
#             y[j]           = True
#             y_tight_time[j] = t
#             newly_served   = (~client_served) & (dist_w[:, j] <= t + eps)
#             alpha[newly_served]    = t   # freeze α_i at current time
#             client_served[newly_served] = True

#         # All remaining unserved clients still have α_i = t (still growing)
#         alpha[~client_served] = t

#         # β_ij = max(0, α_i - d_ij * dem_i)
#         beta = np.maximum(0.0, alpha[:, None] - dist_w)

#         # x_ij: assign each client to its nearest open facility
#         open_j = np.where(y)[0]
#         if open_j.size > 0:
#             nearest      = np.argmin(dist_w[:, open_j], axis=1)  # [n_cli]
#             x_assignment = open_j[nearest]                        # [n_cli] facility indices

#         # Save snapshot at this event
#         alpha_traces.append(alpha.copy())
#         beta_traces.append(beta.copy())
#         x_assignment_traces.append(x_assignment.copy())
#         y_traces.append(y.copy())
#         t_steps.append(t)

#     # Freeze any clients still unserved after the loop
#     alpha[~client_served] = t
#     client_served[:] = True

#     # ------------------------------------------------------------------
#     # Conflict resolution: select a non-conflicting subset of tight
#     # facilities. Two tight facilities conflict if their client
#     # neighborhoods (clients with α_i >= d_ij) overlap.
#     # ------------------------------------------------------------------
#     tight_indices = np.where(y)[0]
#     if tight_indices.size == 0:
#         y[int(np.argmin(facility_costs))] = True
#     else:
#         neighborhoods = [np.where(alpha >= dist_w[:, j] - eps)[0] for j in tight_indices]
#         conflicts     = _build_conflict_graph(neighborhoods)

#         order    = np.argsort(y_tight_time[tight_indices])  # prefer earlier-tight facilities
#         selected = np.zeros(len(tight_indices), dtype=bool)
#         blocked  = np.zeros(len(tight_indices), dtype=bool)
#         for idx in order:
#             if blocked[idx]:
#                 continue
#             selected[idx] = True
#             for nbr in conflicts[idx]:
#                 blocked[nbr] = True

#         y[:] = False
#         y[tight_indices[selected]] = True

#     if not np.any(y):
#         y[int(np.argmin(facility_costs))] = True

#     # Final x_ij after conflict resolution
#     open_indices      = np.where(y)[0]
#     nearest           = np.argmin(dist_w[:, open_indices], axis=1)
#     final_assignments = open_indices[nearest]  # [n_cli] — facility index per client

#     connection_cost = float(np.sum(dist_w[np.arange(n_cli), final_assignments]))
#     opening_cost    = float(np.sum(facility_costs[y]))

#     t_steps_arr = np.array(t_steps)
#     deltas      = np.diff(t_steps_arr, prepend=0.0)  # δ_k = t_k - t_{k-1}, shape [T]

#     solution = UncapSolution(
#         open_facilities=y,
#         client_assignment=final_assignments,
#         total_cost=connection_cost + opening_cost,
#         opening_costs=opening_cost,
#         assignment_cost=connection_cost,
#         solve_time=time.time() - t0,
#     )

#     return JVTrace(
#         alpha=np.array(alpha_traces),                # [T, n_cli]
#         beta=np.array(beta_traces),                  # [T, n_cli, n_fac]
#         assignments=np.array(x_assignment_traces),   # [T, n_cli]
#         open_facilities=np.array(y_traces),          # [T, n_fac]
#         deltas=deltas,                               # [T,]
#         final_solution=solution,
#     )