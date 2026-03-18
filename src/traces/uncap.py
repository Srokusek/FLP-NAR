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

def solve_uncap_jv(instance: UncapInstance, with_traces = True) -> JVTrace:
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

        #check t for next event 2 - facility becomes tight
        t_fac_tight = np.full(n_fac, np.inf, dtype=np.float64)
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
            t_fac_tight[j] = _time_to_tight_sorted(distances, remaining_cost, t)
        
        #check t for next event 1 - client becomes tight
        t_cli_tight = np.inf
        open_j = np.where(y)[0]
        if open_j.size > 0:
            unserved = np.where(~client_served)[0]
            if unserved.size > 0:
                min_dists = np.min(dist_w[np.ix_(unserved, open_j)], axis=1) #calculate minimum distances between unserved clients and open facilities
                future = min_dists > t
                if np.any(future):
                    t_cli_tight = float(np.min(min_dists[future]))

        #move to the nearest time when either facility or client becomes tight
        t_next = min(float(np.min(t_fac_tight)), t_cli_tight)

        if not np.isfinite(t_next):
            break
        t = t_next

        #freeze the newly tight clients in event 1
        if open_j.size > 0:
            for i in np.where(~client_served)[0]:
                d_to_open = dist_w[i, open_j] 
                min_d = np.min(d_to_open)
                if min_d <= t + eps:
                    alpha[i] = min_d #freeze the client when it would become tight
                    client_served[i] = True

        newly_tight_j = np.where(t_fac_tight <= t + eps)[0] #index of facility that became tight
        if newly_tight_j.size == 0 or np.all(y[newly_tight_j]): #do not create a trace for assignign a client
            pass
        else: #create treace only if an facility was opened
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

    if with_traces:
        return JVTrace(
            alpha=np.array(alpha_traces), #[T, n_cli]
            beta=np.array(beta_traces), #[T, n_cli, n_fac]
            assignments=np.array(x_assignment_traces), #[T, n_cli]
            open_facilities=np.array(y_traces), #[T, n_fac]
            deltas=deltas, #[T,]
            client_served=np.array(client_served_trace), #[T, n_cli]
            final_solution=solution,
        )
    else:
        return solution