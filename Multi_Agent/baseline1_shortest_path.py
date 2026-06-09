"""
Baseline (fixed-key formulation): assign agents to targets purely by
traversal cost — shortest path + Hungarian — respecting lock/key feasibility.

This is the simple comparison point for the fixed agent-key formulation, where
each agent is permanently bound to its key set at the start of the simulation
(for now, a single key each). Unlike the main policy in finite_horizon_MA.py,
this baseline:

  * Scores a pair as `shortest_path_distance(agent, target) *
    agent.cost_multiplier` — pure traversal cost, no visibility / observation
    reward and no diverse-path sampling.
  * Assigns via one Hungarian solve (linear_sum_assignment) minimizing total
    cost. No submodular-aware reassignment — fine here, since per-pair cost
    is genuinely independent.
  * Respects lock/key feasibility under partial observability. A target whose
    lock is still UNKNOWN_LOCK is treated optimistically as feasible (the
    planner heads toward it and finds out on arrival); a target whose lock is
    already revealed is only assignable to an agent that holds the matching
    key. Because keys are fixed and single, each agent's only ultimately
    feasible target is the one whose lock matches its key — but until that
    lock is observed the agent just chases the cheapest unknown target, which
    is exactly the naive behavior this baseline is meant to capture.

Key distribution is NOT decided here — the environment (the simulation driver)
fixes agent i to key i and passes the resulting `agent_keys` in. This module
only provides the planning entry point (`replan`).

Layout mirrors finite_horizon_MA.py: imports → constants → _score_pair →
hungarian_assignment → replan (the policy entry point the simulation driver
dispatches against).
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from Multi_Agent.finite_horizon_MA import UNKNOWN_LOCK


# Sentinel cost for (agent, target) pairs that are disconnected or
# lock-infeasible. Large enough that any feasible assignment outranks an
# infeasible one, finite so linear_sum_assignment still has a defined matrix.
UNREACHABLE_COST = 1e9


def _score_pair(state, agent, target):
    """Cost-only scoring: shortest-path distance scaled by the agent's
    cost multiplier. Returns (cost, path); lower is better.

    Returns (UNREACHABLE_COST, None) if the target is disconnected from
    agent.position, or if the target's lock is already known (revealed) and
    the agent lacks the matching key. A target whose lock is still
    UNKNOWN_LOCK is treated optimistically as feasible.
    """
    target_lock = state.nodes[target].get("lock")
    if (target_lock is not None
            and target_lock != UNKNOWN_LOCK
            and not agent.has_key(target_lock)):
        return UNREACHABLE_COST, None

    try:
        path = nx.shortest_path(
            state, agent.position, target, weight="distance"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return UNREACHABLE_COST, None

    distance = sum(
        state.edges[path[k], path[k + 1]]["distance"]
        for k in range(len(path) - 1)
    )
    return distance * agent.cost_multiplier, path


def hungarian_assignment(env_map: nx.Graph, agents) -> dict:
    """One-step bipartite assignment minimizing total traversal cost.

    Builds an (n_agents x n_targets) cost matrix via _score_pair and runs
    linear_sum_assignment in its default minimization mode. Unreachable or
    lock-infeasible pairs get UNREACHABLE_COST so the matrix is well-defined.

    Returns {agent_idx: target_node}.
    """
    targets = [
        n for n, d in env_map.nodes(data=True) if d.get("type") == "target_unreached"
    ]
    if not targets:
        return {}

    n_agents = len(agents)
    n_targets = len(targets)
    cost = np.full((n_agents, n_targets), UNREACHABLE_COST, dtype=float)

    for i, agent in enumerate(agents):
        for j, target in enumerate(targets):
            c, _path = _score_pair(env_map, agent, target)
            cost[i, j] = c

    row_ind, col_ind = linear_sum_assignment(cost)
    # Drop assignments that landed on an infeasible cell — with fewer feasible
    # targets than agents, Hungarian still pads the matching with
    # UNREACHABLE_COST pairs; those agents simply get no target this round.
    return {
        int(r): targets[c]
        for r, c in zip(row_ind, col_ind)
        if cost[r, c] < UNREACHABLE_COST
    }


def replan(env_map: nx.Graph, agents, edge_reward_ratio=None,
           obs_discount_factor=None, sample_recursion=None,
           sample_num_obstacle=None, sample_obstacle_hop=None,
           target_reward_ratio=None, verbose: bool = False):
    """Baseline replan entry point.

    Clears every agent's planned_path, runs hungarian_assignment, and writes
    a fresh weighted shortest path from each assigned agent's position to its
    target.

    The reward / sampling / target kwargs are accepted purely for signature
    compatibility with the policy dispatch in multi_agent_simulation.py —
    this baseline minimizes only shortest-path traversal cost and never
    considers observation reward.
    """
    for agent in agents:
        agent.planned_path = []

    assignment = hungarian_assignment(env_map, agents)

    if verbose:
        print("=" * 60)
        print("baseline1_shortest_path Hungarian assignment (cost-only)")
        for i, t in assignment.items():
            print(f"  Agent {i} (keys={agents[i].possessed_keys}) -> Target @ {t}")
        print("=" * 60)

    for i, target in assignment.items():
        try:
            agents[i].planned_path = nx.shortest_path(
                env_map, agents[i].position, target, weight="distance"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            agents[i].planned_path = []
