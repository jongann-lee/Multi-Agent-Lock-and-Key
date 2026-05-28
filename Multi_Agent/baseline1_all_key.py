"""
Baseline 1: every agent carries every key (so no target is ever locked out for
anyone), and agent-to-target assignment uses the Hungarian algorithm on a
plain shortest-path-cost matrix.

This is a deliberately weak comparison point against the main policy in
finite_horizon_MA.py:
  * No visibility / observation reward — the planner sees `distance` only.
    Scoring a pair is just `shortest_path_distance(agent, target) *
    agent.cost_multiplier`. Paths that happen to cover useful observation
    edges get no credit for it.
  * No submodular-aware reassignment — Hungarian assumes per-pair costs are
    independent, which is fine here since the cost is independent.
  * No lock-distribution problem — every agent carries every key, so the
    planner can ignore locks entirely (after the simulation tells it which
    keys exist). Carrying every key inflates the per-edge traversal cost
    (= (1 + num_keys) * distance), which is the explicit knob this baseline
    is meant to expose.

Layout mirrors finite_horizon_MA.py: imports → constants → Agent (re-imported)
→ _score_pair → hungarian_assignment → default_agent_keys + replan (the
uniform policy interface the simulation driver dispatches against).
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from Multi_Agent.finite_horizon_MA import Agent, UNKNOWN_LOCK


# Sentinel cost for (agent, target) pairs that are disconnected or
# lock-infeasible. Large enough that any feasible assignment outranks an
# infeasible one, finite so linear_sum_assignment still has a defined matrix.
UNREACHABLE_COST = 1e9


def default_agent_keys(keys, num_agents):
    """Baseline-1 key distribution: every agent receives `keys` verbatim.

    The caller (multi_agent_simulation.py) tells us at startup which keys
    exist — typically the full list of real lock IDs produced by the
    target-lock permutation. We just hand the same copy to each agent.
    """
    return [list(keys) for _ in range(num_agents)]


def _score_pair(state, agent, target):
    """Cost-only scoring: shortest-path distance scaled by the agent's
    key-count cost multiplier. Returns (cost, path); lower is better.

    Returns (UNREACHABLE_COST, None) if the target is disconnected from
    agent.position or if the target's lock is already known and the agent
    lacks the matching key (the latter shouldn't happen under baseline-1's
    "every agent has every key" default, but the check costs nothing and
    guards against caller overrides).

    Unlike finite_horizon_MA._score_pair this does NOT consider visibility
    reward, diverse-path sampling, or any other reward-shaping — that's the
    whole point of the baseline.
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
    return {int(r): targets[c] for r, c in zip(row_ind, col_ind)}


def replan(env_map: nx.Graph, agents, reward_ratio=None,
           obs_discount_factor=None, sample_recursion=None,
           sample_num_obstacle=None, sample_obstacle_hop=None,
           verbose: bool = False):
    """Baseline-1 replan entry point.

    Clears every agent's planned_path, runs hungarian_assignment, and writes
    a fresh weighted shortest path from each assigned agent's position to
    its target.

    The reward / sampling kwargs (reward_ratio, obs_discount_factor,
    sample_recursion, sample_num_obstacle, sample_obstacle_hop) are accepted
    purely for signature compatibility with the policy dispatch in
    multi_agent_simulation.py — this baseline minimizes only shortest-path
    traversal cost and never considers observation reward.
    """
    for agent in agents:
        agent.planned_path = []

    assignment = hungarian_assignment(env_map, agents)

    if verbose:
        print("=" * 60)
        print("baseline1_all_key Hungarian assignment (cost-only)")
        for i, t in assignment.items():
            print(f"  Agent {i} -> Target @ {t}")
        print("=" * 60)

    for i, target in assignment.items():
        try:
            agents[i].planned_path = nx.shortest_path(
                env_map, agents[i].position, target, weight="distance"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            agents[i].planned_path = []
