"""
Baseline planner: shortest-path tracking with Hungarian assignment.

A deliberately simple comparison point for the main reward-maximizing planner
(finite_horizon_MA.sequential_greedy_assignment via simulation_utils._replan).
It ignores visibility *reward* and diverse-path sampling entirely: agents take
plain weighted shortest paths toward a goal, and the agent->target matching is a
one-shot Hungarian assignment on those shortest-path distances.

Goal selection per target (the only subtlety):
  * Target NOT visible: the planner knows only its current position, so that is
    the goal — a pure chase of the last-known location.
  * Target visible: the engagement rule also reveals the target's next few
    planned steps, so the agent aims *ahead* to intercept rather than chase its
    tail. The target is at planned[k-1] after k turns, so the agent picks the
    earliest of those steps it can reach in <= k turns — the soonest cell where
    it is guaranteed to meet (or beat) the target. A far agent ends up at the
    furthest visible step (e.g. the 3rd); a near agent cuts the corner to the
    closest interceptable step. Choosing a cell it can actually reach in time —
    instead of always "where the target will be in `distance` steps" — is what
    avoids perpetually trailing one step behind at close range.

Because the targets move, this is meant to be re-run every turn (the simulation
driver replans each turn under this policy); each replan re-evaluates the goals
against the targets' current positions / revealed plans.

Structure mirrors finite_horizon_MA.py: imports -> constants -> goal/cost
helper -> hungarian_assignment -> replan (the entry point the driver calls).
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment


# Sentinel cost for (agent, target) pairs that are disconnected on the planner's
# map. Large enough that any feasible assignment outranks an infeasible one, but
# finite so linear_sum_assignment still has a well-defined matrix.
UNREACHABLE_COST = 1e9


def _goal_and_path(env_map, agent, target, visible, max_visible_steps):
    """Goal node + weighted shortest path for `agent` pursuing `target`.

    Returns (path, cost). `path` is a node list (agent.position .. goal); `cost`
    is its weighted length scaled by the agent's cost_multiplier. Returns
    (None, UNREACHABLE_COST) if the target's current position is unreachable.

    Invisible target -> goal is its current position (chase last-known).

    Visible target -> interception. The target is at planned[k-1] after k turns,
    so to guarantee a catch the agent must reach that cell in <= k turns. We aim
    at the EARLIEST such reachable step (smallest k in 1..horizon with
    path_hops(agent, planned[k-1]) <= k). Reaching it early just means the agent
    waits there for the target to walk in. If no step is reachable in time (the
    target is outrunning us within the horizon), we chase the furthest visible
    step as best effort.

    Aiming at a cell we can reach in time — rather than always at planned[k-1]
    for k = distance — is what stops the close-range tail-chase: a far agent
    still ends up at the 3rd step, but a near agent cuts the corner to whichever
    upcoming cell it can actually meet the target on.
    """
    if not visible or not target.planned:
        try:
            path = nx.shortest_path(
                env_map, agent.position, target.position, weight="distance"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, UNREACHABLE_COST
    else:
        horizon = min(max_visible_steps, len(target.planned))
        path = None
        last_reachable = None
        for k in range(1, horizon + 1):
            try:
                cand = nx.shortest_path(
                    env_map, agent.position, target.planned[k - 1], weight="distance"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            last_reachable = cand
            if len(cand) - 1 <= k:        # at planned[k-1] by the time the target is
                path = cand
                break
        if path is None:
            # Can't intercept within the horizon: chase the furthest step we can
            # still path to (falling back to the current position if need be).
            if last_reachable is not None:
                path = last_reachable
            else:
                try:
                    path = nx.shortest_path(
                        env_map, agent.position, target.position, weight="distance"
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    return None, UNREACHABLE_COST

    distance = sum(
        env_map.edges[path[i], path[i + 1]]["distance"]
        for i in range(len(path) - 1)
    )
    return path, distance * agent.cost_multiplier


def hungarian_assignment(env_map, agents, targets, visible_nodes,
                         max_visible_steps=3) -> dict:
    """One-shot bipartite assignment minimizing total shortest-path cost.

    Builds an (n_agents x n_unreached_targets) cost matrix of weighted
    shortest-path distances to each target's goal (current position, or an
    interception lead point for visible targets) and runs linear_sum_assignment.

    Args:
        env_map: planner's view of the graph (paths are planned on this).
        agents: list of Agent objects.
        targets: list of Target objects; reached targets are skipped.
        visible_nodes: set of nodes currently within some agent's line of sight
            (see simulation_utils.agent_visible_nodes). A target is "visible"
            iff its position is in this set.
        max_visible_steps: how far ahead the planner may aim for a visible target.

    Returns {agent_idx: (target, path)}, where `path` is the weighted shortest
    path the agent should follow. Same tuple shape as the greedy assignment.
    """
    pending = [t for t in targets if not t.reached]
    if not pending or not agents:
        return {}

    n_agents, n_targets = len(agents), len(pending)
    cost = np.full((n_agents, n_targets), UNREACHABLE_COST, dtype=float)
    paths: dict = {}

    for i, agent in enumerate(agents):
        for j, target in enumerate(pending):
            visible = target.position in visible_nodes
            path, c = _goal_and_path(env_map, agent, target, visible, max_visible_steps)
            cost[i, j] = c
            paths[(i, j)] = (target, path)

    row_ind, col_ind = linear_sum_assignment(cost)
    return {int(r): paths[(int(r), int(c))] for r, c in zip(row_ind, col_ind)}


def replan(env_map, agents, targets, visible_nodes, max_visible_steps=3,
           verbose=False):
    """Baseline replan entry point.

    Clears every agent's planned_path, runs the Hungarian shortest-path
    assignment, and routes each assigned agent down the plain weighted shortest
    path to its goal.
    """
    for agent in agents:
        agent.planned_path = []

    assignment = hungarian_assignment(
        env_map, agents, targets, visible_nodes, max_visible_steps
    )

    if verbose:
        print("=" * 60)
        print("baseline1_naive_tracking: Hungarian shortest-path assignment")
        for i, (target, path) in assignment.items():
            goal = path[-1] if path else None
            vis = "visible" if target.position in visible_nodes else "hidden"
            print(f"  Agent {i} ({agents[i].position}) -> target {target.id} "
                  f"@ {target.position} [{vis}], goal {goal}")
        print("=" * 60)

    for i, (target, path) in assignment.items():
        agents[i].planned_path = list(path) if path else []
