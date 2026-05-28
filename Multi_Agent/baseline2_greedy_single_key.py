"""
Baseline 2: each agent carries exactly one key at a time and shuttles back to
the source between unlocks to swap it for a different one.

Key lifecycle
-------------
The lock does NOT eat the key. When an agent steps onto its matching target
the simulation marks the target as `target_reached` (existing mechanic), but
the key stays in the agent's hand the whole time. The agent then walks back
to source carrying it, deposits the (now-spent) key into the source pool,
and pulls a different key out. So:

  * `possessed_keys` always has 0 or 1 entries.
  * "Spent" is not a flag — it's derived: an agent's key is spent iff its
    lock id appears among the targets currently marked `target_reached`.
    This is the planner-visible signal for "do not assign me, route me
    home."
  * The source pool is the union of fresh + spent keys still at base. On
    pickup we filter for the first key whose lock is NOT in `reached_locks`.
    No fresh key available → agent ends up with empty hands and idles at
    source (other agents are finishing whatever's left).

Initialization
--------------
`default_agent_keys` shuffles the full key list, hands one distinct key to
each agent, and stashes the rest in the module-level source pool.

Per-replan logic (in order)
---------------------------
1. **At-source swap.**  Every spent agent currently at source returns its
   key to the pool and pulls the next fresh key. If no fresh key is
   available the agent winds up empty-handed.
2. **Tier-1 forced matches.**  For every target_unreached whose lock is
   already known and equals some fresh-key agent's key, force-pair them.
3. **Tier-2 Hungarian on unknowns.**  Remaining fresh-key agents are
   Hungarian-matched to unknown-lock targets by cost-only scoring
   (shortest-path distance × cost_multiplier, like baseline1). They scout
   in the hope of revealing a match.
4. **Routing.**  Assigned agents → shortest path to target. Spent agents
   not at source → shortest path home. Empty-handed agents and agents
   already at source after a swap with no plan → empty plan (idle).
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import random

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from Multi_Agent.finite_horizon_MA import Agent, UNKNOWN_LOCK


# Same sentinel idea as baseline1: large finite cost for infeasible pairs so
# linear_sum_assignment still has a defined matrix.
UNREACHABLE_COST = 1e9

# Per-run mutable state. The simulation runs sequentially, so a module-level
# variable is enough and dramatically simpler than threading a context
# object through the policy dispatch.
_source_pool: list = []


def default_agent_keys(keys, num_agents):
    """Baseline-2 key distribution: each agent gets ONE distinct random key.

    Side effect: stores the remaining keys in the module-level `_source_pool`
    in random shuffle order. Raises ValueError if there aren't enough
    distinct keys for every agent.
    """
    global _source_pool
    if num_agents > len(keys):
        raise ValueError(
            f"baseline2 needs at least num_agents={num_agents} distinct keys "
            f"but only {len(keys)} are available."
        )
    shuffled = list(keys)
    random.shuffle(shuffled)
    initial = [[shuffled[i]] for i in range(num_agents)]
    _source_pool = shuffled[num_agents:]
    return initial


def _find_source(env_map):
    """Return the source node coordinate. There's exactly one in our setup."""
    for n, d in env_map.nodes(data=True):
        if d.get("type") == "source":
            return n
    return None


def _reached_locks(env_map):
    """Set of lock IDs whose targets have already been reached.

    This is the planner-visible "done list" — an agent whose only key
    matches one of these is on the walk back to source.
    """
    return {
        env_map.nodes[t]["lock"]
        for t, d in env_map.nodes(data=True)
        if d.get("type") == "target_reached"
        and d.get("lock", UNKNOWN_LOCK) != UNKNOWN_LOCK
    }


def _path_cost(env_map, source_node, target_node, cost_multiplier):
    """Shortest-path distance × cost_multiplier. Returns (cost, path).

    (UNREACHABLE_COST, None) if disconnected.
    """
    try:
        path = nx.shortest_path(
            env_map, source_node, target_node, weight="distance"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return UNREACHABLE_COST, None
    distance = sum(
        env_map.edges[path[k], path[k + 1]]["distance"]
        for k in range(len(path) - 1)
    )
    return distance * cost_multiplier, path


def replan(env_map: nx.Graph, agents, reward_ratio=None,
           obs_discount_factor=None, sample_recursion=None,
           sample_num_obstacle=None, sample_obstacle_hop=None,
           verbose: bool = False):
    """Baseline-2 replan entry point.

    Reward / sampling kwargs are accepted for signature compatibility with
    the policy dispatch but ignored — this baseline is cost-only.
    """
    global _source_pool

    source_node = _find_source(env_map)
    reached_locks = _reached_locks(env_map)

    def is_spent(agent):
        """An agent's key is spent iff it matches an already-reached target."""
        return bool(agent.possessed_keys) and agent.possessed_keys[0] in reached_locks

    # --- Phase 1: at-source swap for spent agents ---
    # Push the spent key back into the pool, scan the pool for the first
    # fresh key (i.e. lock not yet reached), and pull it. If nothing fresh
    # remains we leave the agent empty-handed — pool's full of spent keys
    # and any remaining unreached locks are with other agents.
    for agent in agents:
        if agent.position != source_node:
            continue
        if not is_spent(agent):
            continue
        spent_key = agent.possessed_keys[0]
        _source_pool.append(spent_key)
        agent.possessed_keys = []
        for i, k in enumerate(_source_pool):
            if k not in reached_locks:
                fresh = _source_pool.pop(i)
                agent.possessed_keys = [fresh]
                if verbose:
                    print(f"  [swap] Agent at source swapped key "
                          f"{spent_key} -> {fresh} (pool: {_source_pool}).")
                break
        else:
            if verbose:
                print(f"  [drop] Agent at source dropped spent key "
                      f"{spent_key}; no fresh keys remain (pool: "
                      f"{_source_pool}). Idle.")

    # --- Phase 2: Tier-1 forced known-match assignment ---
    targets = [
        n for n, d in env_map.nodes(data=True) if d.get("type") == "target_unreached"
    ]
    known_target_by_lock = {
        env_map.nodes[t]["lock"]: t
        for t in targets
        if env_map.nodes[t].get("lock", UNKNOWN_LOCK) != UNKNOWN_LOCK
    }

    assignment: dict[int, object] = {}
    matched_targets: set = set()
    for i, agent in enumerate(agents):
        if not agent.possessed_keys or is_spent(agent):
            continue
        key = agent.possessed_keys[0]
        t = known_target_by_lock.get(key)
        if t is not None and t not in matched_targets:
            assignment[i] = t
            matched_targets.add(t)

    if verbose and assignment:
        for i, t in assignment.items():
            print(f"  [tier-1] Agent {i} (key={agents[i].possessed_keys[0]}) "
                  f"-> Target {t} (matching lock)")

    # --- Phase 3: Tier-2 Hungarian on remaining fresh-key agents → unknowns ---
    remaining_agents = [
        i for i, a in enumerate(agents)
        if i not in assignment and a.possessed_keys and not is_spent(a)
    ]
    unknown_targets = [
        t for t in targets
        if t not in matched_targets
        and env_map.nodes[t].get("lock", UNKNOWN_LOCK) == UNKNOWN_LOCK
    ]

    if remaining_agents and unknown_targets:
        cost = np.full(
            (len(remaining_agents), len(unknown_targets)),
            UNREACHABLE_COST, dtype=float,
        )
        for r, ai in enumerate(remaining_agents):
            for c, t in enumerate(unknown_targets):
                cost[r, c], _ = _path_cost(
                    env_map, agents[ai].position, t,
                    agents[ai].cost_multiplier,
                )
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < UNREACHABLE_COST:
                ai = remaining_agents[r]
                assignment[ai] = unknown_targets[c]
                if verbose:
                    print(f"  [tier-2] Agent {ai} "
                          f"(key={agents[ai].possessed_keys[0]}) -> "
                          f"Target {unknown_targets[c]} (unknown lock, "
                          f"cost {cost[r, c]:.2f})")

    # --- Phase 4: write planned paths ---
    for agent in agents:
        agent.planned_path = []

    # Assigned (fresh-key) agents → path to assigned target.
    for i, target in assignment.items():
        _, path = _path_cost(
            env_map, agents[i].position, target, agents[i].cost_multiplier,
        )
        agents[i].planned_path = path or []

    # Spent agents not at source → path home. (Spent agents AT source were
    # handled in Phase 1.)
    for i, agent in enumerate(agents):
        if not is_spent(agent):
            continue
        if agent.position == source_node:
            continue
        _, path = _path_cost(env_map, agent.position, source_node,
                             agent.cost_multiplier)
        agent.planned_path = path or []
        if verbose:
            print(f"  [route-home] Agent {i} routed back to source "
                  f"({len(agent.planned_path)} steps).")

    if verbose:
        print("=" * 60)
