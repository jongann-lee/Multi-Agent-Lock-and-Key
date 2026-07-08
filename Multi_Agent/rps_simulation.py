"""
Map-agnostic Rock-Scissor-Paper simulation core.

This is the turn-based driver for the `rock_scissor_paper` direction. It is
deliberately decoupled from the real-DEM benchmark so the mechanics can be
exercised on a tiny synthetic graph in a fast unit test (see test_rps.py). The
real-map entry point (rps_real_map.py) just builds the two graphs and hands
them here.

Two-graph partial observability (same contract as the base multi-agent sim):
  * ``env_map``      -- the planner's view. Targets start with
                        ``rps_type = UNKNOWN_TYPE``; the planner only learns a
                        target's type by *observing* it.
  * ``ground_truth`` -- reality: true edge set (obstacles removed), true
                        ``rps_type`` on every target, and ``visible_edges``
                        pruned to the true edges.

What this module adds on top of the base loop:
  1. **Heterogeneous visibility by agent type.** A SCOUT senses its full
     polytope ``visible_edges`` and is the ONLY observer. Combat agents
     (rock/scissor/paper) are **blind** -- they sense nothing, so they neither
     discover blockages nor reveal target types by looking. An attacker learns
     a target's type only by stepping onto it (combat always reveals).
  2. **Combat on contact.** When an agent steps onto a live target the
     encounter is resolved by rps.resolve_encounter: a winning combat agent
     eliminates the target, a losing one dies, a draw leaves both in place,
     and a scout always dies. Every encounter reveals the target's type.
  3. **Deaths.** A dead agent is removed from play for the rest of the run.

The assignment policy is pluggable: pass any ``policy(env_map, agents, ...)``
that sets each living agent's ``planned_path``. ``naive_type_aware_replan``
below is a minimal SAFE placeholder so the loop runs; replace it with the real
baseline.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import networkx as nx

from Multi_Agent.rps import (
    SCOUT, UNKNOWN_TYPE, DRAW, AGENT_WINS, AGENT_DIES,
    beats, resolve_encounter, TYPE_NAMES,
)


# ---------------------------------------------------------------------------
# Edge / path helpers (direction-agnostic; the real graph is a DiGraph with
# both orientations present for every grid adjacency).
# ---------------------------------------------------------------------------

def _ekey(u, v):
    """Direction-agnostic key for the edge between ``u`` and ``v``."""
    return frozenset((u, v))


def _path_distance(graph, path):
    """Sum of edge 'distance' along ``path`` (missing edges skipped)."""
    return sum(
        graph.edges[path[k], path[k + 1]]["distance"]
        for k in range(len(path) - 1)
        if graph.has_edge(path[k], path[k + 1])
    )


def _path_hits_blocked(path, blocked_keys):
    """True if any consecutive pair in ``path`` is a newly blocked edge."""
    return any(
        _ekey(path[k], path[k + 1]) in blocked_keys
        for k in range(len(path) - 1)
    )


# ---------------------------------------------------------------------------
# Type-aware sensing.
# ---------------------------------------------------------------------------

def sensed_edges_truth(ground_truth, agent):
    """Edges the agent ACTUALLY senses from its position, by type.

    SCOUT  -> full polytope ``visible_edges`` (terrain-limited, long range).
    combat -> nothing: attackers are BLIND. They learn a target's type only by
              engaging it (combat always reveals), never by looking.
    """
    if agent.agent_type != SCOUT:
        return []
    pos = agent.position
    return [e for e in ground_truth.nodes[pos].get("visible_edges", [])
            if ground_truth.has_edge(*e)]


def sensed_edges_assumed(env_map, agent):
    """Edges the planner ASSUMES the agent senses. Scout only -- attackers are
    blind, so they sense nothing and discover no blockages themselves."""
    if agent.agent_type != SCOUT:
        return []
    pos = agent.position
    return list(env_map.nodes[pos].get("visible_edges", []))


def sensed_nodes_truth(ground_truth, agent):
    """Nodes the agent can see (endpoints of its truly-sensed edges + self)."""
    nodes = {agent.position}
    for u, v in sensed_edges_truth(ground_truth, agent):
        nodes.add(u)
        nodes.add(v)
    return nodes


# ---------------------------------------------------------------------------
# Observation: blockage discovery + target-type revelation.
# ---------------------------------------------------------------------------

def observe_and_reveal(env_map, ground_truth, agents, events=None, turn=0,
                       verbose=False):
    """Each living agent observes from its position.

    Mutates ``env_map``: marks sensed edges observed, removes newly discovered
    blocked edges, and reveals the ``rps_type`` of any target the agent can
    see. Returns ``(newly_blocked_edge_keys, num_types_revealed)`` -- both are
    replan triggers (a freshly revealed type can let an idle combat agent act).
    """
    newly_blocked = set()
    revealed = 0

    for idx, agent in enumerate(agents):
        if not agent.alive:
            continue

        observable = {_ekey(*e) for e in sensed_edges_truth(ground_truth, agent)}
        assumed_edges = sensed_edges_assumed(env_map, agent)
        assumed = {_ekey(*e) for e in assumed_edges}
        newly_blocked |= (assumed - observable)

        # The agent has now looked at every edge in its footprint.
        for e in assumed_edges:
            if env_map.has_edge(*e):
                env_map.edges[e]["observed_edge"] = True

        # Reveal the type of any target within sight.
        for node in sensed_nodes_truth(ground_truth, agent):
            data = env_map.nodes.get(node, {})
            if data.get("type") not in ("target_unreached", "target_reached"):
                continue
            if data.get("rps_type", UNKNOWN_TYPE) != UNKNOWN_TYPE:
                continue
            true_type = ground_truth.nodes[node].get("rps_type", UNKNOWN_TYPE)
            env_map.nodes[node]["rps_type"] = true_type
            revealed += 1
            if events is not None:
                events.append({"turn": turn, "event": "reveal", "agent": idx,
                               "node": node, "rps_type": true_type})
            if verbose:
                print(f"  [reveal] agent {idx} ({TYPE_NAMES[agent.agent_type]}) "
                      f"saw target {node} -> {TYPE_NAMES[true_type]}")

    # Apply the blockages to the planner's map.
    if newly_blocked:
        for key in newly_blocked:
            u, v = tuple(key)
            for e in ((u, v), (v, u)):
                if env_map.has_edge(*e):
                    env_map.remove_edge(*e)
        for node in env_map.nodes():
            ve = env_map.nodes[node].get("visible_edges")
            if ve is not None:
                env_map.nodes[node]["visible_edges"] = [
                    e for e in ve if _ekey(*e) not in newly_blocked
                ]

    return newly_blocked, revealed


# ---------------------------------------------------------------------------
# Combat resolution (transition-triggered: an encounter happens the instant an
# agent steps onto a live target).
# ---------------------------------------------------------------------------

def resolve_combat_on_arrival(env_map, ground_truth, agent, agent_idx, node,
                              events=None, turn=0, verbose=False):
    """Resolve an encounter for ``agent`` that just stepped onto live ``node``.

    Always reveals the target's type. Returns the outcome string
    (rps.DRAW / AGENT_WINS / AGENT_DIES) and mutates state:
      * AGENT_WINS -> node becomes 'target_reached' (eliminated).
      * AGENT_DIES -> agent.alive = False.
      * DRAW       -> no state change beyond revealing the type.
    """
    true_type = ground_truth.nodes[node].get("rps_type", UNKNOWN_TYPE)
    env_map.nodes[node]["rps_type"] = true_type  # an encounter always reveals
    outcome = resolve_encounter(agent.agent_type, true_type)

    if outcome == AGENT_WINS:
        env_map.nodes[node]["type"] = "target_reached"
    elif outcome == AGENT_DIES:
        agent.alive = False

    if events is not None:
        events.append({"turn": turn, "event": outcome, "agent": agent_idx,
                       "node": node, "agent_type": agent.agent_type,
                       "rps_type": true_type})
    if verbose:
        print(f"  [combat] agent {agent_idx} ({TYPE_NAMES[agent.agent_type]}) "
              f"vs target {node} ({TYPE_NAMES[true_type]}) -> {outcome}")
    return outcome


# ---------------------------------------------------------------------------
# Placeholder policy. SAFE and simple, NOT optimized -- replace with the
# real baseline. It never sends a combat agent at a target it can't beat and
# never routes a scout onto a target (which would kill it).
# ---------------------------------------------------------------------------

def _safe_path(env_map, src, goal, avoid):
    """Shortest ``src``->``goal`` path that avoids the ``avoid`` nodes.

    Routes on a copy of ``env_map`` with ``avoid`` removed (never removing
    ``src`` or ``goal``) so the path never crosses an unwanted target node.
    Returns ``(path, cost)`` or ``(None, inf)`` if unreachable.
    """
    if src == goal:
        return [src], 0.0
    graph = env_map
    avoid = [n for n in avoid if n != src and n != goal]
    if avoid:
        graph = env_map.copy()
        graph.remove_nodes_from(avoid)
    try:
        path = nx.shortest_path(graph, src, goal, weight="distance")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, float("inf")
    return path, _path_distance(graph, path)


def naive_type_aware_replan(env_map, agents, reward_ratio=1.0,
                            obs_discount_factor=1.0, sample_recursion=0,
                            sample_num_obstacle=0, sample_obstacle_hop=0,
                            verbose=False):
    """Minimal type-aware assignment so the loop runs end-to-end.

    * Combat agents: claim the nearest REVEALED target they beat, routing
      AROUND every other live target (never walking through an enemy they
      can't fight). If none beatable, idle and wait for a scout to reveal.
    * Scouts: approach the nearest UNKNOWN-type target to reveal it, also
      routing around every live target and stopping one node short so the
      scout never steps onto a target (which would kill it). If nothing is
      unknown, idle.

    The reward / sampling kwargs are accepted for interface compatibility with
    reward-driven policies but ignored here (this stub scores by distance).
    """
    for a in agents:
        a.planned_path = []

    live_targets = [n for n, d in env_map.nodes(data=True)
                    if d.get("type") == "target_unreached"]
    if not live_targets:
        return
    live_set = set(live_targets)

    # --- combat agents -> nearest beatable revealed target ---
    claimed = set()
    for a in agents:
        if not a.alive or a.agent_type == SCOUT:
            continue
        best_t, best_path, best_cost = None, None, float("inf")
        for t in live_targets:
            if t in claimed:
                continue
            tt = env_map.nodes[t].get("rps_type", UNKNOWN_TYPE)
            if tt == UNKNOWN_TYPE or not beats(a.agent_type, tt):
                continue
            path, cost = _safe_path(env_map, a.position, t, live_set - {t})
            if path is not None and cost < best_cost:
                best_t, best_path, best_cost = t, path, cost
        if best_t is not None:
            a.planned_path = best_path
            claimed.add(best_t)
            if verbose:
                print(f"  [assign] {TYPE_NAMES[a.agent_type]} @ {a.position} "
                      f"-> beat target {best_t} "
                      f"({TYPE_NAMES[env_map.nodes[best_t]['rps_type']]})")

    unknown = [t for t in live_targets
               if env_map.nodes[t].get("rps_type", UNKNOWN_TYPE) == UNKNOWN_TYPE]

    def approach_unknown(a):
        """Route ``a`` to one node short of the nearest unknown target, going
        around all live targets so it reveals the type without stepping on
        (and dying to) any target. Returns True if a path was set."""
        best_path, best_cost = None, float("inf")
        for goal in unknown:
            path, cost = _safe_path(env_map, a.position, goal, live_set - {goal})
            if path is not None and cost < best_cost:
                best_path, best_cost = path, cost
        if best_path is not None and len(best_path) >= 2:
            a.planned_path = best_path[:-1]  # stop before stepping onto target
            if verbose:
                print(f"  [assign] {TYPE_NAMES[a.agent_type]} @ {a.position} "
                      f"-> approach {best_path[-1]} (stop at {best_path[-2]})")
            return True
        return False

    # --- scouts -> approach nearest unknown target to reveal it ---
    for a in agents:
        if a.alive and a.agent_type == SCOUT and unknown:
            approach_unknown(a)

    # NOTE: attackers are blind, so only a scout can reveal a target's type
    # without an encounter. This stub therefore needs a scout in the roster to
    # make progress safely; with none, its combat agents idle on unknown-only
    # maps. (baseline1_all_key instead has blind attackers charge unknowns and
    # accept the gamble.)


# ---------------------------------------------------------------------------
# Main loop.
# ---------------------------------------------------------------------------

def run_rps_simulation(env_map, ground_truth, agents, policy=None,
                       reward_ratio=1.0, obs_discount_factor=1.0,
                       sample_recursion=0, sample_num_obstacle=0,
                       sample_obstacle_hop=0, max_turns=2000, verbose=False,
                       render_dir=None):
    """Run the RPS simulation until all targets are cleared or it stalls.

    Args:
        env_map: planner view (copied internally; targets carry UNKNOWN types).
        ground_truth: reality (true edges, true target ``rps_type``).
        agents: list of Agent (typed, all positioned at the source).
        policy: ``policy(env_map, living_agents, **kwargs)`` that sets each
            living agent's ``planned_path``. Defaults to
            :func:`naive_type_aware_replan`.
        max_turns: hard cap against non-terminating realizations.

    Returns a result dict with per-agent costs/survival, turn count,
    completion, remaining/eliminated targets, and the event log.
    """
    policy = policy or naive_type_aware_replan
    env_map = env_map.copy()
    events = []

    def living():
        return [a for a in agents if a.alive]

    def do_replan():
        policy(env_map, living(), reward_ratio=reward_ratio,
               obs_discount_factor=obs_discount_factor,
               sample_recursion=sample_recursion,
               sample_num_obstacle=sample_num_obstacle,
               sample_obstacle_hop=sample_obstacle_hop, verbose=verbose)

    def _maybe_render(idx):
        if render_dir is None:
            return
        from Multi_Agent.simulation_utils import render_rps_frame  # lazy (matplotlib)
        render_rps_frame(env_map, ground_truth, agents, idx,
                         os.path.join(render_dir, f"frame_{idx:04d}.png"))

    do_replan()  # initial plan
    _maybe_render(0)

    turn = 0
    while turn < max_turns:
        live_targets = [n for n, d in env_map.nodes(data=True)
                        if d.get("type") == "target_unreached"]
        if not live_targets:
            break  # success: everything cleared
        if not living():
            break  # all agents dead

        # --- 1. Observe (blockage discovery + type revelation) ---
        newly_blocked, revealed = observe_and_reveal(env_map, ground_truth,
                                                     agents, events, turn, verbose)

        # --- 2. Replan on new information: a freshly revealed target type, or
        #        a discovered blockage that hits some agent's planned path. ---
        replanned = False
        blockage_replan = bool(newly_blocked) and any(
            len(a.planned_path) >= 2 and _path_hits_blocked(a.planned_path, newly_blocked)
            for a in living()
        )
        if revealed or blockage_replan:
            do_replan()
            replanned = True

        # --- 3. Move each living agent; resolve combat on contact ---
        any_progress = False
        combat_happened = False
        for idx, agent in enumerate(agents):
            if not agent.alive:
                continue
            for _ in range(agent.movement_modifier):
                if len(agent.planned_path) < 2:
                    break
                next_node = agent.planned_path[1]
                if not ground_truth.has_edge(agent.position, next_node):
                    break  # edge blocked in reality; wait for replan
                cost = ground_truth.edges[agent.position, next_node]["distance"]
                agent.move(agent.position, next_node, cost)
                any_progress = True
                # Stepping onto a live target triggers an encounter.
                if env_map.nodes[next_node].get("type") == "target_unreached":
                    resolve_combat_on_arrival(env_map, ground_truth, agent, idx,
                                              next_node, events, turn, verbose)
                    combat_happened = True
                    break  # stop this agent's movement for the turn

        # --- 4. Replan after any encounter (win/draw/death all change state) ---
        if combat_happened:
            do_replan()
            replanned = True

        # --- 5. Stall detection ---
        if not any_progress and not replanned:
            break

        turn += 1
        _maybe_render(turn)

    remaining = [n for n, d in env_map.nodes(data=True)
                 if d.get("type") == "target_unreached"]
    eliminated = [n for n, d in env_map.nodes(data=True)
                  if d.get("type") == "target_reached"]
    return {
        "env_map": env_map,
        "agents": agents,
        "turns": turn,
        "completed": not remaining,
        "remaining_targets": remaining,
        "eliminated_targets": eliminated,
        "deaths": [i for i, a in enumerate(agents) if not a.alive],
        "survivors": [i for i, a in enumerate(agents) if a.alive],
        "total_cost": sum(a.total_traversal_cost for a in agents),
        "per_agent_cost": [a.total_traversal_cost for a in agents],
        "events": events,
    }
