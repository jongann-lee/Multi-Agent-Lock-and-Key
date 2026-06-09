"""
Multi-agent benchmark on the real terrain (DEM) map.

Turn-based simulation: each turn an agent observes from its position, the
planner replans if a previously-planned edge is now known to be blocked or
if any agent has reached a target, and then every agent moves one step along
its current planned path. Repeats until all targets are reached.

Assignment between agents and remaining targets uses
Multi_Agent.finite_horizon_assignment (Hungarian on the path-reward matrix).
"""

import sys
import os
import time
import copy
import json
import csv
import pickle
import argparse
import subprocess
import random

import numpy as np
import networkx as nx

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Real_Life_Maps.real_map_generation import RealTerrainGrid
from Graph_Generation.target_graph import create_fully_connected_target_graph
from Multi_Agent.finite_horizon_MA import Agent, UNKNOWN_LOCK
from Multi_Agent.simulation_utils import (
    render_simulation_frame, clear_frame_dir, make_mp4_from_frames,
    render_replan_debug_frame, clear_debug_dir,
)


# ---------------------------------------------------------------------------
# Policy dispatch — each entry is a module that exposes a uniform
# `replan(env_map, agents, ...)` entry point. Keys are assigned by the
# environment (agent i -> key i), not the policy. Add new policies by adding
# the module name to POLICIES.
# ---------------------------------------------------------------------------

POLICIES = ("finite_horizon_MA", "baseline1_shortest_path")


def _load_policy(name):
    """Resolve a `--policy NAME` string to the actual module.

    Kept as an explicit allowlist (rather than `importlib` on arbitrary
    strings) so typos surface as a clean error and the valid set shows up in
    --help and in the error message.
    """
    import importlib
    if name not in POLICIES:
        raise ValueError(
            f"unknown policy {name!r}; expected one of {POLICIES}"
        )
    return importlib.import_module(f"Multi_Agent.{name}")


# ---------------------------------------------------------------------------
# 1. DEM + road loading
# ---------------------------------------------------------------------------

def get_grid_from_local_dem(file_path, n_size):
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(file_path) as dataset:
        data = dataset.read(
            1, out_shape=(n_size, n_size), resampling=Resampling.bilinear
        )
        if dataset.nodata is not None:
            data = np.where(data == dataset.nodata, np.nan, data)
        return data


def load_real_terrain(dem_path, n_size=64):
    height_grid = get_grid_from_local_dem(dem_path, n_size)
    return np.rot90(height_grid, k=-1)


def load_roads(road_pkl):
    if road_pkl is None or not os.path.exists(road_pkl):
        return set(), set()
    with open(road_pkl, "rb") as f:
        data = pickle.load(f)
    return data["road_nodes"], data["road_edges"]


# Default obstacles, mirroring Real_Life_Maps/visualization.ipynb.
DEFAULT_OBSTACLE_SPECS = [
    ((47, 43), 5, 3),
    ((30, 36), 3, 5),
    ((31, 14), 4, 4),
    ((36, 55), 5, 3),
    ((14, 17), 4, 4),
]
# DEFAULT_OBSTACLE_SPECS = []



# ---------------------------------------------------------------------------
# 4. Multi-agent simulation
# ---------------------------------------------------------------------------

def _path_blocked_by(path, blocked_pairs):
    """True if any consecutive pair in `path` is in `blocked_pairs`."""
    for k in range(len(path) - 1):
        if (path[k], path[k + 1]) in blocked_pairs or (path[k + 1], path[k]) in blocked_pairs:
            return True
    return False


def run_multi_agent_simulation(env_graph, blocked_env_graph, source, num_agents,
                               edge_reward_ratio, obs_discount_factor=1.0,
                               sample_recursion=0, sample_num_obstacle=0,
                               sample_obstacle_hop=0, target_reward_ratio=0.0,
                               agent_keys=None,
                               policy="finite_horizon_MA",
                               max_turns=2000, render_dir=None, debug_dir=None):
    """
    Args:
        env_graph: clean planner graph (will be deep-copied for the run).
        blocked_env_graph: ground-truth graph with ovals applied.
        source: starting node for all agents.
        num_agents: number of agents.
        edge_reward_ratio: lambda weighting edge-visibility reward vs distance.
        agent_keys: optional list of length `num_agents`, where each element is
            an iterable of lock IDs that agent carries. If None, falls back to
            the fixed-key rule "agent i carries key i".
        policy: either a name from POLICIES or a policy module already imported
            by the caller. Exposes `replan(env_map, agents, ...)`.
        max_turns: hard cap to avoid infinite loops on unsolvable realizations.
        render_dir: if set, write one PNG per turn to this directory.

    Returns:
        dict with per-agent costs, trajectories, turn count, and whether all
        targets were reached.
    """
    if isinstance(policy, str):
        policy = _load_policy(policy)
    env_map = env_graph.copy()

    if agent_keys is None:
        # Fixed-key formulation: agent i carries key i. Keys come from the
        # environment, not the policy.
        agent_keys = [[i] for i in range(num_agents)]
    elif len(agent_keys) != num_agents:
        raise ValueError(
            f"agent_keys has length {len(agent_keys)} but num_agents={num_agents}"
        )

    agents = [Agent(source, possessed_keys=agent_keys[i]) for i in range(num_agents)]
    replan_times: list[float] = []
    replan_count = 0

    def _maybe_render(idx):
        if render_dir is None:
            return
        path = os.path.join(render_dir, f"frame_{idx:04d}.png")
        render_simulation_frame(env_map, blocked_env_graph, agents, idx, path)

    def _maybe_debug_replan(turn_idx):
        if debug_dir is None:
            return
        path = os.path.join(debug_dir, f"replan_{replan_count:04d}_turn{turn_idx:04d}.png")
        render_replan_debug_frame(env_map, blocked_env_graph, agents,
                                  replan_count, turn_idx, path)

    t0 = time.perf_counter()
    policy.replan(env_map, agents, edge_reward_ratio, obs_discount_factor,
                  sample_recursion, sample_num_obstacle, sample_obstacle_hop,
                  target_reward_ratio=target_reward_ratio, verbose=True)
    replan_times.append(time.perf_counter() - t0)
    _maybe_debug_replan(0)
    replan_count += 1

    _maybe_render(0)

    # Carried across loop iterations: True when the most recent move phase
    # drained some agent's planned_path from length >= 2 to < 2 (i.e. the
    # agent actually walked to the end of its plan rather than receiving an
    # empty plan from a replan). Consumed as a one-shot replan trigger at
    # the top of the next iteration. Required for baseline2_greedy_single_key
    # so an agent arriving at source can replan and pick up a new key —
    # otherwise no existing trigger (newly_blocked / newly_revealed_locks /
    # target_reached) fires on reaching source, and the agent just sits.
    plan_just_exhausted = False

    turn = 0
    while turn < max_turns:
        unreached = [n for n, d in env_map.nodes(data=True) if d.get("type") == "target_unreached"]
        if not unreached:
            break

        # --- 1. Observe from each agent's current position ---
        newly_blocked = set()
        for agent in agents:
            observable = set(blocked_env_graph.nodes[agent.position].get("visible_edges", []))
            assumed = set(env_map.nodes[agent.position].get("visible_edges", []))
            newly_blocked.update(assumed - observable)
            for e in assumed:
                if env_map.has_edge(*e):
                    env_map.edges[e]["observed_edge"] = True

        if newly_blocked:
            for e in list(newly_blocked):
                if env_map.has_edge(*e):
                    env_map.remove_edge(*e)
            for node in env_map.nodes():
                if "visible_edges" in env_map.nodes[node]:
                    env_map.nodes[node]["visible_edges"] = [
                        e for e in env_map.nodes[node]["visible_edges"] if e not in newly_blocked
                    ]

        # --- 1b. Reveal target locks at observed nodes ---
        # Each agent observes every node in its `visible_nodes` set (computed by
        # RealTerrainGrid.compute_all_visibilities) plus its own cell. For any
        # such node that is a target with `lock == UNKNOWN_LOCK` in the
        # planner's view, copy the ground-truth lock from blocked_env_graph onto
        # env_map. Track whether anything changed so we can trigger a replan —
        # the previous assignment may have been computed under an optimistic
        # "lock unknown -> feasible" assumption that just got falsified.
        newly_revealed_locks = False
        for agent in agents:
            observed_nodes = set(
                blocked_env_graph.nodes[agent.position].get("visible_nodes", [])
            )
            observed_nodes.add(agent.position)  # standing on a node observes it
            for n in observed_nodes:
                if env_map.nodes[n].get("lock", UNKNOWN_LOCK) != UNKNOWN_LOCK:
                    continue  # already known
                true_lock = blocked_env_graph.nodes[n].get("lock")
                if true_lock is None or true_lock == UNKNOWN_LOCK:
                    continue  # not a target / nothing to reveal
                env_map.nodes[n]["lock"] = true_lock
                newly_revealed_locks = True

        # --- 2. Replan triggers ---
        replan_needed = plan_just_exhausted  # consume the carried one-shot
        plan_just_exhausted = False
        if newly_revealed_locks:
            replan_needed = True
        if newly_blocked:
            blocked_pairs = set()
            for u, v in newly_blocked:
                blocked_pairs.add((u, v))
                blocked_pairs.add((v, u))
            for agent in agents:
                if len(agent.planned_path) >= 2 and _path_blocked_by(agent.planned_path, blocked_pairs):
                    replan_needed = True
                    break

        for agent in agents:
            if env_map.nodes[agent.position].get("type") == "target_unreached":
                lock = env_map.nodes[agent.position].get("lock")
                # Targets without a lock are openable by anyone; otherwise the
                # agent has to be carrying the matching key. An agent standing
                # on a locked target it can't open is just sitting there — the
                # target stays `target_unreached` until a key-bearing agent
                # arrives.
                if lock is None or agent.has_key(lock):
                    env_map.nodes[agent.position]["type"] = "target_reached"
                    replan_needed = True

        if replan_needed:
            t0 = time.perf_counter()
            policy.replan(env_map, agents, edge_reward_ratio, obs_discount_factor,
                          sample_recursion, sample_num_obstacle, sample_obstacle_hop,
                          target_reward_ratio=target_reward_ratio, verbose=True)
            replan_times.append(time.perf_counter() - t0)
            _maybe_debug_replan(turn)
            replan_count += 1

        # If the replan-triggers block just flipped the last unreached target
        # to target_reached, break out NOW — before the move phase. Otherwise
        # baseline2 (which routes spent agents back to source) would tick one
        # extra step of "walk home" for every just-unlocked agent in the
        # final turn, an asymmetric penalty vs baseline1 / finite_horizon_MA
        # whose agents simply stop at the final target. The terminal unlock
        # itself still counts — it was paid for by the previous turn's move.
        if not [n for n, d in env_map.nodes(data=True)
                if d.get("type") == "target_unreached"]:
            break

        # --- 3. Move each agent up to movement_modifier steps ---
        any_progress = False
        for agent in agents:
            initial_plan_len = len(agent.planned_path)
            for _ in range(agent.movement_modifier):
                if len(agent.planned_path) < 2:
                    break
                next_node = agent.planned_path[1]
                if not blocked_env_graph.has_edge(agent.position, next_node):
                    break
                cost = blocked_env_graph.edges[agent.position, next_node]["distance"]
                agent.move(agent.position, next_node, cost)
                any_progress = True
            # Distinguish "agent finished its planned path by walking" from
            # "agent was assigned an empty plan by the most recent replan":
            # only the former should trigger another replan next turn.
            if initial_plan_len >= 2 and len(agent.planned_path) < 2:
                plan_just_exhausted = True

        if not any_progress and not replan_needed:
            break

        turn += 1
        _maybe_render(turn)

    completed = not [n for n, d in env_map.nodes(data=True) if d.get("type") == "target_unreached"]
    return {
        "agents": agents,
        "turns": turn,
        "completed": completed,
        "total_cost": sum(a.total_traversal_cost for a in agents),
        "max_agent_cost": max(a.total_traversal_cost for a in agents),
        "per_agent_cost": [a.total_traversal_cost for a in agents],
        "replan_times": replan_times,
    }


# ---------------------------------------------------------------------------
# 5. Benchmark driver
# ---------------------------------------------------------------------------

def run_real_map_multi_agent_benchmark(
    dem_path,
    road_pkl=None,
    n_size=64,
    source=(0, 0),
    targets=((14, 54), (1, 29), (33, 17), (34, 35), (63,37), (37,5), (49,58)),
    num_agents=2,
    edge_reward_ratio=1.0,
    obs_discount_factor=1.0,
    target_num_neighbors=4,
    target_recursion=2,
    target_num_obstacles=3,
    target_obstacle_hop=4,
    sample_recursion=4,
    sample_num_obstacle=3,
    sample_obstacle_hop=4,
    target_reward_ratio=0.0,
    obstacle_specs=None,
    agent_keys=None,
    policy="finite_horizon_MA",
    output_csv="./my_policy_simulation/real_map_ma_results.csv",
    output_json="./my_policy_simulation/real_map_ma_summary.json",
    render=False,
    render_dir="./my_policy_simulation/output_images",
    output_mp4="./my_policy_simulation/simulation.mp4",
    mp4_fps=4,
    debug=False,
    debug_dir="./my_policy_simulation/replan_debug",
):
    here = os.path.dirname(os.path.abspath(__file__))
    output_csv_path = os.path.join(here, output_csv)
    output_json_path = os.path.join(here, output_json)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # --- Hyperparameter banner ---
    print("=" * 50)
    print("HYPERPARAMETERS")
    print("=" * 50)
    print(f"edge_reward_ratio:     {edge_reward_ratio}")
    print(f"target_reward_ratio:   {target_reward_ratio}")
    print(f"obs_discount_factor:   {obs_discount_factor}")
    print(f"target_num_neighbors:  {target_num_neighbors}")
    print(f"target_recursion:      {target_recursion}")
    print(f"target_num_obstacles:  {target_num_obstacles}")
    print(f"target_obstacle_hop:   {target_obstacle_hop}")
    print(f"sample_recursion:      {sample_recursion}")
    print(f"sample_num_obstacle:   {sample_num_obstacle}")
    print(f"sample_obstacle_hop:   {sample_obstacle_hop}")
    print(f"policy:                {policy if isinstance(policy, str) else policy.__name__}")
    print("=" * 50)

    # Resolve the policy *once* so every downstream call (default keys, inner
    # runner, summary) sees the same module.
    if isinstance(policy, str):
        policy_name = policy
        policy = _load_policy(policy)
    else:
        policy_name = policy.__name__.split(".")[-1]

    # --- Build clean graph ---
    print("Loading DEM and roads...")
    height_grid = load_real_terrain(dem_path, n_size)
    road_nodes, road_edges = load_roads(road_pkl)
    print(f"Roads: {len(road_nodes)} nodes, {len(road_edges)} edges")

    terrain = RealTerrainGrid(
        height_grid,
        source=source,
        targets=list(targets),
        k_up=1.0, k_down=2.0,
        road_nodes=road_nodes, road_edges=road_edges,
    )

    print("Computing visibilities...")
    terrain.compute_all_visibilities()
    env_graph = terrain.get_graph().copy()
    print(f"Graph: {env_graph.number_of_nodes()} nodes, {env_graph.number_of_edges()} edges")

    # Populate `num_used` on edges (needed by path reward) as a side effect of
    # building the target graph. We don't use the returned target_graph here.
    print("Populating edge `num_used` via target-graph construction...")
    create_fully_connected_target_graph(
        env_graph, source=source, targets=list(targets),
        num_neighbors=target_num_neighbors,
        recursions=target_recursion,
        num_obstacles=target_num_obstacles,
        obstacle_hop=target_obstacle_hop,
    )

    # --- Apply obstacles (all of them, deterministically) ---
    if obstacle_specs is None:
        obstacle_specs = DEFAULT_OBSTACLE_SPECS
    print(f"Applying {len(obstacle_specs)} obstacle ovals")

    blocked_terrain = copy.deepcopy(terrain)
    for center, rx, ry in obstacle_specs:
        blocked_terrain.add_obstacle(center=center, rx=rx, ry=ry)
    blocked_env_graph = blocked_terrain.get_graph().copy()

    obs_edges = [(u, v) for u, v in blocked_env_graph.edges()
                 if blocked_env_graph.nodes[u].get("type") == "obstacle"
                 or blocked_env_graph.nodes[v].get("type") == "obstacle"]
    blocked_env_graph.remove_edges_from(obs_edges)
    obs_set = set(obs_edges)
    for node in blocked_env_graph.nodes():
        if "visible_edges" in blocked_env_graph.nodes[node]:
            blocked_env_graph.nodes[node]["visible_edges"] = [
                e for e in blocked_env_graph.nodes[node]["visible_edges"] if e not in obs_set
            ]

    for t in targets:
        if not nx.has_path(blocked_env_graph, source, t):
            print(f"Target {t} unreachable from source {source} after blocking. Aborting.")
            return

    # --- Assign locks to targets ---
    # Locks are drawn from the agent key set {0, ..., num_agents-1} (agent i
    # holds key i). To keep the lock types evenly represented, we build a
    # repeated permutation: cycle the keys 0,1,2,0,1,2,... out to num_targets
    # entries, then shuffle. E.g. 3 agents, 7 targets -> [0,1,2,0,1,2,0]
    # shuffled. This guarantees every lock is openable by some agent and that
    # no lock type dominates the map.
    #
    # Locks are partially observable: blocked_env_graph (ground truth) carries
    # the true lock IDs; env_graph (planner's initial view) gets UNKNOWN_LOCK
    # for every target. The simulation reveals the true lock onto env_map
    # whenever an agent observes the target node.
    lock_ids = [i % num_agents for i in range(len(targets))]
    random.shuffle(lock_ids)
    target_locks = {t: lock_ids[i] for i, t in enumerate(targets)}
    for t, lock_id in target_locks.items():
        env_graph.nodes[t]["lock"] = UNKNOWN_LOCK
        blocked_env_graph.nodes[t]["lock"] = lock_id
    print(f"Target locks (ground truth): {target_locks}")

    # --- Run the single deterministic realization ---
    render_path = None
    if render:
        render_path = os.path.join(here, render_dir)
        clear_frame_dir(render_path)
        print(f"Rendering frames to {render_path}")

    debug_path = None
    if debug:
        debug_path = os.path.join(here, debug_dir)
        clear_debug_dir(debug_path)
        print(f"Saving replan-debug frames to {debug_path}")

    # Fixed-key formulation: agent i carries key i. Keys are an environment
    # property, not a policy choice. Pass an explicit `agent_keys` to override.
    if agent_keys is None:
        effective_agent_keys = [[i] for i in range(num_agents)]
        print(f"agent_keys (fixed: agent i -> key i): {effective_agent_keys}")
    else:
        effective_agent_keys = [list(k) for k in agent_keys]
        print(f"agent_keys: {effective_agent_keys}")

    print(f"\nRunning {num_agents} agents")
    t0 = time.perf_counter()
    result = run_multi_agent_simulation(
        env_graph, blocked_env_graph, source, num_agents, edge_reward_ratio,
        obs_discount_factor=obs_discount_factor,
        sample_recursion=sample_recursion,
        sample_num_obstacle=sample_num_obstacle,
        sample_obstacle_hop=sample_obstacle_hop,
        target_reward_ratio=target_reward_ratio,
        agent_keys=effective_agent_keys,
        policy=policy,
        render_dir=render_path,
        debug_dir=debug_path,
    )
    total_runtime = time.perf_counter() - t0

    if render:
        mp4_path = os.path.join(here, output_mp4)
        try:
            make_mp4_from_frames(render_path, mp4_path, fps=mp4_fps)
            print(f"MP4 written to {mp4_path}")
        except FileNotFoundError:
            print("ffmpeg not found — frames saved but MP4 was not generated.")
            print("Install with: brew install ffmpeg")
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed: {e.stderr.decode(errors='ignore') if e.stderr else e}")

    replan_times = result["replan_times"]
    mean_replan_time = float(np.mean(replan_times)) if replan_times else 0.0

    print("\n" + "=" * 50)
    print("MULTI-AGENT REAL MAP RESULTS")
    print("=" * 50)
    print(f"Map: WV_DEM {n_size}x{n_size} | {num_agents} agents | {len(targets)} targets")
    print(f"Obstacles: {len(obstacle_specs)}")
    print(f"Completed: {result['completed']} | Turns: {result['turns']}")
    avg_cost = result['total_cost'] / num_agents if num_agents else 0.0
    print(f"Avg cost per agent: {avg_cost:.2f}")
    print(f"Max agent cost:     {result['max_agent_cost']:.2f}")
    print(f"Per-agent cost: {[f'{c:.2f}' for c in result['per_agent_cost']]}")
    print(f"Total simulation runtime: {total_runtime:.3f}s")
    print(f"Replans: {len(replan_times)} | Mean replan time: {mean_replan_time*1000:.2f}ms")
    print("=" * 50)

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent_idx", "cost", "trajectory_length"])
        for i, agent in enumerate(result["agents"]):
            writer.writerow([i, f"{agent.total_traversal_cost:.4f}", len(agent.trajectory)])
    print(f"Per-agent results: {output_csv_path}")

    summary = {
        "map": f"WV_DEM_{n_size}x{n_size}",
        "policy": policy_name,
        "num_agents": num_agents,
        "num_targets": len(targets),
        "num_obstacles": len(obstacle_specs),
        "edge_reward_ratio": edge_reward_ratio,
        "target_reward_ratio": target_reward_ratio,
        "completed": result["completed"],
        "turns": result["turns"],
        "avg_cost_per_agent": float(avg_cost),
        "max_agent_cost": float(result["max_agent_cost"]),
        "per_agent_cost": [float(c) for c in result["per_agent_cost"]],
        "total_runtime": float(total_runtime),
        "num_replans": len(replan_times),
        "mean_replan_time": mean_replan_time,
        "replan_times": [float(t) for t in replan_times],
        "target_locks": {str(t): lock for t, lock in target_locks.items()},
        "agent_keys": [list(k) for k in effective_agent_keys],
    }
    with open(output_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {output_json_path}")


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-agent benchmark on real DEM map")
    parser.add_argument("--dem-path", type=str, default=None)
    parser.add_argument("--road-pkl", type=str, default=None)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--output", type=str, default="./my_policy_simulation/real_map_ma_results.csv")
    parser.add_argument("--output-summary", type=str, default="./my_policy_simulation/real_map_ma_summary.json")
    parser.add_argument("--render", action="store_true",
                        help="Save a PNG per turn and compile into an MP4 at 4 fps.")
    parser.add_argument("--render-dir", type=str, default="./my_policy_simulation/output_images")
    parser.add_argument("--output-mp4", type=str, default="./my_policy_simulation/simulation.mp4")
    parser.add_argument("--mp4-fps", type=int, default=4)
    parser.add_argument("--debug", action="store_true",
                        help="On every replan, save a separate frame overlaying all "
                             "(agent, target) shortest paths as dashed lines.")
    parser.add_argument("--debug-dir", type=str, default="./my_policy_simulation/replan_debug")
    parser.add_argument("--policy", type=str, default="finite_horizon_MA", choices=POLICIES,
                        help="Which planning policy to run. Default is the main "
                             "(submodular-aware) policy in finite_horizon_MA.py.")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    real_maps_dir = os.path.join(project_root, "Real_Life_Maps")
    if args.dem_path is None:
        args.dem_path = os.path.join(real_maps_dir, "WV_DEM.tif")
    if args.road_pkl is None:
        candidate = os.path.join(real_maps_dir, "WV_roads.pkl")
        args.road_pkl = candidate if os.path.exists(candidate) else None

    if not os.path.exists(args.dem_path):
        print(f"DEM file not found: {args.dem_path}")
        sys.exit(1)

    # --- Algorithm hyperparameters (set here, not via CLI) ---
    edge_reward_ratio = 1.0
    target_reward_ratio = 0.0
    obs_discount_factor = 0.9
    target_num_neighbors = 3
    target_recursion = 2
    target_num_obstacles = 3
    target_obstacle_hop = 4
    sample_recursion = 2
    sample_num_obstacle = 3
    sample_obstacle_hop = 4
    

    # Random seed for reproducibility.
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)

    run_real_map_multi_agent_benchmark(
        dem_path=args.dem_path,
        road_pkl=args.road_pkl,
        n_size=args.grid_size,
        num_agents=args.num_agents,
        edge_reward_ratio=edge_reward_ratio,
        obs_discount_factor=obs_discount_factor,
        target_num_neighbors=target_num_neighbors,
        target_recursion=target_recursion,
        target_num_obstacles=target_num_obstacles,
        target_obstacle_hop=target_obstacle_hop,
        sample_recursion=sample_recursion,
        sample_num_obstacle=sample_num_obstacle,
        sample_obstacle_hop=sample_obstacle_hop,
        target_reward_ratio=target_reward_ratio,
        policy=args.policy,
        output_csv=args.output,
        output_json=args.output_summary,
        render=args.render,
        render_dir=args.render_dir,
        output_mp4=args.output_mp4,
        mp4_fps=args.mp4_fps,
        debug=args.debug,
        debug_dir=args.debug_dir,
    )


if __name__ == "__main__":
    main()
