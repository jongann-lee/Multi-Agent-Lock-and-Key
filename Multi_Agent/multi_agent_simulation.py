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

import numpy as np
import networkx as nx

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Real_Life_Maps.real_map_generation import RealTerrainGrid
from Graph_Generation.target_graph import create_fully_connected_target_graph
from Multi_Agent.finite_horizon_MA import Agent, sequential_greedy_assignment
from Multi_Agent.simulation_utils import (
    render_simulation_frame, clear_frame_dir, make_mp4_from_frames,
    render_replan_debug_frame, clear_debug_dir,
)


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

def _replan(env_map, agents, reward_ratio, obs_discount_factor=1.0,
            sample_recursion=0, sample_num_obstacle=0, sample_obstacle_hop=0):
    """Run the greedy assignment and write a fresh shortest path onto each agent."""
    for agent in agents:
        agent.planned_path = []
    assignment = sequential_greedy_assignment(
        env_map, agents, reward_ratio, obs_discount_factor,
        sample_recursion=sample_recursion,
        sample_num_obstacle=sample_num_obstacle,
        sample_obstacle_hop=sample_obstacle_hop,
        verbose=True,
    )
    for i, target in assignment.items():
        try:
            agents[i].planned_path = nx.shortest_path(
                env_map, agents[i].position, target, weight="distance"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            agents[i].planned_path = []


def _path_blocked_by(path, blocked_pairs):
    """True if any consecutive pair in `path` is in `blocked_pairs`."""
    for k in range(len(path) - 1):
        if (path[k], path[k + 1]) in blocked_pairs or (path[k + 1], path[k]) in blocked_pairs:
            return True
    return False


def run_multi_agent_simulation(env_graph, blocked_env_graph, source, num_agents,
                               reward_ratio, obs_discount_factor=1.0,
                               sample_recursion=0, sample_num_obstacle=0,
                               sample_obstacle_hop=0,
                               max_turns=2000, render_dir=None, debug_dir=None):
    """
    Args:
        env_graph: clean planner graph (will be deep-copied for the run).
        blocked_env_graph: ground-truth graph with ovals applied.
        source: starting node for all agents.
        num_agents: number of agents.
        reward_ratio: lambda for path reward.
        max_turns: hard cap to avoid infinite loops on unsolvable realizations.
        render_dir: if set, write one PNG per turn to this directory.

    Returns:
        dict with per-agent costs, trajectories, turn count, and whether all
        targets were reached.
    """
    env_map = env_graph.copy()
    agents = [Agent(source) for _ in range(num_agents)]
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
    _replan(env_map, agents, reward_ratio, obs_discount_factor,
            sample_recursion, sample_num_obstacle, sample_obstacle_hop)
    replan_times.append(time.perf_counter() - t0)
    _maybe_debug_replan(0)
    replan_count += 1

    _maybe_render(0)

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

        # --- 2. Replan triggers ---
        replan_needed = False
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
                env_map.nodes[agent.position]["type"] = "target_reached"
                replan_needed = True

        if replan_needed:
            t0 = time.perf_counter()
            _replan(env_map, agents, reward_ratio, obs_discount_factor,
            sample_recursion, sample_num_obstacle, sample_obstacle_hop)
            replan_times.append(time.perf_counter() - t0)
            _maybe_debug_replan(turn)
            replan_count += 1

        # --- 3. Move each agent up to movement_modifier steps ---
        any_progress = False
        for agent in agents:
            for _ in range(agent.movement_modifier):
                if len(agent.planned_path) < 2:
                    break
                next_node = agent.planned_path[1]
                if not blocked_env_graph.has_edge(agent.position, next_node):
                    break
                cost = blocked_env_graph.edges[agent.position, next_node]["distance"]
                agent.move(agent.position, next_node, cost)
                any_progress = True

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
    reward_ratio=1.0,
    obs_discount_factor=1.0,
    target_num_neighbors=4,
    target_recursion=2,
    target_num_obstacles=3,
    target_obstacle_hop=4,
    sample_recursion=4,
    sample_num_obstacle=3,
    sample_obstacle_hop=4,
    obstacle_specs=None,
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
    print(f"reward_ratio:          {reward_ratio}")
    print(f"obs_discount_factor:   {obs_discount_factor}")
    print(f"target_num_neighbors:  {target_num_neighbors}")
    print(f"target_recursion:      {target_recursion}")
    print(f"target_num_obstacles:  {target_num_obstacles}")
    print(f"target_obstacle_hop:   {target_obstacle_hop}")
    print(f"sample_recursion:      {sample_recursion}")
    print(f"sample_num_obstacle:   {sample_num_obstacle}")
    print(f"sample_obstacle_hop:   {sample_obstacle_hop}")
    print("=" * 50)

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

    print(f"\nRunning {num_agents} agents")
    t0 = time.perf_counter()
    result = run_multi_agent_simulation(
        env_graph, blocked_env_graph, source, num_agents, reward_ratio,
        obs_discount_factor=obs_discount_factor,
        sample_recursion=sample_recursion,
        sample_num_obstacle=sample_num_obstacle,
        sample_obstacle_hop=sample_obstacle_hop,
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
        "num_agents": num_agents,
        "num_targets": len(targets),
        "num_obstacles": len(obstacle_specs),
        "reward_ratio": reward_ratio,
        "completed": result["completed"],
        "turns": result["turns"],
        "avg_cost_per_agent": float(avg_cost),
        "max_agent_cost": float(result["max_agent_cost"]),
        "per_agent_cost": [float(c) for c in result["per_agent_cost"]],
        "total_runtime": float(total_runtime),
        "num_replans": len(replan_times),
        "mean_replan_time": mean_replan_time,
        "replan_times": [float(t) for t in replan_times],
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
    reward_ratio = 1.0
    obs_discount_factor = 0.9
    target_num_neighbors = 3
    target_recursion = 2
    target_num_obstacles = 3
    target_obstacle_hop = 4
    sample_recursion = 2
    sample_num_obstacle = 3
    sample_obstacle_hop = 4

    run_real_map_multi_agent_benchmark(
        dem_path=args.dem_path,
        road_pkl=args.road_pkl,
        n_size=args.grid_size,
        num_agents=args.num_agents,
        reward_ratio=reward_ratio,
        obs_discount_factor=obs_discount_factor,
        target_num_neighbors=target_num_neighbors,
        target_recursion=target_recursion,
        target_num_obstacles=target_num_obstacles,
        target_obstacle_hop=target_obstacle_hop,
        sample_recursion=sample_recursion,
        sample_num_obstacle=sample_num_obstacle,
        sample_obstacle_hop=sample_obstacle_hop,
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
