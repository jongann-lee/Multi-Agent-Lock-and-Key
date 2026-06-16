"""
Auxiliary (non-rendering) helpers for the multi-agent simulation.

Split out of multi_agent_simulation.py to keep the driver focused on the turn
loop and benchmark plumbing. Three groups live here:

- DEM + road loading: read the terrain raster and the road pickle off disk.
- Planning: `_replan` runs the greedy assignment and routes each agent down its
  scored path; `_recompute_num_used` refreshes the edge `num_used` weights
  against the planner's current map and the targets' current positions before
  each replan.
- Target bookkeeping: the `Target` class (a biased random walk with look-ahead
  planning), plus `_snapshot_base_type` / `_sync_targets_to_graph`, which
  project the moving Target objects onto the planner's internal map via the
  node `type` attribute the assignment code reads.

Rendering helpers live in rendering_utils.py.
"""
import os
import sys
import pickle

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Graph_Generation.target_graph import create_fully_connected_target_graph
from Multi_Agent.finite_horizon_MA import sequential_greedy_assignment


# ---------------------------------------------------------------------------
# DEM + road loading
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


# ---------------------------------------------------------------------------
# Target: biased random walk with look-ahead planning
# ---------------------------------------------------------------------------

# Cardinal moves on the grid graph, as (dr, dc) deltas.
CARDINAL_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


class Target:
    """A target that performs a biased random walk on the ground-truth graph.

    Per-step movement model, relative to the direction of the last move:
      - 0.5  continue forward (same direction),
      - 0.25 turn to each side (the two perpendicular directions).
    Reversing is never chosen. The very first step has no established direction,
    so it picks uniformly among the valid neighbours.

    Moves that leave the map or cross a missing/blocked edge are dropped and the
    remaining valid actions are re-weighted — equivalent to "resample until the
    action is valid", but without the risk of looping. If no forward/side move
    is valid (a dead end) the target reverses; if even that is blocked it stays.

    The next `lookahead` steps are kept planned at all times: that many are
    pre-sampled on construction, and one more is sampled each time the target
    advances. `position` and `reached` are the simulation's source of truth; the
    planner discovers the target via the node `type` it gets stamped onto the
    planner map (see _sync_targets_to_graph).
    """

    def __init__(self, target_id, start, graph, lookahead=10):
        self.id = target_id
        self.graph = graph
        self.lookahead = lookahead
        self.position = start
        self.direction = None          # (dr, dc) of the last move taken
        self.reached = False
        self.history = [start]         # positions actually visited
        self.planned = []              # upcoming positions; planned[0] is next
        # Planning frontier: the position/direction at the tip of `planned`, so
        # further steps can be sampled without disturbing the live position.
        self._frontier_pos = start
        self._frontier_dir = None
        self._extend_plan()

    def _is_valid(self, pos, direction):
        """True if stepping `direction` from `pos` crosses a traversable edge."""
        nxt = (pos[0] + direction[0], pos[1] + direction[1])
        return self.graph.has_node(nxt) and self.graph.has_edge(pos, nxt)

    def _sample_direction(self, pos, direction):
        """Pick the next direction from `pos` given the incoming `direction`.

        Returns a (dr, dc) delta, or None if `pos` has no traversable neighbour.
        """
        if direction is None:
            choices = [d for d in CARDINAL_DIRECTIONS if self._is_valid(pos, d)]
            if not choices:
                return None
            return choices[int(np.random.randint(len(choices)))]

        forward = direction
        left = (-direction[1], direction[0])
        right = (direction[1], -direction[0])
        weighted = [(forward, 0.5), (left, 0.25), (right, 0.25)]
        valid = [(d, w) for d, w in weighted if self._is_valid(pos, d)]
        if valid:
            dirs, weights = zip(*valid)
            total = sum(weights)
            probs = [w / total for w in weights]
            return dirs[int(np.random.choice(len(dirs), p=probs))]

        # Dead end: no forward/side move. Reverse if possible, else stay put.
        back = (-direction[0], -direction[1])
        return back if self._is_valid(pos, back) else None

    def _extend_plan(self):
        """Refill `planned` up to `lookahead` upcoming steps."""
        while len(self.planned) < self.lookahead:
            d = self._sample_direction(self._frontier_pos, self._frontier_dir)
            if d is None:
                break  # stuck: cannot plan any further from here
            nxt = (self._frontier_pos[0] + d[0], self._frontier_pos[1] + d[1])
            self.planned.append(nxt)
            self._frontier_pos = nxt
            self._frontier_dir = d

    def advance(self):
        """Move to the next planned node and sample one more (no-op if reached)."""
        if self.reached or not self.planned:
            return
        nxt = self.planned.pop(0)
        self.direction = (nxt[0] - self.position[0], nxt[1] - self.position[1])
        self.position = nxt
        self.history.append(nxt)
        self._extend_plan()

# ---------------------------------------------------------------------------
# Planning: assignment + num_used recompute
# ---------------------------------------------------------------------------

def _recompute_num_used(env_map, source, target_num_neighbors, target_recursion,
                        target_num_obstacles, target_obstacle_hop):
    """Refresh the `num_used` edge attribute on env_map for the current state.

    `num_used` measures how heavily each edge is used across the diverse paths
    connecting the source to the remaining targets, and the path reward reads
    it. Targets now move, so the set of unreached targets — and hence the paths
    between them — changes every replan; recomputing here rebuilds `num_used`
    against the planner's current map and the targets' current positions.

    create_fully_connected_target_graph resets `num_used` to 0 internally, but
    it *appends* to each node's `stored_path_contributions` list without
    resetting it. That attribute is unused by the multi-agent reward, but the
    lists would otherwise grow without bound across replans, so we clear them
    first.
    """
    unreached = [n for n, d in env_map.nodes(data=True)
                 if d.get("type") == "target_unreached"]
    if not unreached:
        return
    for n in env_map.nodes():
        if "stored_path_contributions" in env_map.nodes[n]:
            env_map.nodes[n]["stored_path_contributions"] = []
    create_fully_connected_target_graph(
        env_map, source=source, targets=unreached,
        num_neighbors=target_num_neighbors,
        recursions=target_recursion,
        num_obstacles=target_num_obstacles,
        obstacle_hop=target_obstacle_hop,
    )


def _replan(env_map, agents, reward_ratio, obs_discount_factor=1.0,
            sample_recursion=0, sample_num_obstacle=0, sample_obstacle_hop=0,
            source=None, target_num_neighbors=4, target_recursion=2,
            target_num_obstacles=2, target_obstacle_hop=2):
    """Run the greedy assignment and send each agent down its reward-maximizing path.

    Before scoring, the `num_used` edge values are recomputed against the
    current map and the targets' current positions, since the path reward reads
    `num_used` and the values go stale as targets move and obstacles are
    discovered. See _recompute_num_used. (Skipped when source is None.)

    sequential_greedy_assignment scores each (agent, target) pair over a set of
    sampled candidate paths and returns the highest-reward one. We follow that
    path directly — it is generally NOT the shortest path for a reward-driven
    policy, so recomputing a shortest path here would discard the planning.
    """
    if source is not None:
        _recompute_num_used(env_map, source, target_num_neighbors,
                            target_recursion, target_num_obstacles,
                            target_obstacle_hop)
    for agent in agents:
        agent.planned_path = []
    assignment = sequential_greedy_assignment(
        env_map, agents, reward_ratio, obs_discount_factor,
        sample_recursion=sample_recursion,
        sample_num_obstacle=sample_num_obstacle,
        sample_obstacle_hop=sample_obstacle_hop,
        verbose=True,
    )
    for i, (target, path) in assignment.items():
        agents[i].planned_path = list(path) if path else []


def _path_blocked_by(path, blocked_pairs):
    """True if any consecutive pair in `path` is in `blocked_pairs`."""
    for k in range(len(path) - 1):
        if (path[k], path[k + 1]) in blocked_pairs or (path[k + 1], path[k]) in blocked_pairs:
            return True
    return False


# ---------------------------------------------------------------------------
# Planner-map sync for moving targets
# ---------------------------------------------------------------------------

def agent_visible_nodes(graph, agents):
    """Union of nodes within any agent's line of sight.

    Visibility is stored per node as `visible_edges` on the ground-truth graph;
    a node is visible from an agent if it is an endpoint of one of those edges.
    This is the observation channel the engagement rule uses: a target on a
    visible node has its next few planned steps revealed to the planner.
    """
    visible = set()
    for agent in agents:
        for u, v in graph.nodes[agent.position].get("visible_edges", []):
            visible.add(u)
            visible.add(v)
    return visible


def _snapshot_base_type(graph):
    """Record each node's non-target base type so target stamps can be undone.

    Original `target_*` markers (e.g. from RealTerrainGrid) collapse to
    'intermediate' — target placement is now driven entirely by Target objects.
    """
    base = {}
    for node, d in graph.nodes(data=True):
        t = d.get("type")
        base[node] = "intermediate" if t in ("target_unreached", "target_reached") else t
    return base


def _sync_targets_to_graph(graph, targets, base_type):
    """Project the (observed) Target positions onto the planner's internal map.

    The planner enumerates targets by scanning for node type 'target_unreached',
    so this is how Target objects become visible to the assignment code. Called
    during the observation stage. The planner sees true positions for now, so
    every target is stamped; partial target observability would stamp only the
    targets currently sighted by some agent.
    """
    # Undo previous target stamps, restoring each node's base (non-target) type.
    for node, d in graph.nodes(data=True):
        if d.get("type") in ("target_unreached", "target_reached"):
            d["type"] = base_type.get(node, "intermediate")
    # Stamp current target positions; never clobber the source marker.
    for tgt in targets:
        node = tgt.position
        if graph.nodes[node].get("type") == "source":
            continue
        graph.nodes[node]["type"] = "target_reached" if tgt.reached else "target_unreached"
