"""
Visualization helpers for the multi-agent simulation.

Mirrors the rendering rules used in Real_Life_Maps/visualization.ipynb:
- terrain by height, obstacles in black, road edges in dark grey
- source = green circle, target_unreached = red X, target_reached = grey X
- each agent gets a single color; solid line = trajectory so far,
  dotted line = planned path
- a target within an agent's line of sight also shows its next few planned
  steps as a magenta dotted line
"""
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.collections as mc
import networkx as nx

from Multi_Agent.simulation_utils import agent_visible_nodes


DEFAULT_AGENT_COLORS = ["blue", "red", "green"]

# When a target is within an agent's line of sight, its next few planned steps
# are previewed as a dotted line in this colour (distinct from the agent reds).
TARGET_PREDICTION_COLOR = "magenta"


def _agent_color(idx, agent_colors=None):
    colors = agent_colors if agent_colors is not None else DEFAULT_AGENT_COLORS
    if idx < len(colors):
        return colors[idx]
    return plt.cm.tab10(idx % 10)


def render_simulation_frame(env_map, blocked_env_graph, agents, turn_idx, output_path,
                            agent_colors=None, title=None, extra_paths=None,
                            targets=None, target_preview_steps=3):
    """Render one frame of the multi-agent simulation to PNG.

    Target rendering rules: a target is always drawn as an X marker — red while
    unreached, grey once reached. In addition, if an unreached target is within
    an agent's line of sight, its next `target_preview_steps` planned steps are
    previewed as a dotted line (the engagement rule that lets the planner peek a
    few steps ahead when it can see the target, shown here as a render-only cue).

    Args:
        env_map: planner's view of the graph (used for the source marker).
        blocked_env_graph: ground-truth graph (used for terrain, obstacles,
            roads, and the agents' line-of-sight `visible_edges`).
        agents: list of Agent instances (each with .position, .trajectory, .planned_path).
        turn_idx: integer turn number — used in title and filename ordering.
        output_path: absolute path to write the PNG.
        agent_colors: optional list of matplotlib colors per agent.
        title: optional title override (default "Turn N").
        extra_paths: optional list of (path, color, linestyle, linewidth) tuples to
            overlay before agent markers. Used by the replan-debug renderer to
            show every candidate (agent, target) shortest path.
        targets: optional list of Target objects. Target markers are drawn from
            each target's live .position/.reached. If omitted, falls back to
            scanning env_map node types (legacy static-target behaviour).
        target_preview_steps: how many upcoming planned steps to preview for a
            visible target (0 disables the preview). Only used when `targets` is
            given.
    """
    pos = nx.get_node_attributes(blocked_env_graph, 'pos')
    all_heights = [d.get('height', 0) for _, d in blocked_env_graph.nodes(data=True)]
    max_height = max(all_heights) if all_heights else 1
    norm = mcolors.Normalize(vmin=0, vmax=max_height + 1)
    cmap_terrain = plt.cm.terrain

    xs = sorted(set(p[0] for p in pos.values()))
    ys = sorted(set(p[1] for p in pos.values()))
    cell_w = xs[1] - xs[0]
    cell_h = ys[1] - ys[0]

    fig, ax = plt.subplots(figsize=(10, 10))

    for node, data in blocked_env_graph.nodes(data=True):
        x, y = pos[node]
        if data.get("type") == "obstacle":
            color = "black"
        else:
            color = cmap_terrain(norm(data.get('height', 0)))
        ax.add_patch(patches.Rectangle(
            (x - cell_w / 2, y - cell_h / 2), cell_w, cell_h,
            linewidth=0, facecolor=color
        ))

    road_segments = [[pos[u], pos[v]] for u, v, d in blocked_env_graph.edges(data=True) if d.get('is_road')]
    if road_segments:
        ax.add_collection(mc.LineCollection(road_segments, colors='#404040', linewidths=2.0, zorder=4))

    for i, agent in enumerate(agents):
        color = _agent_color(i, agent_colors)
        if len(agent.trajectory) >= 2:
            traj_segments = [[pos[agent.trajectory[k]], pos[agent.trajectory[k + 1]]]
                             for k in range(len(agent.trajectory) - 1)]
            ax.add_collection(mc.LineCollection(traj_segments, colors=color, linewidths=4.0, zorder=5))
        if len(agent.planned_path) >= 2:
            plan_segments = [[pos[agent.planned_path[k]], pos[agent.planned_path[k + 1]]]
                             for k in range(len(agent.planned_path) - 1)]
            ax.add_collection(mc.LineCollection(plan_segments, colors=color, linewidths=3.2,
                                                zorder=6, linestyles='dotted'))

    if extra_paths:
        for path, color, linestyle, linewidth in extra_paths:
            if path is None or len(path) < 2:
                continue
            segs = [[pos[path[k]], pos[path[k + 1]]] for k in range(len(path) - 1)]
            ax.add_collection(mc.LineCollection(
                segs, colors=color, linewidths=linewidth,
                zorder=4, linestyles=linestyle, alpha=0.7,
            ))

    src_pts = [pos[n] for n, d in env_map.nodes(data=True) if d.get("type") == "source"]
    unreached_pts, reached_pts = [], []
    if targets is not None:
        # Preview the next few planned steps of any target currently in an
        # agent's line of sight (drawn before the markers so the X sits on top).
        visible_nodes = agent_visible_nodes(blocked_env_graph, agents)
        for tgt in targets:
            if tgt.reached:
                reached_pts.append(pos[tgt.position])
                continue
            unreached_pts.append(pos[tgt.position])
            if target_preview_steps > 0 and tgt.position in visible_nodes:
                preview = [tgt.position] + list(tgt.planned[:target_preview_steps])
                if len(preview) >= 2:
                    segs = [[pos[preview[k]], pos[preview[k + 1]]]
                            for k in range(len(preview) - 1)]
                    ax.add_collection(mc.LineCollection(
                        segs, colors=TARGET_PREDICTION_COLOR, linewidths=2.5,
                        zorder=7, linestyles='dotted'))
    else:
        for node, data in env_map.nodes(data=True):
            t = data.get("type")
            if t == "target_unreached":
                unreached_pts.append(pos[node])
            elif t == "target_reached":
                reached_pts.append(pos[node])

    if src_pts:
        ax.scatter([p[0] for p in src_pts], [p[1] for p in src_pts],
                   marker='o', s=220, facecolor='limegreen', edgecolor='darkgreen',
                   linewidths=1.5, zorder=10)
    if unreached_pts:
        ax.scatter([p[0] for p in unreached_pts], [p[1] for p in unreached_pts],
                   marker='x', s=200, color='red', linewidths=3, zorder=10)
    if reached_pts:
        ax.scatter([p[0] for p in reached_pts], [p[1] for p in reached_pts],
                   marker='x', s=200, color='grey', linewidths=3, zorder=10)

    for i, agent in enumerate(agents):
        color = _agent_color(i, agent_colors)
        x, y = pos[agent.position]
        ax.scatter([x], [y], marker='o', s=130, facecolor=color, edgecolor='black',
                   linewidths=1.2, zorder=11)

    ax.set_xlim(xs[0] - cell_w / 2, xs[-1] + cell_w / 2)
    ax.set_ylim(ys[0] - cell_h / 2, ys[-1] + cell_h / 2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title(title if title is not None else f"Turn {turn_idx}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def clear_frame_dir(frames_dir):
    """Delete existing frame_*.png from frames_dir (creates the dir if absent)."""
    p = Path(frames_dir)
    p.mkdir(parents=True, exist_ok=True)
    for f in p.glob("frame_*.png"):
        f.unlink()


def clear_debug_dir(debug_dir):
    """Delete existing replan_*.png from debug_dir (creates the dir if absent)."""
    p = Path(debug_dir)
    p.mkdir(parents=True, exist_ok=True)
    for f in p.glob("replan_*.png"):
        f.unlink()


def render_replan_debug_frame(env_map, blocked_env_graph, agents, replan_idx,
                              turn_idx, output_path, agent_colors=None,
                              targets=None):
    """Render a debug frame showing every (agent, remaining target) shortest path
    as a dashed line in the agent's color, on top of the regular simulation view.

    Saved separately from the MP4 frames.
    """
    import networkx as nx  # local to keep module-level lazy

    if targets is not None:
        remaining = [tgt.position for tgt in targets if not tgt.reached]
    else:
        remaining = [
            n for n, d in env_map.nodes(data=True) if d.get("type") == "target_unreached"
        ]
    extras = []
    for i, agent in enumerate(agents):
        color = _agent_color(i, agent_colors)
        for t in remaining:
            try:
                path = nx.shortest_path(env_map, agent.position, t, weight="distance")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            extras.append((path, color, "dashed", 2.0))

    title = f"Replan #{replan_idx} @ turn {turn_idx} ({len(remaining)} targets remaining)"
    render_simulation_frame(
        env_map, blocked_env_graph, agents, turn_idx, output_path,
        agent_colors=agent_colors, title=title, extra_paths=extras,
        targets=targets,
    )


def make_mp4_from_frames(frames_dir, output_mp4, fps=4):
    """Combine frame_*.png in frames_dir into an MP4 at the given fps via ffmpeg."""
    frames = sorted(Path(frames_dir).glob("frame_*.png"))
    if not frames:
        return None
    pattern = str(Path(frames_dir) / "frame_%04d.png")
    Path(output_mp4).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_mp4
