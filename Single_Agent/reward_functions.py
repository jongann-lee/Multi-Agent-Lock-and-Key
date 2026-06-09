import numpy as np
import networkx as nx


# Canonical sentinel `lock` value: a target whose lock the planner has not yet
# observed. Real lock IDs are non-negative integers indexed from 0; -1 can
# never collide with one. Defined here (the lowest layer that needs it) and
# re-exported by Multi_Agent.finite_horizon_MA so the Single_Agent reward code
# stays decoupled from the Multi_Agent package (avoids a circular import).
UNKNOWN_LOCK = -1


def edge_visibility_reward(env_graph: nx.Graph, input_node) -> float:
    """
    Defines a reward function on each node of the graph based on
    the combined value of the visible, unexplored edges from that node.
    """

    visible_edges = env_graph.nodes[input_node]["visible_edges"]
    visible_unexplored_edges = [edge for edge in visible_edges if env_graph.edges[edge]["observed_edge"] == False]
    edge_value = np.array([env_graph.edges[edge]["num_used"] for edge in visible_unexplored_edges])
    edge_visibility_reward = np.sum(edge_value)

    return edge_visibility_reward


def target_observation_reward(env_graph: nx.Graph, input_node) -> int:
    """Number of targets visible from `input_node` whose lock is still unknown
    and not yet credited as observed.

    The lock-and-key analogue of `visibility_reward`: instead of valuing
    unobserved *edges* (by `num_used`), it values unobserved *targets* — each
    one whose lock is revealed is worth a flat unit. A "fixed bonus per
    target" policy multiplies this count by its own `target_reward_ratio`.

    A node counts iff it is a target (`lock` attribute present) whose lock is
    still UNKNOWN_LOCK (not yet revealed to the planner) and whose
    `observed_target` flag is not already set. The caller is responsible for
    setting `observed_target` on the counted nodes afterward, mirroring the
    `observed_edge` bookkeeping, so the submodular greedy never double-counts.
    """
    visible_nodes = env_graph.nodes[input_node].get("visible_nodes", [])
    count = 0
    for n in visible_nodes:
        nd = env_graph.nodes[n]
        if ("lock" in nd
                and nd["lock"] == UNKNOWN_LOCK
                and not nd.get("observed_target", False)):
            count += 1
    return count


def target_and_visibility_reward(env_graph: nx.Graph, input_node, unreached_targets) -> float:
    """
    Defines a reward function on each node of the graph based on
    1. The number of remaining targets in the environment.
    2. The combined value of the visible, unexplored edges from that node
    """

    reward_ratio = 0.0 # Weighting factor between target count and visibility

    if input_node in unreached_targets:
        target_reward = -1 * len(unreached_targets) + 1
    else:
        target_reward = -1 * len(unreached_targets) 

    visible_edges = env_graph.nodes[input_node]["visible_edges"]
    visible_unexplored_edges = [edge for edge in visible_edges if env_graph.edges[edge]["observed_edge"] == False]
    edge_value = np.array([env_graph.edges[edge]["num_used"] for edge in visible_unexplored_edges])
    visibility_reward = np.sum(edge_value)

    total_reward = target_reward + reward_ratio * visibility_reward

    return total_reward

    
