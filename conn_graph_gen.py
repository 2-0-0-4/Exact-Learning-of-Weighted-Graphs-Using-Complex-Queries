"""
Connected Graph Generator for Exact Learning of Weighted Graphs Paper

Generates CONNECTED weighted graphs with:
- Pareto-distributed edge weights (CDF: F(w) = 1 - w^{-α})
- Bounded maximum degree D
- Configurable size and density
- Proper structure for testing NT-R / LBL-R on connected graphs

These graphs correspond to the second row of Table 1 in the paper:
  "Exact Learning of Weighted Graphs Using Composite Queries"
  Goodrich, Liu, Panageas (arXiv:2511.14882v1)

Query complexity for connected graphs: O~(D³ · Wmax · n^{3/2})
Compare with disconnected: O~((1 + (1/α)·log D) · D³ · n^{3/2})
"""

import networkx as nx
import numpy as np
import random
import math
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Weight generation (shared with graph_generator.py)
# ---------------------------------------------------------------------------

def pareto_weight(alpha: float = 2.0, scale: float = 1.0) -> float:
    """
    Sample one weight from the Pareto distribution.
    CDF: F(w) = 1 - w^{-α},  w ∈ [scale, ∞).

    Args:
        alpha: Shape parameter (paper uses α > 0; Theorem 10 requires α > 0).
        scale: Minimum weight (paper assumes weights ≥ 1, so scale=1).

    Returns:
        A float weight ≥ scale.
    """
    U = np.random.uniform(0.0, 1.0)
    return max(scale / (U ** (1.0 / alpha)), scale)


def generate_pareto_weights(
    n_weights: int,
    alpha: float = 2.0,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Vectorised Pareto weight generation.

    Args:
        n_weights: How many weights to draw.
        alpha: Pareto shape parameter.
        scale: Minimum weight (= 1 in the paper).

    Returns:
        NumPy array of length n_weights with values ≥ scale.
    """
    U = np.random.uniform(0.0, 1.0, n_weights)
    return np.maximum(scale / (U ** (1.0 / alpha)), scale)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _random_spanning_tree(
    n: int,
    rng: random.Random,
    max_degree: Optional[int] = None,
) -> nx.Graph:
    """
    Build a random labelled tree on vertices {0, …, n-1} that respects the
    degree bound.  Each vertex i ≥ 1 is attached to a random vertex in
    {0, …, i-1} that has not yet reached max_degree.

    If every feasible parent is saturated (very tight degree bound) we fall
    back to connecting to the least-degree vertex so the tree is always
    connected.

    Args:
        n:          Number of vertices.
        rng:        Random instance (for reproducibility).
        max_degree: Maximum allowed degree, or None for no constraint.

    Returns:
        A connected nx.Graph tree with n vertices and n-1 edges (no weights).
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(1, n):
        if max_degree is None:
            parent = rng.randint(0, i - 1)
        else:
            candidates = [v for v in range(i) if G.degree(v) < max_degree]
            if candidates:
                parent = rng.choice(candidates)
            else:
                # Degree constraint is too tight; pick the lowest-degree vertex
                parent = min(range(i), key=lambda v: G.degree(v))
        G.add_edge(i, parent)

    return G


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_connected_graph(
    n_vertices: int,
    alpha: float = 2.0,
    extra_edges: int = 2,
    max_degree: int = 4,
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Generate a single CONNECTED weighted graph.

    Construction:
      1. Build a random spanning tree (guarantees connectivity).
      2. Add `extra_edges` non-tree edges while honouring the degree cap.
      3. Assign independent Pareto(, scale=1) weights to every edge.

    Args:
        n_vertices:  Total number of vertices.
        alpha:       Pareto shape parameter for edge weights.
        extra_edges: Number of non-tree edges to add (controls density).
        max_degree:  Maximum vertex degree (D parameter from paper).
        seed:        RNG seed for full reproducibility.

    Returns:
        A connected nx.Graph with 'weight' attributes on every edge.

    Note:
        The graph is always connected because it starts from a spanning tree.
        Maximum degree is bounded by max_degree (though the spanning-tree
        fallback may slightly exceed it when max_degree is very small).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    rng = random.Random(seed)

    # Step 1: spanning tree
    G = _random_spanning_tree(n_vertices, rng, max_degree=max_degree)

    # Step 2: add extra edges (non-tree, respecting degree cap)
    non_edges = list(nx.non_edges(G))
    rng.shuffle(non_edges)
    added = 0
    for u, v in non_edges:
        if added >= extra_edges:
            break
        if G.degree(u) < max_degree and G.degree(v) < max_degree:
            G.add_edge(u, v)
            added += 1

    # Step 3: assign Pareto weights
    edge_list = list(G.edges())
    weights = generate_pareto_weights(len(edge_list), alpha=alpha, scale=1.0)
    for (u, v), w in zip(edge_list, weights):
        G[u][v]['weight'] = float(w)

    return G


def generate_connected_graph_suite(
    n: int,
    alpha: float = 2.0,
    extra_edges: int = 2,
    max_degree: int = 4,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[nx.Graph, dict]:
    """
    Generate a connected graph and return it together with a metadata dict.

    This mirrors the interface of `generate_disconnected_graph` in
    graph_generator.py so both can be used interchangeably in test scripts.

    Args:
        n:            Total vertices.
        alpha:        Pareto shape parameter.
        extra_edges:  Non-tree edges to add.
        max_degree:   Degree cap.
        seed:         RNG seed.
        verbose:      Print summary statistics.

    Returns:
        (G, metadata) where metadata mirrors the dict from graph_generator.py.
    """
    G = generate_connected_graph(
        n_vertices=n,
        alpha=alpha,
        extra_edges=extra_edges,
        max_degree=max_degree,
        seed=seed,
    )

    degrees = [d for _, d in G.degree()]
    weights_list = [d['weight'] for _, _, d in G.edges(data=True)]

    metadata = {
        'num_components': 1,          # always 1 – this is the point
        'num_vertices': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'max_degree': max(degrees),
        'min_degree': min(degrees),
        'avg_degree': float(np.mean(degrees)),
        'alpha': alpha,
        'extra_edges': extra_edges,
        'seed': seed,
        'weight_min': float(min(weights_list)),
        'weight_max': float(max(weights_list)),
        'weight_mean': float(np.mean(weights_list)),
        'weight_median': float(np.median(weights_list)),
        'weight_std': float(np.std(weights_list)),
    }

    if verbose:
        print(f"\nConnected graph generated:")
        print(f"  Vertices   : {metadata['num_vertices']}")
        print(f"  Edges      : {metadata['num_edges']}")
        print(f"  Components : {metadata['num_components']}")
        print(f"  Degree     : min={metadata['min_degree']}, "
              f"max={metadata['max_degree']}, "
              f"avg={metadata['avg_degree']:.2f}")
        print(f"  Wmax       : {metadata['weight_max']:.3f}")
        print(f"  Weight mean: {metadata['weight_mean']:.3f}")
        print(f"  alpha      : {alpha}")

    return G, metadata


def print_graph_info(G: nx.Graph, metadata: Optional[dict] = None) -> None:
    """Pretty-print graph statistics (mirrors graph_generator.print_graph_info)."""
    print("=" * 70)
    print("GRAPH INFORMATION")
    print("=" * 70)
    print(f"Vertices           : {G.number_of_nodes()}")
    print(f"Edges              : {G.number_of_edges()}")
    print(f"Connected components: {nx.number_connected_components(G)}")

    degrees = [d for _, d in G.degree()]
    print(f"Degree             : min={min(degrees)}, max={max(degrees)}, "
          f"mean={np.mean(degrees):.2f}")

    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    print(f"Edge weights       : min={min(weights):.3f}, max={max(weights):.3f}, "
          f"mean={np.mean(weights):.3f}, median={np.median(weights):.3f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Connected Graph Generator - self-test ===\n")

    print("1. Small connected graph (n=10, D=3, alpha=2.0)")
    G_small, meta = generate_connected_graph_suite(
        n=10, alpha=2.0, extra_edges=2, max_degree=3, seed=0, verbose=True
    )
    print_graph_info(G_small)
    assert nx.is_connected(G_small), "Graph is not connected!"
    print("   Connected\n")

    print("2. Medium connected graph (n=100, D=4, alpha=2.0)")
    G_med, meta = generate_connected_graph_suite(
        n=100, alpha=2.0, extra_edges=3, max_degree=4, seed=42, verbose=True
    )
    print_graph_info(G_med)
    assert nx.is_connected(G_med), "Graph is not connected!"
    max_d = max(d for _, d in G_med.degree())
    print(f"   Connected | max degree = {max_d}\n")

    print("3. Pareto weight distribution check (n=500, alpha=2.0)")
    G_big, _ = generate_connected_graph_suite(n=500, alpha=2.0, seed=7)
    ws = [d['weight'] for _, _, d in G_big.edges(data=True)]
    print(f"   Min weight : {min(ws):.4f}  (should be ≥ 1)")
    print(f"   Mean weight: {np.mean(ws):.4f}")
    print(f"   Max weight : {max(ws):.4f}")
    assert min(ws) >= 1.0, "Weight below 1!"
    print("   All weights ≥ 1\n")

    print("4. Degree constraint check (n=50, D=3)")
    G_deg, _ = generate_connected_graph_suite(
        n=50, alpha=2.0, extra_edges=5, max_degree=3, seed=1
    )
    max_d = max(d for _, d in G_deg.degree())
    print(f"   Max degree achieved: {max_d}  (cap = 3)")
    print(f"   Degree check complete\n")

    print("All self-tests passed.")