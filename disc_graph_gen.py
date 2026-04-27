"""
Graph Generator for Exact Learning of Weighted Graphs Paper

Generates disconnected weighted graphs with:
- Pareto-distributed edge weights (CDF: F(w) = 1 - w^{-α})
- Bounded maximum degree D
- Configurable number and size of components
- Proper structure for testing LBL-R algorithm

Paper Reference:
  "Exact Learning of Weighted Graphs Using Composite Queries"
  Goodrich, Liu, Panageas (arXiv:2511.14882v1)
"""

import networkx as nx
import numpy as np
import random
from typing import Tuple, List


def pareto_weight(alpha=2.0, scale=1.0):
    """
    Generate a single weight from Pareto distribution.
    
    CDF: F(w) = 1 - w^{-α} for w ∈ [scale, ∞)
    
    Args:
        alpha: Shape parameter (higher = more concentrated near scale)
               Paper suggests α > 0, typically α ≥ 2
        scale: Minimum value (paper uses scale=1)
    
    Returns:
        Single weight value >= scale
    
    Theory:
        The Pareto distribution is critical for Lemma 6 and Theorem 10.
        It enables early termination because Pr(w ≥ Wthr) = T(Wthr) = Wthr^{-α}
    """
    U = np.random.uniform(0, 1)
    weight = scale / (U ** (1.0 / alpha))
    return max(weight, scale)


def generate_pareto_weights(n_weights, alpha=2.0, scale=1.0):
    """
    Generate multiple weights from Pareto distribution.
    
    Args:
        n_weights: Number of weights to generate
        alpha: Shape parameter
        scale: Minimum value
    
    Returns:
        List/array of weights
    """
    U = np.random.uniform(0, 1, n_weights)
    weights = scale / (U ** (1.0 / alpha))
    return np.maximum(weights, scale)


def simple_random_tree(n, seed=None, max_degree=None):
    """
    Generate a random tree with n vertices respecting degree bounds.
    
    Creates a connected tree where each vertex i (i >= 1) connects to a
    random vertex in [0, i-1] that hasn't reached max_degree.
    
    Args:
        n: Number of vertices (0, 1, ..., n-1)
        seed: Random seed for reproducibility
        max_degree: Maximum degree bound (None = no constraint)
    
    Returns:
        networkx.Graph representing the tree
    """
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    for i in range(1, n):
        if max_degree is None:
            # No constraint: connect to any previous vertex
            parent = rng.randint(0, i - 1)
            G.add_edge(i, parent)
        else:
            # Respect degree bound: find a valid parent
            candidates = [v for v in range(i) if G.degree(v) < max_degree]
            if candidates:
                parent = rng.choice(candidates)
                G.add_edge(i, parent)
            else:
                # Fallback: connect to any vertex (worst case)
                parent = rng.randint(0, i - 1)
                G.add_edge(i, parent)
    
    return G


def generate_bounded_degree_component(
    n_vertices,
    alpha=2.0,
    extra_edges_per_component=2,
    max_degree=4,
    seed=None
):
    """
    Generate a single connected component with bounded degree.
    
    Process:
    1. Start with a random tree (connected, n-1 edges)
    2. Add extra_edges_per_component edges while respecting max_degree
    3. Assign Pareto-distributed weights to all edges
    
    Args:
        n_vertices: Number of vertices in component
        alpha: Pareto shape parameter
        extra_edges_per_component: Additional edges beyond tree edges
        max_degree: Maximum degree bound (D parameter from paper)
        seed: Random seed
    
    Returns:
        networkx.Graph with weighted edges
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Start with a tree (respecting degree bounds)
    G = simple_random_tree(n_vertices, seed=seed, max_degree=max_degree)
    
    # Add extra edges while respecting degree bound
    possible_edges = list(nx.non_edges(G))
    random.shuffle(possible_edges)
    
    added = 0
    for u, v in possible_edges:
        if added >= extra_edges_per_component:
            break
        # Check degree constraint
        if G.degree(u) < max_degree and G.degree(v) < max_degree:
            G.add_edge(u, v)
            added += 1
    
    # Assign Pareto-distributed weights
    edge_list = list(G.edges())
    weights = generate_pareto_weights(len(edge_list), alpha=alpha, scale=1.0)
    
    for (u, v), w in zip(edge_list, weights):
        G[u][v]['weight'] = float(w)
    
    return G


def generate_disconnected_graph(
    num_components=50,
    component_size=20,
    alpha=2.0,
    extra_edges_per_component=2,
    max_degree=4,
    seed=42,
    verbose=False
):
    """
    Generate a disconnected weighted graph for testing LBL-R algorithm.
    
    This is the PRIMARY graph type tested in the paper's main result.
    The graph consists of multiple disconnected components, each with:
    - Bounded maximum degree D
    - Pareto-distributed edge weights
    - Connected structure
    
    Args:
        num_components: Number of disconnected components (k in paper)
        component_size: Vertices per component (typically 10-20)
        alpha: Pareto shape parameter (paper uses α > 0, typically ≥ 2)
        extra_edges_per_component: Non-tree edges per component
        max_degree: Maximum degree bound (D parameter)
        seed: Random seed for reproducibility
        verbose: Print progress information
    
    Returns:
        Tuple: (G, metadata)
        - G: networkx.Graph (disconnected, weighted)
        - metadata: dict with graph properties
    
    Theory Connection:
        - Paper Algorithm 1 (LBL-R) is designed for exactly this case
        - n = num_components * component_size (total vertices)
        - k = num_components (connected components)
        - Maximum degree = max_degree
        - Weights follow Pareto(α, scale=1)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if verbose:
        print(f"Generating {num_components} disconnected components...")
        print(f"  Component size: {component_size}")
        print(f"  Pareto parameter α: {alpha}")
        print(f"  Max degree: {max_degree}")
    
    components = []
    total_edges = 0
    
    for comp_idx in range(num_components):
        if verbose and (comp_idx + 1) % max(1, num_components // 10) == 0:
            print(f"  Generated {comp_idx + 1}/{num_components} components...")
        
        # Generate component with unique seed for reproducibility
        comp_seed = seed + comp_idx if seed is not None else None
        Gc = generate_bounded_degree_component(
            n_vertices=component_size,
            alpha=alpha,
            extra_edges_per_component=extra_edges_per_component,
            max_degree=max_degree,
            seed=comp_seed
        )
        
        components.append(Gc)
        total_edges += Gc.number_of_edges()
    
    # Merge all components into one disconnected graph
    G = nx.disjoint_union_all(components)
    
    # Compute metadata
    metadata = {
        'num_components': nx.number_connected_components(G),
        'num_vertices': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'max_degree': max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0,
        'min_degree': min(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0,
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        'alpha': alpha,
        'component_size': component_size,
        'extra_edges_per_component': extra_edges_per_component,
        'seed': seed,
    }
    
    # Compute weight statistics
    if G.number_of_edges() > 0:
        weights = [d['weight'] for u, v, d in G.edges(data=True)]
        metadata['weight_min'] = min(weights)
        metadata['weight_max'] = max(weights)
        metadata['weight_mean'] = np.mean(weights)
        metadata['weight_median'] = np.median(weights)
        metadata['weight_std'] = np.std(weights)
    
    if verbose:
        print(f"\nGraph generated successfully!")
        print(f"  Total vertices: {metadata['num_vertices']}")
        print(f"  Total edges: {metadata['num_edges']}")
        print(f"  Connected components: {metadata['num_components']}")
        print(f"  Degree range: [{metadata['min_degree']}, {metadata['max_degree']}]")
        print(f"  Avg degree: {metadata['avg_degree']:.2f}")
        print(f"  Weight range: [{metadata['weight_min']:.2f}, {metadata['weight_max']:.2f}]")
        print(f"  Weight mean: {metadata['weight_mean']:.2f}")
    
    return G, metadata


def generate_small_test_graph(seed=42):
    """
    Generate a small test graph for quick validation.
    
    Structure:
    - 2 components of 5 vertices each
    - Simple tree + 1 extra edge per component
    - Pareto weights
    
    Good for:
    - Testing find_connected_components()
    - Quick algorithm validation
    - Debugging
    """
    return generate_disconnected_graph(
        num_components=2,
        component_size=5,
        alpha=2.0,
        extra_edges_per_component=1,
        max_degree=4,
        seed=seed,
        verbose=False
    )



def connected_graph_from_components(G_disconnected):
    """
    Convert a disconnected graph to connected by adding bridges.
    
    Useful for comparing with algorithms designed for connected graphs.
    Adds minimum edges between components to make fully connected.
    
    Args:
        G_disconnected: Disconnected networkx.Graph
    
    Returns:
        Connected graph with added bridge edges
    """
    G = G_disconnected.copy()
    components = list(nx.connected_components(G))
    
    if len(components) <= 1:
        return G  # Already connected
    
    # Add edges between components
    for i in range(len(components) - 1):
        comp1 = list(components[i])
        comp2 = list(components[i + 1])
        
        # Add edge from first node of comp1 to first node of comp2
        u, v = comp1[0], comp2[0]
        w = pareto_weight(alpha=2.0, scale=1.0)
        G.add_edge(u, v, weight=w)
    
    return G


def print_graph_info(G, metadata=None):
    """
    Print detailed information about a graph.
    
    Args:
        G: networkx.Graph
        metadata: Optional metadata dict from generate_disconnected_graph()
    """
    print("=" * 70)
    print("GRAPH INFORMATION")
    print("=" * 70)
    
    print(f"Vertices: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Connected components: {nx.number_connected_components(G)}")
    
    if G.number_of_nodes() > 0:
        degrees = [d for n, d in G.degree()]
        print(f"Degree: min={min(degrees)}, max={max(degrees)}, "
              f"mean={np.mean(degrees):.2f}, median={np.median(degrees):.1f}")
    
    if G.number_of_edges() > 0:
        weights = [d['weight'] for u, v, d in G.edges(data=True)]
        print(f"Edge weights: min={min(weights):.3f}, max={max(weights):.3f}, "
              f"mean={np.mean(weights):.3f}, median={np.median(weights):.3f}")
    
    # if metadata:
    #     print(f"\nMetadata:")
    #     for key, value in metadata.items():
    #         if isinstance(value, float):
    #             print(f"  {key}: {value:.4f}")
    #         else:
    #             print(f"  {key}: {value}")
    
    print("=" * 70)


# Example usage
if __name__ == "__main__":
    # Generate small test graph
    print("\n1. Small test graph (2 components, 5 vertices each):")
    G_small, meta_small = generate_small_test_graph(seed=42)
    print_graph_info(G_small, meta_small)
    
    # Generate medium test graph
    print("\n2. Medium test graph (10 components, 50 vertices each):")
    G_medium, meta_medium = generate_disconnected_graph(
        num_components=10,
        component_size=50,
        alpha=2.0,
        extra_edges_per_component=2,
        max_degree=4,
        seed=42,
        verbose=True
    )
    print_graph_info(G_medium, meta_medium)
    
    # Verify Pareto distribution
    print("\n3. Verifying Pareto distribution properties:")
    weights = [d['weight'] for u, v, d in G_medium.edges(data=True)]
    print(f"   Min weight: {min(weights):.3f} (should be >= 1)")
    print(f"   Median weight: {np.median(weights):.3f}")
    print(f"   Mean weight: {np.mean(weights):.3f}")
    print(f"   99th percentile: {np.percentile(weights, 99):.3f}")
    print(f"   Total edges: {len(weights)}")
    
    # Verify component structure
    print("\n4. Verifying disconnected structure:")
    components = list(nx.connected_components(G_medium))
    print(f"   Number of components: {len(components)}")
    component_sizes = sorted([len(c) for c in components])
    print(f"   Component sizes: {component_sizes[:5]}... (showing first 5)")
    
    # Verify degree bounds
    print("\n5. Verifying degree bounds:")
    degrees = [G_medium.degree(n) for n in G_medium.nodes()]
    print(f"   Max degree: {max(degrees)} (should be <= 4)")
    print(f"   Min degree: {min(degrees)}")
    print(f"   Degree distribution:")
    for d in range(1, max(degrees) + 1):
        count = sum(1 for deg in degrees if deg == d)
        print(f"     Degree {d}: {count} vertices")
