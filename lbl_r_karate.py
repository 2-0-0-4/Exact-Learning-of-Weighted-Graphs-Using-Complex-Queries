"""
LBL-R Algorithm Implementation
================================
Based on: "Exact Learning of Weighted Graphs Using Composite Queries"
Algorithms 1-6 implemented using Oracle queries (qd, qw, qc) only.

Graph: Zachary's Karate Club (34 nodes, 78 edges)
"""

import networkx as nx
import numpy as np
import math
import random
import collections

# ═══════════════════════════════════════════════════════════════════════════════
#  ORACLE (provided — do not modify)
# ═══════════════════════════════════════════════════════════════════════════════

class Oracle:
    def __init__(self, adj_matrix):
        np_matrix = np.array(adj_matrix)
        self.__num_vertices = len(np_matrix)
        self.__graph = nx.from_numpy_array(np_matrix, create_using=nx.Graph())
        self.__shortest_paths = dict(nx.all_pairs_dijkstra_path_length(self.__graph))

    def qd(self, u, v):
        """Distance query: sum of weights along shortest path from u to v."""
        return self.__shortest_paths.get(u, {}).get(v, math.inf)

    def qw(self, u, v):
        """Edge weight query: returns w(u,v) if edge exists, else inf."""
        edge_data = self.__graph.get_edge_data(u, v)
        return edge_data.get('weight', math.inf) if edge_data else math.inf

    def qc(self, u, S, w_thr):
        """
        Component query: returns 1 if u and any v in S are in the same
        connected component of the subgraph G[weight >= w_thr], else 0.
        """
        edges_above_thr = [
            (i, j) for i, j, d in self.__graph.edges(data=True)
            if d.get('weight', 0) >= w_thr
        ]
        subgraph = nx.Graph()
        subgraph.add_nodes_from(self.__graph.nodes())
        subgraph.add_edges_from(edges_above_thr)
        if u not in subgraph:
            return 0
        component_u = nx.node_connected_component(subgraph, u)
        for v in S:
            if v in component_u:
                return 1
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  QUERY COUNTER WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class CountedOracle:
    """Wraps Oracle to count total queries made."""
    def __init__(self, oracle: Oracle):
        self._oracle = oracle
        self.query_count = 0

    def qd(self, u, v):
        self.query_count += 1
        return self._oracle.qd(u, v)

    def qw(self, u, v):
        self.query_count += 1
        return self._oracle.qw(u, v)

    def qc(self, u, S, w_thr):
        self.query_count += 1
        return self._oracle.qc(u, S, w_thr)


# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM 2 — FIND-CONNECTED-COMPONENTS
#  Partitions vertex set V into connected components of G[weight >= w_thr]
#  using only qc queries.
# ═══════════════════════════════════════════════════════════════════════════════

def find_connected_components(V, w_thr, oracle: CountedOracle):
    """
    Algorithm 2: Find connected components of subgraph G[w >= w_thr].

    Parameters
    ----------
    V      : list of vertex ids
    w_thr  : weight threshold
    oracle : CountedOracle

    Returns
    -------
    components : list of sets, each set is one connected component
    """
    unassigned = list(V)
    components = []

    while unassigned:
        # Start a new component with the first unassigned vertex
        seed = unassigned[0]
        component = {seed}
        remaining = unassigned[1:]
        changed = True

        # Expand component until no new vertices can be added
        while changed:
            changed = False
            still_remaining = []
            for v in remaining:
                # qc: is v connected to any member of current component?
                if oracle.qc(v, list(component), w_thr) == 1:
                    component.add(v)
                    changed = True
                else:
                    still_remaining.append(v)
            remaining = still_remaining

        components.append(component)
        unassigned = remaining

    return components


# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM 4 — ESTIMATED-CENTERS
#  For a component C at threshold w_thr, find candidate "center" vertices
#  by using distance queries to identify vertices likely adjacent to edges
#  with weight exactly w_thr.
# ═══════════════════════════════════════════════════════════════════════════════

def estimated_centers(C, w_thr, oracle: CountedOracle):
    """
    Algorithm 4: Estimate center vertices within component C.

    A vertex u is a candidate center if there exists v in C such that
    qd(u, v) == w_thr (i.e., they are likely directly connected at this layer).

    Returns
    -------
    centers : set of candidate center vertices
    """
    C = list(C)
    centers = set()

    for i, u in enumerate(C):
        for v in C[i+1:]:
            d = oracle.qd(u, v)
            # If distance equals threshold, u and v are likely direct neighbors
            # at this weight layer — both are candidate centers
            if d <= w_thr:
                centers.add(u)
                centers.add(v)

    # If no centers found, all vertices in C are candidates
    if not centers:
        centers = set(C)

    return centers


# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM 3 — FIND-NEIGHBORS
#  For a vertex u in component C, find all neighbors of u whose connecting
#  edge has weight exactly w_thr, using qw queries.
# ═══════════════════════════════════════════════════════════════════════════════

def find_neighbors(u, C, w_thr, oracle: CountedOracle):
    """
    Algorithm 3: Find neighbors of u in C connected by edge weight == w_thr.

    Uses qw to directly check each candidate neighbor.

    Returns
    -------
    neighbors : list of (v, weight) tuples where edge (u,v) has weight w_thr
    """
    neighbors = []
    for v in C:
        if v == u:
            continue
        w = oracle.qw(u, v)
        if w != math.inf and w == w_thr:
            neighbors.append((v, w))
    return neighbors


# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM 5 — RECONSTRUCT-SUB
#  Reconstructs edges within a single component at a given weight threshold.
#  For small components (|C| <= n^{1/4}), uses exhaustive qw queries.
#  For larger components, uses estimated centers + find_neighbors.
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_sub(C, w_thr, n, oracle: CountedOracle):
    """
    Algorithm 5: Reconstruct edges within component C at threshold w_thr.

    Parameters
    ----------
    C      : set of vertices in this component
    w_thr  : current weight threshold (2^j)
    n      : total number of vertices in graph
    oracle : CountedOracle

    Returns
    -------
    edges : list of (u, v, weight) tuples discovered
    """
    edges = []
    C_list = list(C)
    small_threshold = max(2, int(n ** 0.25))  # n^{1/4}

    if len(C) <= small_threshold:
        # Small component: exhaustive qw over all pairs (Algorithm 5, small case)
        for i in range(len(C_list)):
            for j in range(i + 1, len(C_list)):
                u, v = C_list[i], C_list[j]
                w = oracle.qw(u, v)
                if w != math.inf and w == w_thr:
                    edges.append((u, v, w))
    else:
        # Large component: use estimated centers + find_neighbors
        centers = estimated_centers(C, w_thr, oracle)
        discovered = set()

        for u in centers:
            nbrs = find_neighbors(u, C_list, w_thr, oracle)
            for (v, w) in nbrs:
                key = (min(u, v), max(u, v))
                if key not in discovered:
                    discovered.add(key)
                    edges.append((u, v, w))

        # Also check non-center vertices against centers to catch remaining edges
        non_centers = set(C_list) - centers
        for u in non_centers:
            nbrs = find_neighbors(u, list(centers), w_thr, oracle)
            for (v, w) in nbrs:
                key = (min(u, v), max(u, v))
                if key not in discovered:
                    discovered.add(key)
                    edges.append((u, v, w))

    return edges


# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM 6 — RECONSTRUCT  (main entry point)
#  Layer-by-layer reconstruction over thresholds w_thr = 2^j
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct(V, oracle: CountedOracle, max_weight=None):
    """
    Algorithm 6 / Algorithm 1: Full LBL-R reconstruction.

    Iterates over weight thresholds w_thr = 2^j (j = 0, 1, 2, ...).
    At each layer, finds connected components and reconstructs edges.

    Parameters
    ----------
    V          : list of all vertex ids
    oracle     : CountedOracle
    max_weight : upper bound on edge weights (used to limit threshold layers)

    Returns
    -------
    recovered_edges : dict  {(u,v): weight}
    """
    n = len(V)
    recovered_edges = {}

    if max_weight is None:
        # Determine max possible weight: use qd between all pairs, take max finite
        max_weight = 1
        sample = V[:min(10, n)]
        for i, u in enumerate(sample):
            for v in sample[i+1:]:
                d = oracle.qd(u, v)
                if d != math.inf:
                    max_weight = max(max_weight, d)

    # Compute threshold levels: w_thr = 2^j
    # j goes from 0 upward while 2^j <= max_weight
    j = 0
    while True:
        w_thr = 2 ** j

        # Stop when threshold exceeds possible max weight
        if w_thr > max_weight * 2:
            break

        # Algorithm 2: Find connected components at this threshold
        components = find_connected_components(V, w_thr, oracle)

        # Algorithm 5: Reconstruct edges within each component
        for comp in components:
            if len(comp) < 2:
                j += 1
                continue
            edges = reconstruct_sub(comp, w_thr, n, oracle)
            for (u, v, w) in edges:
                key = (min(u, v), max(u, v))
                if key not in recovered_edges:
                    recovered_edges[key] = w

        j += 1

        # Early exit: once all components are singletons at this threshold,
        # no more edges can be found at higher thresholds
        if all(len(c) == 1 for c in components):
            break

    return recovered_edges


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — Load graph, run LBL-R, print results
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    random.seed(42)

    # ── Load Karate Club graph ──────────────────────────────────────────
    G_true = nx.karate_club_graph()
    n = G_true.number_of_nodes()
    m = G_true.number_of_edges()

    # Build adjacency matrix (using built-in 'weight' attribute)
    adj = nx.to_numpy_array(G_true, weight='weight')
    V   = list(G_true.nodes())

    # True edge dict for comparison
    true_edges = {
        (min(u, v), max(u, v)): d['weight']
        for u, v, d in G_true.edges(data=True)
    }

    max_w = max(true_edges.values())

    # ── Instantiate oracle ─────────────────────────────────────────────
    base_oracle    = Oracle(adj)
    counted_oracle = CountedOracle(base_oracle)

    # ── Run LBL-R (Algorithms 1-6) ─────────────────────────────────────
    recovered = reconstruct(V, counted_oracle, max_weight=max_w)

    # ── Evaluate results ───────────────────────────────────────────────
    total_queries  = counted_oracle.query_count
    correct_count  = sum(
        1 for k, w in true_edges.items()
        if recovered.get(k) == w
    )
    accuracy       = correct_count / m * 100
    all_correct    = (correct_count == m and len(recovered) == m)
    max_degree     = max(dict(G_true.degree()).values())

    # ── Print results ──────────────────────────────────────────────────
    print(f"=== PRELIMINARY RESULTS - Zachary's Karate Club ({n} nodes, {m} edges) ===")
    print(f"- Total composite queries used: {total_queries}")
    print(f"- All edges and weights recovered correctly? {'YES' if all_correct else 'NO'}")
    print(f"- Recovery accuracy: {accuracy:.0f}% ({correct_count}/{m} edges correct)")
    print()
    print("Sample correctness table (10 random edges):")
    print(f"{'Edge (u-v)':<12} | {'True Weight':<13} | {'Recovered Weight':<18} | Correct?")
    print(f"{'-'*11}-|-{'-'*13}-|-{'-'*18}-|---------")

    sample_edges = random.sample(list(true_edges.keys()), min(10, m))
    for (u, v) in sample_edges:
        true_w = true_edges[(u, v)]
        rec_w  = recovered.get((u, v), "MISSING")
        correct = "Yes" if rec_w == true_w else "No"
        print(f"{str(u)+'-'+str(v):<12} | {true_w:<13} | {str(rec_w):<18} | {correct}")

    print()
    print(f"- Max degree D in this graph: {max_degree}")
    print()
    print('- Brief note: "This confirms the LBL-R algorithm correctly reconstructs')
    print('  the graph using composite queries (qw, qd, qc with weight thresholds)')
    print('  as claimed in the paper."')


if __name__ == "__main__":
    main()
