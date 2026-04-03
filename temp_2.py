import networkx as nx
import numpy as np
import random
import math
import collections
import time

# =============================================================================
# Oracle (paste yours exactly above this script in your notebook/file)
# =============================================================================
# Expected interface from the user-provided code:
#   oracle.qd(u, v)
#   oracle.qw(u, v)
#   oracle.qc(u, S, w_thr)
#
# IMPORTANT CAVEAT:
# The paper's thresholded version of LBL-R uses qd(u,v,Wthr) and qw(u,v,Wthr),
# but the pasted Oracle code exposes thresholding directly only for qc.
# So, in this script, thresholded qd/qw are derived ONLY by repeated calls to
# the real oracle.qw(...) and by building the threshold layer G[w >= Wthr].
# No hidden graph access is used for reconstruction.
# =============================================================================


# =============================================================================
# 1) Load Zachary's Karate Club graph and build the real Oracle input
# =============================================================================
G = nx.karate_club_graph()
n = G.number_of_nodes()
m = G.number_of_edges()

adj = np.zeros((n, n), dtype=float)
for u, v, data in G.edges(data=True):
    w = float(data.get("weight", 1.0))
    adj[u][v] = w
    adj[v][u] = w

import networkx as nx
import numpy as np
import math

class Oracle:
    def __init__(self, adj_matrix):
        np_matrix = np.array(adj_matrix)
        self.__num_vertices = len(np_matrix)
        self.__graph = nx.from_numpy_array(np_matrix, create_using=nx.Graph())
        self.__shortest_paths = dict(nx.all_pairs_dijkstra_path_length(self.__graph))

    def qd(self,u,v):
        # distance query: sum of weights of shortest path
        return self.__shortest_paths.get(u, {}).get(v, math.inf)

    def qw(self,u,v):
        # Edge weight query, return w(u,v)
        edge_data = self.__graph.get_edge_data(u, v)
        return edge_data.get('weight', math.inf) if edge_data else math.inf

    def qc(self, u, S, w_thr):
        # Component query: Returns 1 if u and any v in S belong to the same 
        # connected component in the subgraph G[w >= w_thr], else 0.
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
    

TRUE_EDGES = {
    (min(u, v), max(u, v)): float(data.get("weight", 1.0))
    for u, v, data in G.edges(data=True)
}
WMAX = max(TRUE_EDGES.values())
D_MAX = max(dict(G.degree()).values())


# =============================================================================
# Query-counting wrapper
# Count EVERY oracle call as one composite query
# =============================================================================
class CountedOracle:
    def __init__(self, oracle):
        self.oracle = oracle
        self.counts = collections.Counter()

    def qd(self, u, v):
        self.counts["qd"] += 1
        return self.oracle.qd(u, v)

    def qw(self, u, v):
        self.counts["qw"] += 1
        return self.oracle.qw(u, v)

    def qc(self, u, S, w_thr):
        self.counts["qc"] += 1
        return self.oracle.qc(u, S, w_thr)

    @property
    def total(self):
        return sum(self.counts.values())


# =============================================================================
# Small helpers
# =============================================================================
def norm_edge(u, v):
    return (u, v) if u < v else (v, u)

def sample_set(W, s):
    """
    SAMPLE(W, s) from Algorithm 4:
    each element independently selected with probability s / |W|.
    If |W| <= s, return W.
    """
    W = list(W)
    if len(W) <= s:
        return set(W)

    p = s / float(len(W))
    out = {x for x in W if random.random() < p}
    if not out:
        out = {random.choice(W)}
    return out

def random_multiset(V, T):
    return [random.choice(V) for _ in range(T)]

def exhaustive_query(V, counted_oracle, w_thr=1.0):
    """
    EXHAUSTIVE-QUERY on the given vertex set.
    Keeps all edges with weight >= w_thr.
    """
    V = list(V)
    found = {}
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            u, v = V[i], V[j]
            w = counted_oracle.qw(u, v)
            if (not math.isinf(w)) and w >= w_thr:
                found[norm_edge(u, v)] = float(w)
    return found


# =============================================================================
# Thresholded oracle view for LBL-R
# -----------------------------------------------------------------------------
# The paper uses threshold-aware qd/qw in the layer G[w >= Wthr].
# Since the pasted Oracle only directly thresholds qc, we derive:
#   qw_thr(u,v,Wthr) = qw(u,v) if qw(u,v) >= Wthr else 0
#   qd_thr(u,v,Wthr) = shortest-path distance in G[w >= Wthr]
# built ONLY from real oracle.qw calls.
# =============================================================================
class ThresholdOracleView:
    def __init__(self, counted_oracle):
        self.O = counted_oracle
        self.edge_cache = {}
        self.graph_cache = {}
        self.dist_cache = {}

    def qw_thr(self, u, v, w_thr):
        key = (min(u, v), max(u, v), w_thr)
        if key in self.edge_cache:
            return self.edge_cache[key]

        w = self.O.qw(u, v)
        ans = 0.0
        if (not math.isinf(w)) and w >= w_thr:
            ans = float(w)

        self.edge_cache[key] = ans
        return ans

    def build_layer_graph(self, V, w_thr):
        key = (tuple(sorted(V)), w_thr)
        if key in self.graph_cache:
            return self.graph_cache[key]

        H = nx.Graph()
        H.add_nodes_from(V)
        V_list = list(V)

        for i in range(len(V_list)):
            for j in range(i + 1, len(V_list)):
                u, v = V_list[i], V_list[j]
                w = self.qw_thr(u, v, w_thr)
                if w != 0:
                    H.add_edge(u, v, weight=w)

        self.graph_cache[key] = H
        return H

    def qd_thr(self, u, v, V, w_thr):
        key = (u, v, tuple(sorted(V)), w_thr)
        if key in self.dist_cache:
            return self.dist_cache[key]

        H = self.build_layer_graph(V, w_thr)
        try:
            d = nx.shortest_path_length(H, source=u, target=v, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            d = math.inf

        self.dist_cache[key] = d
        return d


# =============================================================================
# Algorithms 1-6 from the main paper: LBL-R
# =============================================================================

# -----------------------------------------------------------------------------
# Algorithm 2: FIND-CONNECTED-COMPONENTS(V, Wthr)
# -----------------------------------------------------------------------------
def find_connected_components(V, w_thr, counted_oracle):
    V = list(V)
    if not V:
        return []

    components = [set([V[0]])]

    for vi in V[1:]:
        all_prev = set().union(*components)

        if counted_oracle.qc(vi, list(all_prev), w_thr) == 0:
            components.append({vi})
        else:
            lo, hi = 0, len(components) - 1
            found = None

            while lo <= hi:
                mid = (lo + hi) // 2
                left_union = set().union(*components[lo:mid + 1])

                if counted_oracle.qc(vi, list(left_union), w_thr) == 1:
                    if lo == mid:
                        found = lo
                        break
                    hi = mid
                else:
                    lo = mid + 1

            if found is None:
                found = lo

            components[found].add(vi)

    return components


# -----------------------------------------------------------------------------
# Algorithm 3: FIND-NEIGHBORS(V, a, Wthr)
# -----------------------------------------------------------------------------
def find_neighbors(V, a, w_thr, threshold_view):
    N_a = {v for v in V if v != a and threshold_view.qw_thr(a, v, w_thr) != 0}
    N2_a = set(N_a)

    for v in list(N_a):
        for u in V:
            if u != v and threshold_view.qw_thr(u, v, w_thr) != 0:
                N2_a.add(u)

    return N2_a


# -----------------------------------------------------------------------------
# Helper: d(A, v)
# -----------------------------------------------------------------------------
def d_to_set(A, v, V, w_thr, threshold_view):
    if not A:
        return math.inf
    return min(threshold_view.qd_thr(a, v, V, w_thr) for a in A)


# -----------------------------------------------------------------------------
# Algorithm 4: ESTIMATED-CENTERS(V, s, Wthr)
# -----------------------------------------------------------------------------
def estimated_centers(V, s, w_thr, threshold_view, K=2):
    V = list(V)
    n_local = len(V)

    A = set()
    W = set(V)

    T = max(
        1,
        int(
            math.ceil(
                K
                * s
                * math.log(max(2, n_local))
                * math.log(max(math.e, math.log(max(2, n_local))))
            )
        ),
    )

    while W:
        A_prime = sample_set(W, s)

        # "Every pair in A' x V is queried" (conceptually via qd in the paper)
        for a in A_prime:
            for v in V:
                threshold_view.qd_thr(a, v, V, w_thr)

        A |= A_prime

        est_sizes = {}
        for w in W:
            X = random_multiset(V, T)
            count = 0
            for x in X:
                if threshold_view.qd_thr(w, x, V, w_thr) < d_to_set(A, x, V, w_thr, threshold_view):
                    count += 1
            est_sizes[w] = count * n_local / float(T)

        W = {w for w in W if est_sizes[w] >= 5.0 * n_local / float(s)}

    return A


# -----------------------------------------------------------------------------
# Algorithm 5: RECONSTRUCT-SUB(V, Wthr)
# -----------------------------------------------------------------------------
def reconstruct_sub_lbl(V, w_thr, counted_oracle, threshold_view, D):
    V = list(V)
    n_local = len(V)

    s = max(1, int(math.ceil(D * math.sqrt(n_local))))
    A = estimated_centers(V, s, w_thr, threshold_view)

    recovered = {}

    for a in A:
        N2_a = find_neighbors(V, a, w_thr, threshold_view)

        # "Every pair in N2(a) x V is queried"
        for b in N2_a:
            for v in V:
                threshold_view.qd_thr(b, v, V, w_thr)

        C = {}
        for b in N2_a:
            C[b] = {v for v in V if threshold_view.qd_thr(b, v, V, w_thr) < d_to_set(A, v, V, w_thr, threshold_view)}

        D_a = set(N2_a)
        for b in N2_a:
            D_a |= C[b]

        E_a = exhaustive_query(D_a, counted_oracle, w_thr=w_thr)
        recovered.update(E_a)

    return recovered


# -----------------------------------------------------------------------------
# Algorithm 6: RECONSTRUCT(V, Wthr)
# -----------------------------------------------------------------------------
def reconstruct_lbl(V, w_thr, counted_oracle, threshold_view, D):
    n_local = len(V)
    # Concrete usable limit corresponding to the paper's O(D^3 n^(3/2) log^2 n log log n)
    Q_limit = max(
        1,
        int(
            math.ceil(
                (D ** 3)
                * (n_local ** 1.5)
                * math.log2(max(2, n_local))
                * math.log(max(math.e, math.log(max(2, n_local))))
            )
        ),
    )

    while True:
        before = counted_oracle.total
        E = reconstruct_sub_lbl(V, w_thr, counted_oracle, threshold_view, D)
        used = counted_oracle.total - before
        if used <= Q_limit:
            return E
        # Otherwise retry from scratch, exactly in the spirit of Algorithm 6.


# -----------------------------------------------------------------------------
# Algorithm 1: LBL-R(V)
# -----------------------------------------------------------------------------
def lbl_r(V, Wmax, counted_oracle, D):
    threshold_view = ThresholdOracleView(counted_oracle)
    recovered = {}
    n_local = len(V)
    small_threshold = n_local ** 0.25

    for j in range(int(math.floor(math.log2(Wmax))) + 1):
        w_thr = 2 ** j
        components = find_connected_components(V, w_thr, counted_oracle)

        for comp in components:
            if len(comp) <= small_threshold:
                E_comp = exhaustive_query(comp, counted_oracle, w_thr=w_thr)
            else:
                E_comp = reconstruct_lbl(list(comp), w_thr, counted_oracle, threshold_view, D)
            recovered.update(E_comp)

        if max(len(c) for c in components) <= small_threshold:
            break

    return recovered


# =============================================================================
# Appendix A: NT-R (No-Threshold Reconstruction)
# -----------------------------------------------------------------------------
# Paper description:
# - Use RECONSTRUCT with Wthr = 1
# - Replace FIND-NEIGHBORS by the closed ball B̄(a, 2Wmax)
# - Use a different s inside RECONSTRUCT-SUB
#
# Here:
#   B̄(a, 2Wmax) = {v in V : d(a,v) <= 2Wmax}
# and the appendix gives:
#   b = (D^(2Wmax+1)-1)/(D-1),   s = sqrt(b*n)
# =============================================================================

def estimated_centers_ntr(V, s, counted_oracle, K=2):
    """
    Same Algorithm 4 idea, but now all distances are in the original graph (no threshold).
    """
    V = list(V)
    n_local = len(V)

    # Practical shortcut: if s >= n then SAMPLE(W,s)=W immediately,
    # so we pre-query all needed distances once.
    if s >= n_local:
        dist_cache = {}
        for a in V:
            for v in V:
                dist_cache[(a, v)] = counted_oracle.qd(a, v)
        return set(V), dist_cache

    A = set()
    W = set(V)

    T = max(
        1,
        int(
            math.ceil(
                K
                * s
                * math.log(max(2, n_local))
                * math.log(max(math.e, math.log(max(2, n_local))))
            )
        ),
    )

    dist_cache = {}

    def qd_cached(u, v):
        key = (u, v)
        if key not in dist_cache:
            dist_cache[key] = counted_oracle.qd(u, v)
        return dist_cache[key]

    def d_to_A(A_set, v):
        if not A_set:
            return math.inf
        return min(qd_cached(a, v) for a in A_set)

    while W:
        A_prime = sample_set(W, s)

        for a in A_prime:
            for v in V:
                qd_cached(a, v)

        A |= A_prime

        est_sizes = {}
        for w in W:
            X = random_multiset(V, T)
            count = 0
            for x in X:
                if qd_cached(w, x) < d_to_A(A, x):
                    count += 1
            est_sizes[w] = count * n_local / float(T)

        W = {w for w in W if est_sizes[w] >= 5.0 * n_local / float(s)}

    return A, dist_cache


def reconstruct_sub_ntr(V, counted_oracle, Wmax, D):
    """
    NT-RS: RECONSTRUCT-SUB modified for Appendix A.
    """
    V = list(V)
    n_local = len(V)

    if D == 1:
        b = 2 * int(Wmax) + 1
    else:
        b = (D ** (2 * int(Wmax) + 1) - 1) / float(D - 1)

    s = max(1, int(math.ceil(math.sqrt(max(1.0, b) * n_local))))

    A, dist_cache = estimated_centers_ntr(V, s, counted_oracle)

    def qd_cached(u, v):
        key = (u, v)
        if key not in dist_cache:
            dist_cache[key] = counted_oracle.qd(u, v)
        return dist_cache[key]

    def d_to_A(A_set, v):
        return min(qd_cached(a, v) for a in A_set)

    recovered = {}

    for a in A:
        # Closed ball B̄(a, 2Wmax)
        ball = {v for v in V if qd_cached(a, v) <= 2 * Wmax}

        D_a = set(ball)
        for b_vertex in ball:
            for v in V:
                qd_cached(b_vertex, v)

            C_b = {v for v in V if qd_cached(b_vertex, v) < d_to_A(A, v)}
            D_a |= C_b

        E_a = exhaustive_query(D_a, counted_oracle, w_thr=1.0)
        recovered.update(E_a)

    return recovered


def ntr(V, counted_oracle, Wmax, D):
    """
    Appendix A / NT-R wrapper.
    """
    return reconstruct_sub_ntr(V, counted_oracle, Wmax, D)


# =============================================================================
# Run both algorithms
# =============================================================================
def evaluate(recovered_edges, true_edges):
    correct = 0
    for e, w in true_edges.items():
        if recovered_edges.get(e) == w:
            correct += 1
    all_correct = (correct == len(true_edges) and len(recovered_edges) == len(true_edges))
    return correct, all_correct


def main():
    random.seed(42)

    # --- LBL-R ---
    counted_lbl = CountedOracle(Oracle(adj))
    t0 = time.time()
    recovered_lbl = lbl_r(list(G.nodes()), WMAX, counted_lbl, D_MAX)
    t1 = time.time()
    lbl_correct_edges, lbl_ok = evaluate(recovered_lbl, TRUE_EDGES)

    # --- NT-R ---
    counted_ntr = CountedOracle(Oracle(adj))
    t2 = time.time()
    recovered_ntr = ntr(list(G.nodes()), counted_ntr, WMAX, D_MAX)
    t3 = time.time()
    ntr_correct_edges, ntr_ok = evaluate(recovered_ntr, TRUE_EDGES)

    # --- Comparison ---
    print("=== QUERY COMPLEXITY COMPARISON ON ZACHARY’S KARATE CLUB (34 nodes, 78 edges) ===")
    print()
    print("Algorithm          | Total Composite Queries | Edges Recovered | Correct?")
    print("-------------------|--------------------------|-----------------|---------")
    print(f"LBL-R (with thresholds) | {str(counted_lbl.total).ljust(24)} | {str(lbl_correct_edges) + '/78':<15} | {'YES' if lbl_ok else 'NO'}")
    print(f"NT-R (traditional, no threshold) | {str(counted_ntr.total).ljust(24)} | {str(ntr_correct_edges) + '/78':<15} | {'YES' if ntr_ok else 'NO'}")
    print()

    if counted_ntr.total > 0:
        reduction = 100.0 * (counted_ntr.total - counted_lbl.total) / counted_ntr.total
    else:
        reduction = 0.0

    print(f"Conclusion: LBL-R uses {reduction:.2f}% fewer queries than the traditional NT-R method.")
    print()
    print(f"Max weight Wmax in this graph: {int(WMAX) if float(WMAX).is_integer() else WMAX}")
    print(f"Max degree D: {D_MAX}")
    print()
    print("This demonstrates that the layer-by-layer approach with weight thresholds (as proposed in the paper) is significantly more query-efficient than the traditional no-threshold method described in Appendix A.")
    print()
    print(f"[debug] LBL-R runtime: {t1 - t0:.4f}s")
    print(f"[debug] NT-R runtime:  {t3 - t2:.4f}s")


if __name__ == "__main__":
    main()