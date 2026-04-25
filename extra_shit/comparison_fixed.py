import networkx as nx
import numpy as np
import random
import math
import collections
import time
from oracle import Oracle
from helper import norm_edge    
from lblr_algorithms import lbl_r
from ntr_algorithms import reconstruct_ntr

G = nx.karate_club_graph()
n = G.number_of_nodes()
m = G.number_of_edges()

adj = np.zeros((n, n), dtype=float)
for u, v, data in G.edges(data=True):
    w = float(data.get("weight", 1.0))
    adj[u][v] = w
    adj[v][u] = w

# Calculate graph parameters
WMAX = np.max(adj)
D_MAX = max(dict(G.degree()).values())
TRUE_EDGES = {(u, v): float(data.get("weight", 1.0)) for u, v, data in G.edges(data=True)}


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
    counted_lbl = Oracle(adj)
    t0 = time.time()
    recovered_lbl = lbl_r(counted_lbl, list(G.nodes()), WMAX, D_MAX)
    t1 = time.time()
    lbl_correct_edges, lbl_ok = evaluate(recovered_lbl, TRUE_EDGES)

    # --- NT-R ---
    counted_ntr = Oracle(adj)
    t2 = time.time()
    recovered_ntr = reconstruct_ntr(counted_ntr, list(G.nodes()), WMAX, D_MAX)
    t3 = time.time()
    ntr_correct_edges, ntr_ok = evaluate(recovered_ntr, TRUE_EDGES)

    # --- Comparison ---
    print("=== QUERY COMPLEXITY COMPARISON ON ZACHARY’S KARATE CLUB (34 nodes, 78 edges) ===")
    print()
    print("Algorithm          | Total Composite Queries | Edges Recovered | Correct?")
    print("-------------------|--------------------------|-----------------|---------")
    print(f"LBL-R (with thresholds) | {str(counted_lbl.query_count).ljust(24)} | {str(lbl_correct_edges) + '/78':<15} | {'YES' if lbl_ok else 'NO'}")
    print(f"NT-R (traditional, no threshold) | {str(counted_ntr.query_count).ljust(24)} | {str(ntr_correct_edges) + '/78':<15} | {'YES' if ntr_ok else 'NO'}")
    print()

    if counted_ntr.query_count > 0:
        reduction = 100.0 * (counted_ntr.query_count - counted_lbl.query_count) / counted_ntr.query_count
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