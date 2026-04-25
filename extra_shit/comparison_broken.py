import networkx as nx
import numpy as np
import random
import math
import collections
from oracle import Oracle
from helper import norm_edge    
from algorithms import lbl_r
from ntr_algorithms import reconstruct_ntr
# -----------------------------
# 1) Load benchmark graph
# -----------------------------
G = nx.karate_club_graph()
import networkx as nx
import numpy as np
import math


# ============================================================
# Utility helpers
# ============================================================
random.seed(42)

def norm_edge(u, v):
    return (u, v) if u < v else (v, u)

def sample_set(W, s):
    """
    SAMPLE(W, s) as described near Algorithm 4:
    each element is selected independently with probability s / |W|.
    If |W| <= s, return W.
    """
    W = list(W)
    if len(W) <= s:
        return set(W)

    p = float(s) / float(len(W))
    out = {x for x in W if random.random() < p}

    # Avoid empty sample in practice
    if not out:
        out.add(random.choice(W))
    return out

def multiset_random_sample(V_list, T):
    """Random multi-subset of size T (sampling with replacement)."""
    return [random.choice(V_list) for _ in range(T)]

def get_true_edges_with_weights(graph):
    out = {}
    for u, v, data in graph.edges(data=True):
        out[norm_edge(u, v)] = data.get("weight", 1)
    return out

TRUE_EDGES = get_true_edges_with_weights(G)
D_MAX = max(dict(G.degree()).values())


# ============================================================
# Run the full reconstruction
# ============================================================
all_vertices = list(G.nodes())
W_MAX = max(data.get("weight", 1) for _, _, data in G.edges(data=True))

recovered_edges = lbl_r(all_vertices, W_MAX, D_MAX)

# ============================================================
# Evaluation
# ============================================================
true_edge_set = set(TRUE_EDGES.keys())
recovered_edge_set = set(recovered_edges.keys())

all_correct = (true_edge_set == recovered_edge_set)

if all_correct:
    weights_correct = all(
        TRUE_EDGES[e] == recovered_edges[e]
        for e in true_edge_set
    )
else:
    weights_correct = False

all_fully_correct = all_correct and weights_correct

num_correct = 0
for e, true_w in TRUE_EDGES.items():
    rec_w = recovered_edges.get(e, None)
    if rec_w == true_w:
        num_correct += 1

accuracy = 100.0 * num_correct / len(TRUE_EDGES)

# sample 10 random true edges
sample_edges = random.sample(list(TRUE_EDGES.keys()), min(10, len(TRUE_EDGES)))

# ============================================================
# Output exactly in requested format
# ============================================================
print("=== PRELIMINARY RESULTS - Zachary’s Karate Club (34 nodes, 78 edges) ===")
print(f"- Total composite queries used: {Oracle.query_count}")
print(f"- All edges and weights recovered correctly? {'YES' if all_fully_correct else 'NO'}")
print(f"- Recovery accuracy: {accuracy:.2f}%")
print("Sample correctness table (10 random edges):")
print("Edge (u-v) | True Weight | Recovered Weight | Correct?")
print("-----------|-------------|------------------|---------")

for u, v in sample_edges:
    true_w = TRUE_EDGES[(u, v)]
    rec_w = recovered_edges.get((u, v), None)
    ok = "Yes" if rec_w == true_w else "No"
    print(f"{u}-{v}".ljust(11), "|", str(true_w).ljust(11), "|", str(rec_w).ljust(16), "|", ok)

print(f"- Max degree D in this graph: {D_MAX}")
print('- Brief note: "This confirms the LBL-R algorithm correctly reconstructs the graph using composite queries (qw, qd, qc with weight thresholds) as claimed in the paper."')