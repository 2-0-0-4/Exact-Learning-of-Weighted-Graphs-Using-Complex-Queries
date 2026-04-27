"""
EXHAUSTIVE-QUERY Test on Disconnected Weighted Graphs

Tests the brute-force O(n²) algorithm on proper disconnected graphs
with Pareto-distributed weights.

Paper: "Exact Learning of Weighted Graphs Using Composite Queries"
"""

import networkx as nx
import numpy as np
from oracle import Oracle
from helper import norm_edge
from lblr_algorithms import exhaustive_query
from disc_graph_gen import generate_small_test_graph, print_graph_info

# 1. Generate a benchmark graph (Small disconnected with Pareto weights)
G, metadata = generate_small_test_graph(seed=42)
print_graph_info(G, metadata)

# Convert to adjacency matrix
adj = nx.to_numpy_array(G, weight='weight')
oracle = Oracle(adj)

# 2. Parameters
all_nodes = list(G.nodes())
D_MAX = max(dict(G.degree()).values())
W_MAX = int(max(d.get('weight', 1) for _, _, d in G.edges(data=True)))

print(f"\nRunning EXHAUSTIVE-QUERY Reconstruction...")
print(f"Nodes: {len(all_nodes)}, Max Weight: {W_MAX}, Max Degree: {D_MAX}")
print(f"Expected Queries: {len(all_nodes) * (len(all_nodes) - 1) // 2} (Θ(n²))")

# 3. Run EXHAUSTIVE-QUERY
recovered_edges = exhaustive_query(oracle, all_nodes, w_thr=1)

# 4. Verification
true_edges = {norm_edge(u, v): d.get('weight', 1) for u, v, d in G.edges(data=True)}
correct = 0
for edge, weight in true_edges.items():
    if recovered_edges.get(edge) == weight:
        correct += 1

accuracy = (correct / len(true_edges)) * 100

print(f"\n{'='*60}")
print(f"EXHAUSTIVE-QUERY Results")
print(f"{'='*60}")
print(f"Total Queries: {oracle.query_count}")
print(f"Edges Recovered: {len(recovered_edges)} / {len(true_edges)}")
print(f"Edges Correct: {correct} / {len(true_edges)}")
print(f"Reconstruction Accuracy: {accuracy:.2f}%")
print(f"{'='*60}")
