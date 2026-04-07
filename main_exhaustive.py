import networkx as nx
import numpy as np
from oracle import Oracle
from helper import norm_edge
from lblr_algorithms import exhaustive_query

# 1. Generate a benchmark graph (Karate Club)
G = nx.karate_club_graph()
# Assign some random weights to make it a 'Weighted' graph reconstruction
for u, v in G.edges():
    G[u][v]['weight'] = np.random.randint(1, 5)

adj = nx.to_numpy_array(G, weight='weight')
oracle = Oracle(adj)

# 2. Parameters
all_nodes = list(G.nodes())
W_MAX = int(max(d['weight'] for _, _, d in G.edges(data=True)))
D_MAX = max(dict(G.degree()).values())

print(f"Starting EXHAUSTIVE-QUERY Reconstruction...")
print(f"Nodes: {len(all_nodes)}, Max Weight: {W_MAX}, Max Degree: {D_MAX}")
print(f"Expected Queries: {len(all_nodes) * (len(all_nodes) - 1) // 2} (Θ(n²))")

# 3. Run EXHAUSTIVE-QUERY: query every pair of vertices with threshold 1
# This is a brute-force O(n²) algorithm that queries all pairs
recovered_edges = exhaustive_query(oracle, all_nodes, w_thr=1)

# 4. Verification
true_edges = {norm_edge(u, v): d['weight'] for u, v, d in G.edges(data=True)}
correct = 0
for edge, weight in true_edges.items():
    if recovered_edges.get(edge) == weight:
        correct += 1

accuracy = (correct / len(true_edges)) * 100
print(f"\n=== EXHAUSTIVE-QUERY Results ===")
print(f"Total Queries: {oracle.query_count}")
print(f"Reconstruction Accuracy: {accuracy:.2f}%")
print(f"Edges Recovered: {len(recovered_edges)} / {len(true_edges)}")
