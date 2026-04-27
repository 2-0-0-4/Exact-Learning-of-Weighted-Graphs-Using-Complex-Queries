import networkx as nx
import numpy as np
from oracle import Oracle
from helper import norm_edge
from lblr_algorithms import lbl_r
from disc_graph_gen import generate_small_test_graph, print_graph_info

# 1. Generate a benchmark graph (Small disconnected with Pareto weights)
G, metadata = generate_small_test_graph(seed=42)
print_graph_info(G, metadata)

adj = nx.to_numpy_array(G, weight='weight')
oracle = Oracle(adj)

D_MAX = max(dict(G.degree()).values())
W_MAX = max(d.get('weight', 1) for _, _, d in G.edges(data=True))

print("Running LBL-R Reconstruction...")
recovered_edges = lbl_r(oracle, list(G.nodes()), W_MAX, D_MAX)

true_edges = {norm_edge(u, v): d.get('weight', 1) for u, v, d in G.edges(data=True)}
correct_count = sum(1 for e, w in true_edges.items() if recovered_edges.get(e) == w)

print(f"=== Results ===")
print(f"Total Queries: {oracle.query_count}")
print(f"Accuracy: {(correct_count/len(true_edges))*100:.2f}%")