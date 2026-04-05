import networkx as nx
import numpy as np
from oracle import Oracle
from helper import norm_edge
from lblr_algorithms import lbl_r

G = nx.karate_club_graph()
adj = nx.to_numpy_array(G, weight='weight')
for u, v, d in G.edges(data=True):
    if 'weight' not in d: d['weight'] = 1

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