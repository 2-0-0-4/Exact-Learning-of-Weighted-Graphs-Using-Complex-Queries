import networkx as nx
import numpy as np
import time
from oracle import Oracle
from helper import norm_edge
from lblr_algorithms import lbl_r, exhaustive_query
from ntr_algorithms import reconstruct_ntr

# ── Generate Benchmark Graph ───────────────────────────────────────────────

G = nx.karate_club_graph()
for u, v in G.edges():
    G[u][v]['weight'] = np.random.randint(1, 5)

adj = nx.to_numpy_array(G, weight='weight')

# Get parameters
all_nodes = list(G.nodes())
D_MAX = max(dict(G.degree()).values())
W_MAX = int(max(d.get('weight', 1) for _, _, d in G.edges(data=True)))

# True edges for verification
true_edges = {norm_edge(u, v): d.get('weight', 1) for u, v, d in G.edges(data=True)}

print("=" * 80)
print("ALGORITHM COMPARISON ON ZACHARY'S KARATE CLUB")
print("=" * 80)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Parameters: D_MAX={D_MAX}, W_MAX={W_MAX}")
print()

# ── Run all three algorithms ───────────────────────────────────────────────

results = {}

# 1. LBL-R
print("Running LBL-R...")
oracle_lblr = Oracle(adj)
t0 = time.time()
recovered_lblr = lbl_r(oracle_lblr, all_nodes, W_MAX, D_MAX)
t1 = time.time()
correct_lblr = sum(1 for e, w in true_edges.items() if recovered_lblr.get(e) == w)
results['LBL-R'] = {
    'queries': oracle_lblr.query_count,
    'edges': len(recovered_lblr),
    'correct': correct_lblr,
    'accuracy': (correct_lblr / len(true_edges)) * 100,
    'time': t1 - t0
}

# 2. NT-R
print("Running NT-R...")
oracle_ntr = Oracle(adj)
t0 = time.time()
recovered_ntr = reconstruct_ntr(oracle_ntr, all_nodes, W_MAX, D_MAX)
t1 = time.time()
correct_ntr = sum(1 for e, w in true_edges.items() if recovered_ntr.get(e) == w)
results['NT-R'] = {
    'queries': oracle_ntr.query_count,
    'edges': len(recovered_ntr),
    'correct': correct_ntr,
    'accuracy': (correct_ntr / len(true_edges)) * 100,
    'time': t1 - t0
}

# 3. EXHAUSTIVE-QUERY
print("Running EXHAUSTIVE-QUERY...")
oracle_exhaustive = Oracle(adj)
t0 = time.time()
recovered_exhaustive = exhaustive_query(oracle_exhaustive, all_nodes, w_thr=1)
t1 = time.time()
correct_exhaustive = sum(1 for e, w in true_edges.items() if recovered_exhaustive.get(e) == w)
results['EXHAUSTIVE-QUERY'] = {
    'queries': oracle_exhaustive.query_count,
    'edges': len(recovered_exhaustive),
    'correct': correct_exhaustive,
    'accuracy': (correct_exhaustive / len(true_edges)) * 100,
    'time': t1 - t0
}

# ── Display comparison ─────────────────────────────────────────────────────

print()
print("=" * 100)
print("RESULTS COMPARISON")
print("=" * 100)
print(f"{'Algorithm':<20} | {'Queries':<10} | {'Edges':<10} | {'Accuracy':<10} | {'Correct':<10} | {'Time (s)':<10}")
print("-" * 100)

for algo, data in results.items():
    print(f"{algo:<20} | {data['queries']:<10} | {data['edges']:<10} | {data['accuracy']:>8.2f}% | {data['correct']:<10} | {data['time']:>8.4f}")

print()
