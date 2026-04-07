import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
from oracle import Oracle
from helper import norm_edge
from lblr_algorithms import lbl_r, exhaustive_query
from ntr_algorithms import reconstruct_ntr

# ── Benchmark Configuration ────────────────────────────────────────────────

def add_pareto_weights(G, alpha=2.0):
    for u, v in list(G.edges()):
        weight = np.random.pareto(alpha) + 1.0   # min weight = 1
        G[u][v]['weight'] = weight
    return G

def run_comparison(n_nodes):
    """Run algorithm comparison on random regular graph with n_nodes vertices."""
    print(f"Running benchmark for n={n_nodes}...", end=" ", flush=True)
    
    # Generate graph
    G = nx.random_regular_graph(d=3, n=n_nodes, seed=42)
    G = add_pareto_weights(G, alpha=2.0)
    
    # Get adjacency matrix and parameters
    adj = nx.to_numpy_array(G, weight='weight')
    all_nodes = list(G.nodes())
    D_MAX = max(dict(G.degree()).values())
    W_MAX = int(max(d.get('weight', 1) for _, _, d in G.edges(data=True)))
    true_edges = {norm_edge(u, v): d.get('weight', 1) for u, v, d in G.edges(data=True)}
    
    results = {}
    
    # 1. LBL-R
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
    
    print("✓")
    return results

# ── Run benchmarks for multiple graph sizes ────────────────────────────────

print("=" * 80)
print("BENCHMARKING: Graph Size vs Query Complexity and Execution Time")
print("=" * 80)

all_results = {}
graph_sizes = [20, 50, 80, 100]  # All even, so n*d is even for d=3

for n in graph_sizes:
    all_results[n] = run_comparison(n)

# ── Prepare data for plotting ──────────────────────────────────────────────

sizes = list(all_results.keys())
lbl_queries = [all_results[n]['LBL-R']['queries'] for n in sizes]
ntr_queries = [all_results[n]['NT-R']['queries'] for n in sizes]
exh_queries = [all_results[n]['EXHAUSTIVE-QUERY']['queries'] for n in sizes]

lbl_times = [all_results[n]['LBL-R']['time'] for n in sizes]
ntr_times = [all_results[n]['NT-R']['time'] for n in sizes]
exh_times = [all_results[n]['EXHAUSTIVE-QUERY']['time'] for n in sizes]

# ── Plot 1: Queries vs Graph Size ──────────────────────────────────────────

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(sizes, lbl_queries, marker='o', label='LBL-R', linewidth=2, markersize=8)
plt.plot(sizes, ntr_queries, marker='s', label='NT-R', linewidth=2, markersize=8)
plt.plot(sizes, exh_queries, marker='^', label='EXHAUSTIVE-QUERY', linewidth=2, markersize=8)
plt.xlabel('Graph Size (nodes)', fontsize=12)
plt.ylabel('Number of Queries', fontsize=12)
plt.title('Query Complexity vs Graph Size', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(sizes)

# ── Plot 2: Time vs Graph Size ────────────────────────────────────────────

plt.subplot(1, 2, 2)
plt.plot(sizes, lbl_times, marker='o', label='LBL-R', linewidth=2, markersize=8)
plt.plot(sizes, ntr_times, marker='s', label='NT-R', linewidth=2, markersize=8)
plt.plot(sizes, exh_times, marker='^', label='EXHAUSTIVE-QUERY', linewidth=2, markersize=8)
plt.xlabel('Graph Size (nodes)', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
plt.title('Execution Time vs Graph Size', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(sizes)

plt.tight_layout()
plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Plots saved to algorithm_comparison.png")

# ── Display Summary Table ──────────────────────────────────────────────────

print("\n" + "=" * 150)
print("SUMMARY: QUERY COMPLEXITY AND EXECUTION TIME ACROSS GRAPH SIZES")
print("=" * 150)
print(f"{'Size':<6} | {'LBL-R Queries':<15} | {'NT-R Queries':<15} | {'Exhaust Queries':<15} | {'LBL-R Time':<12} | {'NT-R Time':<12} | {'Exhaust Time':<12}")
print("-" * 150)

for n in sizes:
    results = all_results[n]
    lbl_q = results['LBL-R']['queries']
    ntr_q = results['NT-R']['queries']
    exh_q = results['EXHAUSTIVE-QUERY']['queries']
    lbl_t = results['LBL-R']['time']
    ntr_t = results['NT-R']['time']
    exh_t = results['EXHAUSTIVE-QUERY']['time']
    
    print(f"{n:<6} | {lbl_q:<15} | {ntr_q:<15} | {exh_q:<15} | {lbl_t:<12.4f} | {ntr_t:<12.4f} | {exh_t:<12.4f}")

print("=" * 150)

plt.show()
