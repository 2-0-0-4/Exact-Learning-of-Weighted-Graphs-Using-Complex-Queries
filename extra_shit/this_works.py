import networkx as nx
import numpy as np
import time
import random

from oracle import Oracle
from helper import norm_edge
#from lblr_algorithms import lbl_r, exhaustive_query
from lblr_algorithms import lbl_r, exhaustive_query

from ntr_algorithms import reconstruct_ntr


# ── Simple random tree generator ───────────────────────────────────────────
def simple_random_tree(n, seed=None):
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(1, n):
        parent = rng.randint(0, i - 1)
        G.add_edge(i, parent)

    return G


# ── LBL-R friendly disconnected graph generator ────────────────────────────
def generate_lblr_friendly_graph(
    num_components=100,
    component_size=10,
    seed=42,
    weight_low=1,
    weight_high=4,
    extra_edges_per_component=2,
    max_degree=4
):
    random.seed(seed)
    np.random.seed(seed)

    components = []

    for i in range(num_components):
        # Start with a sparse connected tree
        Gc = simple_random_tree(component_size, seed=seed + i)

        # Add a few extra edges while keeping degree bounded
        possible_edges = list(nx.non_edges(Gc))
        random.shuffle(possible_edges)

        added = 0
        for u, v in possible_edges:
            if added >= extra_edges_per_component:
                break
            if Gc.degree(u) < max_degree and Gc.degree(v) < max_degree:
                Gc.add_edge(u, v)
                added += 1

        # Assign random integer weights
        for u, v in Gc.edges():
            Gc[u][v]["weight"] = random.randint(weight_low, weight_high)

        components.append(Gc)

    # Make one disconnected graph
    G = nx.disjoint_union_all(components)
    return G


# ── Generate benchmark graph ───────────────────────────────────────────────
G = generate_lblr_friendly_graph(
    num_components=1000,
    component_size=10,
    seed=42,
    weight_low=1,
    weight_high=30,
    extra_edges_per_component=2,
    max_degree=4
)

adj = nx.to_numpy_array(G, weight="weight")

all_nodes = list(G.nodes())
D_MAX = max(dict(G.degree()).values())
W_MAX = int(max(d.get("weight", 1) for _, _, d in G.edges(data=True)))

true_edges = {
    norm_edge(u, v): d.get("weight", 1)
    for u, v, d in G.edges(data=True)
}

print("=" * 100)
print("ALGORITHM COMPARISON ON 1000-NODE DISCONNECTED GRAPH")
print("=" * 100)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Connected components: {nx.number_connected_components(G)}")
print(f"Parameters: D_MAX={D_MAX}, W_MAX={W_MAX}")
print()

component_sizes = sorted([len(c) for c in nx.connected_components(G)])
print(f"Component sizes: {component_sizes[:10]} ...")
print()

results = {}

# ── 1. LBL-R ───────────────────────────────────────────────────────────────
print("Running LBL-R...")
oracle_lblr = Oracle(adj)
t0 = time.time()
recovered_lblr = lbl_r(oracle_lblr, all_nodes, W_MAX, D_MAX)
t1 = time.time()

correct_lblr = sum(
    1 for e, w in true_edges.items()
    if recovered_lblr.get(e) == w
)

results["LBL-R"] = {
    "queries": oracle_lblr.query_count,
    "edges": len(recovered_lblr),
    "correct": correct_lblr,
    "accuracy": (correct_lblr / len(true_edges)) * 100 if len(true_edges) > 0 else 0,
    "time": t1 - t0
}

# ── 3. EXHAUSTIVE-QUERY ────────────────────────────────────────────────────
print("Running EXHAUSTIVE-QUERY...")
oracle_exhaustive = Oracle(adj)
t0 = time.time()
recovered_exhaustive = exhaustive_query(oracle_exhaustive, all_nodes, w_thr=1)
t1 = time.time()

correct_exhaustive = sum(
    1 for e, w in true_edges.items()
    if recovered_exhaustive.get(e) == w
)

results["EXHAUSTIVE-QUERY"] = {
    "queries": oracle_exhaustive.query_count,
    "edges": len(recovered_exhaustive),
    "correct": correct_exhaustive,
    "accuracy": (correct_exhaustive / len(true_edges)) * 100 if len(true_edges) > 0 else 0,
    "time": t1 - t0
}

# ── Display results ────────────────────────────────────────────────────────
print()
print("=" * 120)
print("RESULTS COMPARISON")
print("=" * 120)
print(f"{'Algorithm':<20} | {'Queries':<12} | {'Recovered Edges':<16} | {'Correct':<10} | {'Accuracy':<12} | {'Time (s)':<10}")
print("-" * 120)

for algo, data in results.items():
    print(
        f"{algo:<20} | "
        f"{data['queries']:<12} | "
        f"{data['edges']:<16} | "
        f"{data['correct']:<10} | "
        f"{data['accuracy']:>9.2f}%   | "
        f"{data['time']:>8.4f}"
    )

print()
