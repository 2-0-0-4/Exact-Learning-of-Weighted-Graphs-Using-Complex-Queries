import os
import gzip
import tarfile
import networkx as nx
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
TAR_PATH      = "big_dataset/facebook.tar.gz"
COMBINED_PATH = "big_dataset/facebook_combined.txt.gz"
# ───────────────────────────────────────────────────────────────────────────


# ── helper: compute stats for a graph ───────────────────────────────────────
def analyze_graph(G, name="Graph"):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees) if degrees else 0
    max_degree = np.max(degrees) if degrees else 0
    min_degree = np.min(degrees) if degrees else 0

    density = nx.density(G)
    num_components = nx.number_connected_components(G)

    print(f"\n── {name} ──")
    print(f"Nodes: {num_nodes}")
    print(f"Edges: {num_edges}")
    print(f"Avg degree: {avg_degree:.2f}")
    print(f"Max degree: {max_degree}")
    print(f"Min degree: {min_degree}")
    print(f"Density (sparsity): {density:.6f}")
    print(f"Connected components: {num_components}")

    return {
        "nodes": num_nodes,
        "edges": num_edges,
        "avg_degree": avg_degree,
        "density": density,
        "components": num_components
    }


# ── 1. analyze combined graph ──────────────────────────────────────────────
def analyze_combined(path):
    G = nx.Graph()

    with gzip.open(path, 'rt') as f:
        for line in f:
            u, v = map(int, line.strip().split())
            G.add_edge(u, v)

    stats = analyze_graph(G, "Combined Graph")
    return G, stats


# ── 2. analyze ego graphs inside tar ───────────────────────────────────────
def analyze_ego_tar(tar_path):
    graphs = []

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".edges"):
                f = tar.extractfile(member)
                if f is None:
                    continue

                G = nx.Graph()
                for line in f:
                    u, v = map(int, line.decode().strip().split())
                    G.add_edge(u, v)

                graphs.append((member.name, G))

    print(f"\nTotal ego graphs: {len(graphs)}")

    all_stats = []
    for name, G in graphs:
        stats = analyze_graph(G, name)
        all_stats.append(stats)

    return graphs, all_stats


# ── run everything ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # combined graph
    combined_graph, combined_stats = analyze_combined(COMBINED_PATH)

    # ego graphs
    ego_graphs, ego_stats = analyze_ego_tar(TAR_PATH)

    # ── summary ────────────────────────────────────────────────────────────
    print("\n=== SUMMARY ===")
    print(f"Total ego graphs: {len(ego_graphs)}")

    avg_nodes = np.mean([s["nodes"] for s in ego_stats])
    avg_edges = np.mean([s["edges"] for s in ego_stats])
    avg_density = np.mean([s["density"] for s in ego_stats])
    avg_components = np.mean([s["components"] for s in ego_stats])

    print(f"Avg nodes per ego graph: {avg_nodes:.2f}")
    print(f"Avg edges per ego graph: {avg_edges:.2f}")
    print(f"Avg density: {avg_density:.6f}")
    print(f"Avg connected components: {avg_components:.2f}")