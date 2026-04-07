"""
Facebook Ego Network Visualizer
================================
Supports two modes:
  1. Individual ego networks  — from facebook_tar.gz (10 ego nodes)
  2. Full combined network    — from facebook_combined_txt.gz (all edges)

Requirements:
    pip install networkx matplotlib numpy
"""

import tarfile
import gzip
import io
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
TAR_PATH      = "big_dataset/facebook.tar.gz"          # contains per-ego files
COMBINED_PATH = "big_dataset/facebook_combined.txt.gz" # all edges in one file
# ───────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def load_ego_ids(tar_path: str) -> list[str]:
    """Return sorted list of ego node IDs found in the tar archive."""
    ids = set()
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            base = os.path.basename(m.name)
            if base.endswith(".edges"):
                ids.add(base.replace(".edges", ""))
    return sorted(ids, key=lambda x: int(x))


def load_ego_graph(tar_path: str, ego_id: str) -> tuple[nx.Graph, dict]:
    """
    Build an undirected graph for one ego node.

    Returns
    -------
    G        : NetworkX Graph (ego node included)
    circles  : dict  {circle_name: [node_ids]}
    """
    G = nx.Graph()
    circles: dict[str, list[int]] = {}

    with tarfile.open(tar_path, "r:gz") as tf:
        # ── edges ──────────────────────────────────────────────────────
        edges_name = f"facebook/{ego_id}.edges"
        try:
            ef = tf.extractfile(edges_name)
            neighbors = set()
            for line in ef.read().decode().splitlines():
                parts = line.split()
                if len(parts) == 2:
                    u, v = int(parts[0]), int(parts[1])
                    G.add_edge(u, v)
                    neighbors.update([u, v])
            # ego node connects to every neighbor
            ego = int(ego_id)
            G.add_node(ego)
            for n in neighbors:
                G.add_edge(ego, n)
        except KeyError:
            pass

        # ── circles ────────────────────────────────────────────────────
        circles_name = f"facebook/{ego_id}.circles"
        try:
            cf = tf.extractfile(circles_name)
            for line in cf.read().decode().splitlines():
                parts = line.split()
                if parts:
                    circles[parts[0]] = [int(x) for x in parts[1:]]
        except KeyError:
            pass

    return G, circles


def load_combined_graph(combined_path: str) -> nx.Graph:
    """Build a graph from the combined edge list (all ego networks merged)."""
    G = nx.Graph()
    with gzip.open(combined_path, "rt") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    return G


# ══════════════════════════════════════════════════════════════════════════
#  PLOT: single ego network
# ══════════════════════════════════════════════════════════════════════════

def plot_ego_network(
    tar_path: str,
    ego_id: str,
    layout: str = "spring",
    show_circles: bool = True,
    node_size: int = 40,
    figsize: tuple = (14, 10),
) -> None:
    """
    Visualise one ego network.

    Parameters
    ----------
    layout       : 'spring' | 'kamada_kawai' | 'spectral'
    show_circles : colour nodes by social circle membership
    """
    G, circles = load_ego_graph(tar_path, ego_id)
    ego = int(ego_id)

    print(f"Ego {ego_id}: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges, {len(circles)} circles")

    # ── layout ─────────────────────────────────────────────────────────
    seed = 42
    if layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed, k=0.3)

    # ── colours ────────────────────────────────────────────────────────
    palette = plt.cm.tab20.colors
    node_color = {}

    if show_circles and circles:
        for i, (_, members) in enumerate(circles.items()):
            color = palette[i % len(palette)]
            for m in members:
                if m in G:
                    node_color[m] = color

    colors  = [node_color.get(n, "#cccccc") for n in G.nodes() if n != ego]
    sizes   = [node_size] * len([n for n in G.nodes() if n != ego])
    nodes   = [n for n in G.nodes() if n != ego]

    # ── draw ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#0f0f1a")
    fig.patch.set_facecolor("#0f0f1a")

    # edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15,
                           edge_color="white", width=0.4)

    # regular nodes
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax,
                           node_color=colors, node_size=sizes, alpha=0.85)

    # ego node (highlighted)
    if ego in pos:
        nx.draw_networkx_nodes(G, pos, nodelist=[ego], ax=ax,
                               node_color="#ff4444", node_size=300,
                               edgecolors="white", linewidths=1.5)
        nx.draw_networkx_labels(G, pos, labels={ego: str(ego)}, ax=ax,
                                font_color="white", font_size=7,
                                font_weight="bold")

    # ── legend for circles ─────────────────────────────────────────────
    if show_circles and circles:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=palette[i % len(palette)],
                       markersize=8, label=name)
            for i, name in enumerate(circles)
        ]
        ax.legend(handles=handles, title="Circles", title_fontsize=9,
                  fontsize=7, loc="upper left",
                  facecolor="#1e1e30", labelcolor="white",
                  edgecolor="#555555")

    ax.set_title(f"Facebook Ego Network — Node {ego_id}\n"
                 f"{G.number_of_nodes()} nodes · {G.number_of_edges()} edges · "
                 f"{len(circles)} circles",
                 color="white", fontsize=13, pad=12)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
#  PLOT: all ego networks in a grid
# ══════════════════════════════════════════════════════════════════════════

def plot_all_ego_networks(
    tar_path: str,
    layout: str = "spring",
    node_size: int = 10,
    cols: int = 5,
) -> None:
    """Draw all ego networks side-by-side in a grid."""
    ego_ids = load_ego_ids(tar_path)
    rows = (len(ego_ids) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 4, rows * 3.5))
    fig.patch.set_facecolor("#0f0f1a")
    axes = np.array(axes).flatten()

    for idx, ego_id in enumerate(ego_ids):
        ax = axes[idx]
        ax.set_facecolor("#0f0f1a")

        G, circles = load_ego_graph(tar_path, ego_id)
        ego = int(ego_id)

        pos = nx.spring_layout(G, seed=42, k=0.4)

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.12,
                               edge_color="white", width=0.3)

        regular = [n for n in G.nodes() if n != ego]
        nx.draw_networkx_nodes(G, pos, nodelist=regular, ax=ax,
                               node_color="#4a9eff", node_size=node_size,
                               alpha=0.7)
        if ego in pos:
            nx.draw_networkx_nodes(G, pos, nodelist=[ego], ax=ax,
                                   node_color="#ff4444", node_size=80,
                                   edgecolors="white", linewidths=1)

        ax.set_title(f"Ego {ego_id}\n{G.number_of_nodes()}n · "
                     f"{G.number_of_edges()}e · {len(circles)}c",
                     color="white", fontsize=8)
        ax.axis("off")

    # hide unused subplots
    for ax in axes[len(ego_ids):]:
        ax.set_visible(False)

    fig.suptitle("Facebook — All Ego Networks", color="white",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
#  PLOT: full combined network (sampled for performance)
# ══════════════════════════════════════════════════════════════════════════

def plot_combined_network(
    combined_path: str,
    max_nodes: int = 2000,
    layout: str = "spring",
    figsize: tuple = (16, 12),
) -> None:
    """
    Visualise the merged network.  Samples up to `max_nodes` nodes if the
    graph is large (full graph has ~4k nodes / 88k edges).
    """
    print("Loading combined graph…")
    G_full = load_combined_graph(combined_path)
    print(f"Full graph: {G_full.number_of_nodes()} nodes, "
          f"{G_full.number_of_edges()} edges")

    # sample a subgraph if needed
    if G_full.number_of_nodes() > max_nodes:
        print(f"Sampling {max_nodes} highest-degree nodes…")
        top = sorted(G_full.degree(), key=lambda x: x[1], reverse=True)
        keep = {n for n, _ in top[:max_nodes]}
        G = G_full.subgraph(keep).copy()
    else:
        G = G_full

    print(f"Plotting: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # degree-based colour
    degrees = np.array([G.degree(n) for n in G.nodes()])
    norm    = mcolors.Normalize(vmin=degrees.min(), vmax=degrees.max())
    cmap    = cm.plasma
    colors  = [cmap(norm(d)) for d in degrees]
    sizes   = 5 + 60 * (degrees / degrees.max())

    pos = nx.spring_layout(G, seed=42, k=0.15)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#0f0f1a")
    fig.patch.set_facecolor("#0f0f1a")

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.06,
                           edge_color="white", width=0.3)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=colors, node_size=sizes, alpha=0.9)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cbar.set_label("Node degree", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(
        f"Facebook Combined Network\n"
        f"{G_full.number_of_nodes()} nodes · {G_full.number_of_edges()} edges"
        + (f" (showing top-{max_nodes} by degree)" if G_full.number_of_nodes() > max_nodes else ""),
        color="white", fontsize=13, pad=12,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
#  PLOT: degree distribution
# ══════════════════════════════════════════════════════════════════════════

def plot_degree_distribution(combined_path: str) -> None:
    """Log-log degree distribution of the combined network."""
    G = load_combined_graph(combined_path)
    degrees = sorted([d for _, d in G.degree()], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f1a")

    for ax in axes:
        ax.set_facecolor("#111122")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#555")
        ax.spines["left"].set_color("#555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # linear
    axes[0].hist(degrees, bins=50, color="#4a9eff", edgecolor="none", alpha=0.85)
    axes[0].set_xlabel("Degree", color="white")
    axes[0].set_ylabel("Count", color="white")
    axes[0].set_title("Degree Distribution (linear)", color="white")

    # log-log
    unique, counts = np.unique(degrees, return_counts=True)
    axes[1].scatter(unique, counts, s=15, color="#ff7c4a", alpha=0.8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Degree (log)", color="white")
    axes[1].set_ylabel("Count (log)", color="white")
    axes[1].set_title("Degree Distribution (log-log)", color="white")

    fig.suptitle("Facebook Network — Degree Distribution", color="white",
                 fontsize=13)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── 1. All ego networks in a grid ──────────────────────────────────
    print("=" * 60)
    print("1/4  All ego networks (grid)")
    print("=" * 60)
    plot_all_ego_networks(TAR_PATH, cols=5)

    # ── 2. Single ego network (node 0 — largest) ───────────────────────
    print("=" * 60)
    print("2/4  Single ego network (node 0)")
    print("=" * 60)
    plot_ego_network(TAR_PATH, ego_id="0", layout="spring", show_circles=True)

    # # ── 3. Full combined network ────────────────────────────────────────
    # print("=" * 60)
    # print("3/4  Combined network (top 2000 nodes by degree)")
    # print("=" * 60)
    # plot_combined_network(COMBINED_PATH, max_nodes=2000)

    # # ── 4. Degree distribution ─────────────────────────────────────────
    # print("=" * 60)
    # print("4/4  Degree distribution")
    # print("=" * 60)
    # plot_degree_distribution(COMBINED_PATH)
