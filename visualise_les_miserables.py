import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

def plot_les_miserables():
    G = nx.les_miserables_graph()

    # community detection via greedy modularity
    from networkx.algorithms.community import greedy_modularity_communities
    communities = list(greedy_modularity_communities(G))
    node_community = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_community[node] = i

    palette = plt.cm.tab20.colors
    colors = [palette[node_community[n] % len(palette)] for n in G.nodes()]

    # size by degree
    degrees = np.array([G.degree(n) for n in G.nodes()])
    sizes = 100 + 600 * (degrees / degrees.max())

    # edge weights
    weights = nx.get_edge_attributes(G, 'weight')
    edge_widths = [G[u][v].get('weight', 1) * 0.4 for u, v in G.edges()]

    pos = nx.spring_layout(G, seed=42, k=0.6)

    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_facecolor("#0f0f1a")
    fig.patch.set_facecolor("#0f0f1a")

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35,
                           edge_color="white", width=edge_widths)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=colors, node_size=sizes, alpha=0.95)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            font_color="white", font_size=6.5, font_weight="bold")

    # weight labels
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=weights, ax=ax,
        font_size=5, font_color="#ffdd88",
        bbox=dict(boxstyle="round,pad=0.2", fc="#1e1e30", ec="none", alpha=0.7)
    )

    # legend: one entry per community
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=palette[i % len(palette)],
                   markersize=9, label=f"Community {i+1}")
        for i in range(len(communities))
    ]
    ax.legend(handles=handles, title="Communities", title_fontsize=9,
              fontsize=7, loc="upper left",
              facecolor="#1e1e30", labelcolor="white", edgecolor="#555")

    ax.set_title(
        f"Les Misérables Co-occurrence Network\n"
        f"{G.number_of_nodes()} characters · {G.number_of_edges()} edges · "
        f"{len(communities)} communities · edge thickness & labels = weight",
        color="white", fontsize=13, pad=12
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()

plot_les_miserables()