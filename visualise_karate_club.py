import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

def plot_karate_club():
    G = nx.karate_club_graph()

    # colour by community (the two factions after the split)
    community = {n: G.nodes[n]['club'] for n in G.nodes()}
    color_map = {'Mr. Hi': '#4a9eff', 'Officer': '#ff4a4a'}
    colors = [color_map[community[n]] for n in G.nodes()]

    # size by degree
    degrees = np.array([G.degree(n) for n in G.nodes()])
    sizes = 100 + 400 * (degrees / degrees.max())

    pos = nx.spring_layout(G, seed=42, k=0.5)

    # edge weights
    weights = nx.get_edge_attributes(G, 'weight')
    edge_widths = [G[u][v].get('weight', 1) * 0.8 for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_facecolor("#0f0f1a")
    fig.patch.set_facecolor("#0f0f1a")

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4,
                           edge_color="white", width=edge_widths)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=colors, node_size=sizes, alpha=0.95)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            font_color="white", font_size=7, font_weight="bold")

    # draw weight labels on edges
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=weights, ax=ax,
        font_size=6, font_color="#ffdd88",
        bbox=dict(boxstyle="round,pad=0.2", fc="#1e1e30", ec="none", alpha=0.7)
    )

    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#4a9eff', markersize=10, label="Mr. Hi's faction"),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#ff4a4a', markersize=10, label="Officer's faction"),
        plt.Line2D([0], [0], color='white', linewidth=1, label="weight = 1"),
        plt.Line2D([0], [0], color='white', linewidth=3, label="weight = 3+"),
    ]
    ax.legend(handles=handles, facecolor="#1e1e30", labelcolor="white",
              edgecolor="#555", fontsize=9)

    ax.set_title(
        f"Zachary's Karate Club Network\n"
        f"{G.number_of_nodes()} nodes · {G.number_of_edges()} edges · "
        f"edge thickness & labels = weight",
        color="white", fontsize=13, pad=12
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()

plot_karate_club()