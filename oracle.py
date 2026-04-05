import networkx as nx
import numpy as np
import math

class Oracle:
    def __init__(self, adj_matrix):
        """
        Initializes the oracle with the full weighted graph G.
        The adjacency matrix should have 0 or math.inf for no edge, 
        and real weights >= 1 for existing edges.
        """
        self.graph = nx.from_numpy_array(np.array(adj_matrix), create_using=nx.Graph())
        self.query_count = 0
        self.subgraph_cache = {}

    def _get_thresholded_subgraph(self, w_thr):
        """
        Returns G[w >= w_thr]: the subgraph including all vertices 
        but only edges with weight >= w_thr.
        """
        if w_thr not in self.subgraph_cache:
            edges_above = [
                (u, v, d) for u, v, d in self.graph.edges(data=True) 
                if d.get('weight', 0) >= w_thr
            ]
            sub = nx.Graph()
            sub.add_nodes_from(self.graph.nodes())
            sub.add_edges_from(edges_above)
            self.subgraph_cache[w_thr] = sub
        return self.subgraph_cache[w_thr]

    def qd(self, u, v, w_thr=1):
        """
        Distance query: Returns the sum of edge weights on a shortest 
        weighted path in G[w >= w_thr]. Returns math.inf if no path.
        """
        self.query_count += 1
        sub = self._get_thresholded_subgraph(w_thr)
        try:
            return nx.shortest_path_length(sub, source=u, target=v, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return math.inf

    def qw(self, u, v, w_thr=1):
        """
        Edge-weight query: Returns w(u, v) if the edge exists in G[w >= w_thr].
        Returns 0 if no such edge exists.
        """
        self.query_count += 1
        sub = self._get_thresholded_subgraph(w_thr)
        edge_data = sub.get_edge_data(u, v)
        return edge_data.get('weight', 0) if edge_data else 0

    def qc(self, u, S, w_thr=1):
        """
        Component query: Returns 1 if u and some v in S are in the 
        same connected component of G[w >= w_thr], else 0.
        """
        self.query_count += 1
        sub = self._get_thresholded_subgraph(w_thr)
        if u not in sub: return 0
        
        comp_u = nx.node_connected_component(sub, u)
        return 1 if any(v in comp_u for v in S) else 0