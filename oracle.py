import networkx as nx
import numpy as np
import math

class Oracle:
    def __init__(self, adj_matrix):
        np_matrix = np.array(adj_matrix)
        self.__num_vertices = len(np_matrix)
        self.__graph = nx.from_numpy_array(np_matrix, create_using=nx.Graph())
        self.__shortest_paths = dict(nx.all_pairs_dijkstra_path_length(self.__graph))

    def qd(self,u,v):
        # distance query: sum of weights of shortest path
        return self.__shortest_paths.get(u, {}).get(v, math.inf)

    def qw(self,u,v):
        # Edge weight query, return w(u,v)
        edge_data = self.__graph.get_edge_data(u, v)
        return edge_data.get('weight', math.inf) if edge_data else math.inf

    def qc(self, u, S, w_thr):
        # Component query: Returns 1 if u and any v in S belong to the same 
        # connected component in the subgraph G[w >= w_thr], else 0.
        edges_above_thr = [
            (i, j) for i, j, d in self.__graph.edges(data=True) 
            if d.get('weight', 0) >= w_thr
        ]
        subgraph = nx.Graph()
        subgraph.add_nodes_from(self.__graph.nodes())
        subgraph.add_edges_from(edges_above_thr)
        if u not in subgraph:
            return 0
        component_u = nx.node_connected_component(subgraph, u)
        for v in S:
            if v in component_u:
                return 1
        return 0