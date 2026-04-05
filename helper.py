import networkx as nx
import numpy as np
import math
import random

def norm_edge(u, v):
    return (u, v) if u < v else (v, u)

def sample_set(W, s):
    W = list(W)
    if len(W) <= s: return set(W)
    p = float(s) / len(W)
    out = {x for x in W if random.random() < p}
    return out if out else {random.choice(W)}

def multiset_random_sample(V_list, T):
    return [random.choice(V_list) for _ in range(T)]