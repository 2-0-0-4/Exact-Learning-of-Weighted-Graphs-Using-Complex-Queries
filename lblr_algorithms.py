import math
from helper import norm_edge, sample_set, multiset_random_sample

def find_connected_components(oracle, V, w_thr):
    V_list = list(V)
    if not V_list:
        return []
    components = [set([V_list[0]])]
    all_nodes_discovered = {V_list[0]}
    for i in range(1, len(V_list)):
        vi = V_list[i]
        if oracle.qc(vi, all_nodes_discovered, w_thr) == 0:
            new_comp = {vi}
            components.append(new_comp)
            all_nodes_discovered.add(vi)
        else:
            lo, hi = 0, len(components) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                left_subset_union = set().union(*components[lo:mid + 1])                
                if oracle.qc(vi, left_subset_union, w_thr) == 1:
                    hi = mid
                else:
                    lo = mid + 1
            
            components[lo].add(vi)
            all_nodes_discovered.add(vi)
    return components

def exhaustive_query(oracle, V, w_thr):
    V = list(V)
    recovered = {}
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            w = oracle.qw(V[i], V[j])
            if w >= w_thr and w != float('inf'):
                recovered[norm_edge(V[i], V[j])] = w
    return recovered

def find_neighbors(oracle, V, a, w_thr):
    N_a = {v for v in V if v != a and oracle.qw(a, v, w_thr) != 0}
    N2_a = set(N_a)
    for v in N_a:
        N_v = {u for u in V if u != v and oracle.qw(u, v, w_thr) != 0}
        N2_a.update(N_v)
    return N2_a
import math
from helper import sample_set, multiset_random_sample

def estimated_centers(oracle, V, s, w_thr, K=2):
    V_list = list(V)
    n = len(V_list)
    if n <= 1:
        return set(V_list)
    A = set()
    W = set(V_list)
    logn = math.log(n) if n > 1 else 1
    loglogn = math.log(logn) if logn > 1 else 1
    T = max(1, int(K * s * logn * loglogn))
    dist_A_cache = {}
    while W:
        A_prime = sample_set(W, s)
        for a in A_prime:
            if a not in dist_A_cache:
                dist_A_cache[a] = {v: oracle.qd(a, v, w_thr) for v in V_list}
        A |= A_prime
        new_W = set()
        for w in W:
            X = multiset_random_sample(V_list, T)
            count = 0
            for x in X:
                dwx = oracle.qd(w, x, w_thr)
                dAx = min(dist_A_cache[a][x] for a in A)
                if dwx < dAx:
                    count += 1
            est_size = (count * n) / T
            if est_size >= (5.0 * n / s):
                new_W.add(w)
        
        W = new_W
    return A

def reconstruct_sub(oracle, V, w_thr, D_bound):
    V_list = list(V)
    n = len(V_list)
    if n <= 1:
        return {}
    s = max(1, int(D_bound * math.sqrt(n)))
    A = estimated_centers(oracle, V_list, s, w_thr)
    dist_A = {a: {v: oracle.qd(a, v, w_thr) for v in V_list} for a in A}
    recovered = {}
    for a in A:
        N2_a = find_neighbors(oracle, V_list, a, w_thr)
        dist_N2 = {b: {v: oracle.qd(b, v, w_thr) for v in V_list} for b in N2_a}
        D_a = set(N2_a)
        for b in N2_a:
            for v in V_list:
                dbv = dist_N2[b][v]
                dAv = min(dist_A[alt][v] for alt in A)
                if dbv < dAv:
                    D_a.add(v)
        E_a = exhaustive_query(oracle, D_a, w_thr)
        recovered.update(E_a)
    return recovered


def reconstruct(oracle, V, w_thr, D_bound):
    V_list = list(V)
    n = len(V_list)
    if n <= 1:
        return {}
    logn = math.log(n) if n > 1 else 1
    log2n = math.log2(n) if n > 1 else 1
    loglogn = math.log(logn) if logn > 1 else 1
    Q_limit = int(10 * (D_bound**3) * (n**1.5) * (log2n**2) * loglogn)
    while True:
        start_q = oracle.query_count
        E = reconstruct_sub(oracle, V_list, w_thr, D_bound)
        queries_used = oracle.query_count - start_q
        if queries_used <= Q_limit:
            return E
        print(f"Query limit {Q_limit} exceeded ({queries_used}). Retrying...")

def lbl_r(oracle, all_vertices, W_max, D_bound):
    n = len(all_vertices)
    small_thr = n ** 0.25
    recovered = {}
    max_j = int(math.floor(math.log2(W_max))) if W_max >= 1 else 0
    for j in range(max_j + 1):
        w_thr = 2**j
        comps = find_connected_components(oracle, all_vertices, w_thr)
        for comp in comps:
            if len(comp) <= small_thr:
                E_comp = exhaustive_query(oracle, comp, w_thr)
            else:
                E_comp = reconstruct(oracle, comp, w_thr, D_bound)
            recovered.update(E_comp)        
        if all(len(c) <= small_thr for c in comps):
            break
    return recovered