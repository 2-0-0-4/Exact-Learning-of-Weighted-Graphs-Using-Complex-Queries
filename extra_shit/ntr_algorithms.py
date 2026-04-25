import math
from lblr_algorithms import exhaustive_query
from helper import sample_set, multiset_random_sample

def estimated_centers_ntr(oracle, V_list, s, K=2):
    """
    Algorithm 4 modified for NT-R. 
    Wthr is implicitly 1. Returns (A, dist_A_cache).
    """
    n = len(V_list)
    if n <= 1:
        return set(V_list), {v: {v: 0} for v in V_list}
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
                dist_A_cache[a] = {v: oracle.qd(a, v, 1) for v in V_list}        
        A |= A_prime
        new_W = set()        
        for w in W:
            X = multiset_random_sample(V_list, T)
            count = 0
            for x in X:
                dwx = oracle.qd(w, x, 1)
                dAx = min(dist_A_cache[a][x] for a in A)
                if dwx < dAx:
                    count += 1
            est_size = (count * n) / T
            if est_size >= (5.0 * n / s):
                new_W.add(w)
        W = new_W
    return A, dist_A_cache

def find_neighbors_ntr(a, V_list, Wmax, dist_A_cache):
    """
    Replaces FIND-NEIGHBORS with the closed ball B(a, 2Wmax).
    No new oracle queries are needed here as dist_A_cache contains A x V.
    """
    return {v for v in V_list if dist_A_cache[a][v] <= 2 * Wmax}


def reconstruct_sub_ntr(oracle, V, Wmax, D_bound):
    V_list = list(V)
    n = len(V_list)
    if n <= 1: return {}
    if D_bound <= 1:
        b = 2 * Wmax + 1
    else:
        b = min(n, (D_bound**(2 * Wmax + 1) - 1) / (D_bound - 1))
    s = max(1, int(math.ceil(math.sqrt(b * n))))
    s = min(n, s)
    A, dist_A = estimated_centers_ntr(oracle, V_list, s)
    min_dist_to_A = {v: min(dist_A[a][v] for a in A) for v in V_list}
    recovered = {}
    for a in A:
        ball_a = find_neighbors_ntr(a, V_list, Wmax, dist_A)
        dist_ball = {}
        for u in ball_a:
            dist_ball[u] = {v: oracle.qd(u, v, 1) for v in V_list}
        
        D_a = set(ball_a)
        for u in ball_a:
            for v in V_list:
                if dist_ball[u][v] < min_dist_to_A[v]:
                    D_a.add(v)
        # print(f"Center {a}: Ball size={len(ball_a)}, Cluster size={len(D_a)}")  
        E_a = exhaustive_query(oracle, D_a, w_thr=1)
        recovered.update(E_a)

    return recovered

def reconstruct_ntr(oracle, V, Wmax, D_bound):
    V_list = list(V)
    n = len(V_list)
    if n <= 1: return {}
    log2n = math.log2(max(2, n))
    Q_limit = int(100 * (D_bound**3) * Wmax * (n**1.5) * (log2n**2))
    attempts = 0
    while True:
        attempts += 1
        start_q = oracle.query_count
        E = reconstruct_sub_ntr(oracle, V_list, Wmax, D_bound)
        queries_used = oracle.query_count - start_q
        if queries_used <= Q_limit:
            return E       
        print(f"NT-R limit exceeded ({queries_used}/{Q_limit}). Retrying...")