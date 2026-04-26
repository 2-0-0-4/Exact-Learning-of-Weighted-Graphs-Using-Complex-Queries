
import csv
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

# ── Project imports ──────────────────────────────────────────────────────────
from oracle import Oracle
from helper import norm_edge
from lblr_algorithms import lbl_r
from graph_generator import generate_disconnected_graph, print_graph_info
from conn_graph_gen import (
    generate_connected_graph_suite,
    print_graph_info as print_conn_info,
)


# ============================================================================
# Helper / theory utilities
# ============================================================================

def _safe_log2(x: float) -> float:
    return math.log2(x) if x > 1 else 1e-12


def _safe_loglog(x: float) -> float:
    return math.log(math.log(x)) if x > math.e else 1e-12


def theory_bound_disconnected(n: int, D: int, alpha: float) -> float:
    if n <= 1:
        return 1.0
    log2n = _safe_log2(n)
    loglogn = _safe_loglog(n)
    log_factor = 1.0 + (1.0 / alpha) * _safe_log2(max(D, 2))
    return log_factor * (D ** 3) * (n ** 1.5) * (log2n ** 2) * loglogn


def theory_bound_connected(n: int, D: int, W_max: float) -> float:
    if n <= 1:
        return 1.0
    log2n = _safe_log2(n)
    loglogn = _safe_loglog(n)
    return (D ** 3) * W_max * (n ** 1.5) * (log2n ** 2) * loglogn


def fit_loglog_slope(x_vals: List[float], y_vals: List[float]) -> float:
    """Least-squares slope in log-log space (empirical complexity exponent)."""
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(slope)


# ============================================================================
# Core runner
# ============================================================================

def run_lblr(
    G: nx.Graph,
    label: str,
    verbose: bool = True,
) -> dict:
    adj = nx.to_numpy_array(G, weight="weight")
    all_nodes = list(G.nodes())
    n = G.number_of_nodes()
    m = G.number_of_edges()
    k = nx.number_connected_components(G)

    D_max = max(dict(G.degree()).values()) if n > 0 else 1
    weights_list = [d.get("weight", 1.0) for _, _, d in G.edges(data=True)]
    W_max = max(weights_list) if weights_list else 1.0

    true_edges = {
        norm_edge(u, v): d.get("weight", 1.0)
        for u, v, d in G.edges(data=True)
    }

    if verbose:
        print(f"  [{label}]  n={n}, m={m}, k={k}, D={D_max}, "
              f"Wmax={W_max:.2f} …", end=" ", flush=True)

    oracle = Oracle(adj)
    t0 = time.perf_counter()
    recovered = lbl_r(oracle, all_nodes, int(math.ceil(W_max)), D_max)
    elapsed = time.perf_counter() - t0

    correct = sum(
        1 for e, w in true_edges.items()
        if abs(recovered.get(e, float("nan")) - w) < 1e-9
    )
    accuracy = 100.0 * correct / len(true_edges) if true_edges else 0.0

    if verbose:
        print(f"queries={oracle.query_count:,}  acc={accuracy:.1f}%  "
              f"time={elapsed:.2f}s")

    return {
        # Graph properties
        "n": n,
        "m": m,
        "k_components": k,
        "D_max": D_max,
        "W_max": float(W_max),
        # Algorithm results
        "queries": oracle.query_count,
        "edges_recovered": len(recovered),
        "edges_correct": correct,
        "accuracy_pct": accuracy,
        "time_sec": elapsed,
    }


# ============================================================================
# Test-case runner (connected + disconnected pair)
# ============================================================================

def run_pair(
    n: int,
    alpha: float,
    extra_edges: int,
    max_degree: int,
    num_components: int,      # only for disconnected
    component_size: int,      # only for disconnected
    seed: int,
    label: str,
    verbose: bool = True,
) -> dict:
    # ── Disconnected graph ───────────────────────────────────────────────────
    G_dis, meta_dis = generate_disconnected_graph(
        num_components=num_components,
        component_size=component_size,
        alpha=alpha,
        extra_edges_per_component=extra_edges,
        max_degree=max_degree,
        seed=seed,
        verbose=False,
    )
    n_dis = G_dis.number_of_nodes()

    # ── Connected graph (same total n) ───────────────────────────────────────
    G_con, meta_con = generate_connected_graph_suite(
        n=n_dis,           # match vertex count exactly
        alpha=alpha,
        extra_edges=extra_edges * num_components,   # proportional density
        max_degree=max_degree,
        seed=seed,
        verbose=False,
    )

    if verbose:
        print(f"\n{'-' * 72}")
        print(f"  {label}")
        print(f"  alpha={alpha}  D={max_degree}  n={n_dis}  components={num_components}")
        print(f"{'-' * 72}")

    r_dis = run_lblr(G_dis, label="DISCONNECTED", verbose=verbose)
    r_con = run_lblr(G_con, label="CONNECTED   ", verbose=verbose)

    # ── Theory bounds ────────────────────────────────────────────────────────
    tb_dis = theory_bound_disconnected(r_dis["n"], r_dis["D_max"], alpha)
    tb_con = theory_bound_connected(r_con["n"], r_con["D_max"], r_con["W_max"])

    # ── Ratio metrics ────────────────────────────────────────────────────────
    q_ratio = (r_con["queries"] / r_dis["queries"]
               if r_dis["queries"] > 0 else float("nan"))
    t_ratio = (r_con["time_sec"] / r_dis["time_sec"]
               if r_dis["time_sec"] > 1e-9 else float("nan"))

    row = {
        # Metadata
        "label": label,
        "alpha": alpha,
        "max_degree": max_degree,
        "num_components": num_components,
        "component_size": component_size,
        "seed": seed,
        # Disconnected
        "dis_n": r_dis["n"],
        "dis_m": r_dis["m"],
        "dis_k": r_dis["k_components"],
        "dis_D_max": r_dis["D_max"],
        "dis_W_max": r_dis["W_max"],
        "dis_queries": r_dis["queries"],
        "dis_edges_recovered": r_dis["edges_recovered"],
        "dis_accuracy_pct": r_dis["accuracy_pct"],
        "dis_time_sec": r_dis["time_sec"],
        "dis_theory_bound": tb_dis,
        "dis_queries_over_theory": r_dis["queries"] / tb_dis if tb_dis > 0 else float("nan"),
        # Connected
        "con_n": r_con["n"],
        "con_m": r_con["m"],
        "con_D_max": r_con["D_max"],
        "con_W_max": r_con["W_max"],
        "con_queries": r_con["queries"],
        "con_edges_recovered": r_con["edges_recovered"],
        "con_accuracy_pct": r_con["accuracy_pct"],
        "con_time_sec": r_con["time_sec"],
        "con_theory_bound": tb_con,
        "con_queries_over_theory": r_con["queries"] / tb_con if tb_con > 0 else float("nan"),
        # Comparison
        "query_ratio_con_over_dis": q_ratio,
        "time_ratio_con_over_dis": t_ratio,
        "dis_passes": r_dis["accuracy_pct"] == 100.0,
        "con_passes": r_con["accuracy_pct"] == 100.0,
    }

    return row


# ============================================================================
# CSV persistence
# ============================================================================

CSV_COLUMNS = [
    # Metadata
    "label", "alpha", "max_degree", "num_components", "component_size", "seed",
    # Disconnected graph props + results
    "dis_n", "dis_m", "dis_k", "dis_D_max", "dis_W_max",
    "dis_queries", "dis_edges_recovered", "dis_accuracy_pct", "dis_time_sec",
    "dis_theory_bound", "dis_queries_over_theory",
    # Connected graph props + results
    "con_n", "con_m", "con_D_max", "con_W_max",
    "con_queries", "con_edges_recovered", "con_accuracy_pct", "con_time_sec",
    "con_theory_bound", "con_queries_over_theory",
    # Comparison
    "query_ratio_con_over_dis", "time_ratio_con_over_dis",
    "dis_passes", "con_passes",
    # Scaling-test extras (filled in post-hoc)
    "scaling_dis_loglog_slope", "scaling_con_loglog_slope",
]


def save_csv(rows: List[dict], path: Path) -> None:
    """Write results rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",   # ignore extra keys
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================================
# Test-case definitions
# ============================================================================

# Each entry drives one connected/disconnected pair.
TEST_CASES = [

    # ── Baseline: standard parameters from the paper ─────────────────────────
    {
        "label": "Baseline (standard params)",
        "alpha": 2.0, "max_degree": 4,
        "num_components": 10, "component_size": 20,
        "extra_edges": 2, "seed": 42,
    },

    # ── Effect of Pareto alpha ────────────────────────────────────────────────────
    {
        "label": "Heavy-tailed weights (alpha=1.2)",
        "alpha": 1.2, "max_degree": 4,
        "num_components": 10, "component_size": 20,
        "extra_edges": 2, "seed": 42,
    },
    {
        "label": "Light-tailed weights (alpha=5.0)",
        "alpha": 5.0, "max_degree": 4,
        "num_components": 10, "component_size": 20,
        "extra_edges": 2, "seed": 42,
    },

    # ── Effect of degree bound ────────────────────────────────────────────────
    {
        "label": "Low degree (D=2, tree-like)",
        "alpha": 2.0, "max_degree": 2,
        "num_components": 10, "component_size": 20,
        "extra_edges": 0, "seed": 42,
    },
    {
        "label": "Medium degree (D=6)",
        "alpha": 2.0, "max_degree": 6,
        "num_components": 10, "component_size": 20,
        "extra_edges": 4, "seed": 42,
    },
    {
        "label": "High degree (D=10)",
        "alpha": 2.0, "max_degree": 10,
        "num_components": 5, "component_size": 30,
        "extra_edges": 8, "seed": 42,
    },

    # ── Effect of component structure ─────────────────────────────────────────
    {
        "label": "Many small components (k=50, size=10)",
        "alpha": 2.0, "max_degree": 4,
        "num_components": 50, "component_size": 10,
        "extra_edges": 1, "seed": 42,
    },
    {
        "label": "Few large components (k=2, size=100)",
        "alpha": 2.0, "max_degree": 4,
        "num_components": 2, "component_size": 100,
        "extra_edges": 3, "seed": 42,
    },

    # ── Dense graphs ──────────────────────────────────────────────────────────
    {
        "label": "Dense graph (many extra edges)",
        "alpha": 2.0, "max_degree": 8,
        "num_components": 5, "component_size": 40,
        "extra_edges": 15, "seed": 42,
    },
]

# Scaling tests: n grows, everything else fixed
SCALING_N = [50, 100, 200, 400]

for _n in SCALING_N:
    _k = max(2, _n // 20)
    _sz = max(5, _n // _k)
    TEST_CASES.append({
        "label": f"Scaling n={_n}",
        "alpha": 2.0, "max_degree": 4,
        "num_components": _k, "component_size": _sz,
        "extra_edges": 2, "seed": 42,
    })


# ============================================================================
# Main
# ============================================================================

def main(verbose: bool = True) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("saved_results")
    csv_path = out_dir / f"connected_vs_disconnected_{timestamp}.csv"

    print("=" * 72)
    print("LBL-R  —  Connected vs Disconnected Graph Comparison")
    print("=" * 72)
    print(f"Test cases  : {len(TEST_CASES)}")
    print(f"Output CSV  : {csv_path}")
    print()

    all_rows: List[dict] = []

    for idx, case in enumerate(TEST_CASES, start=1):
        print(f"[{idx:>2}/{len(TEST_CASES)}]", end=" ")
        row = run_pair(
            n=case["num_components"] * case["component_size"],
            alpha=case["alpha"],
            extra_edges=case["extra_edges"],
            max_degree=case["max_degree"],
            num_components=case["num_components"],
            component_size=case["component_size"],
            seed=case["seed"],
            label=case["label"],
            verbose=verbose,
        )
        all_rows.append(row)

    # ── Compute log-log slopes for scaling tests ─────────────────────────────
    scaling_rows = [r for r in all_rows if r["label"].startswith("Scaling")]
    if len(scaling_rows) >= 2:
        scaling_rows.sort(key=lambda r: r["dis_n"])
        ns = [r["dis_n"] for r in scaling_rows]
        dis_qs = [r["dis_queries"] for r in scaling_rows]
        con_qs = [r["con_queries"] for r in scaling_rows]
        slope_dis = fit_loglog_slope(ns, dis_qs)
        slope_con = fit_loglog_slope(ns, con_qs)
        for r in scaling_rows:
            r["scaling_dis_loglog_slope"] = slope_dis
            r["scaling_con_loglog_slope"] = slope_con

    # ── Save CSV ─────────────────────────────────────────────────────────────
    save_csv(all_rows, csv_path)

    # ── Console summary ───────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("SUMMARY  (LBL-R queries: disconnected vs connected)")
    print("=" * 72)
    hdr = (f"{'Label':<40} {'n':>5} {'dis Q':>10} {'con Q':>10} "
           f"{'Q ratio':>8} {'dis acc':>8} {'con acc':>8}")
    print(hdr)
    print("─" * 72)

    all_dis_pass = True
    all_con_pass = True

    for row in all_rows:
        ratio_str = (f"{row['query_ratio_con_over_dis']:>8.2f}x"
                     if math.isfinite(row["query_ratio_con_over_dis"])
                     else f"{'N/A':>9}")
        print(f"{row['label']:<40} {row['dis_n']:>5} "
              f"{row['dis_queries']:>10,} {row['con_queries']:>10,} "
              f"{ratio_str} "
              f"{row['dis_accuracy_pct']:>7.1f}% {row['con_accuracy_pct']:>7.1f}%")
        all_dis_pass = all_dis_pass and row["dis_passes"]
        all_con_pass = all_con_pass and row["con_passes"]

    print("─" * 72)

    # ── Accuracy validation ───────────────────────────────────────────────────
    print()
    print("ACCURACY VALIDATION")
    print(f"  Disconnected (all 100%): {'PASS ' if all_dis_pass else 'FAIL '}")
    print(f"  Connected    (all 100%): {'PASS ' if all_con_pass else 'FAIL '}")

    # ── Scaling analysis ──────────────────────────────────────────────────────
    if len(scaling_rows) >= 2:
        slope_dis = scaling_rows[0].get("scaling_dis_loglog_slope", float("nan"))
        slope_con = scaling_rows[0].get("scaling_con_loglog_slope", float("nan"))
        print()
        print("SCALING ANALYSIS  (log-log slope  approx empirical complexity exponent)")
        print(f"  Disconnected log-log slope : {slope_dis:.3f}  "
              f"(theory: ~1.5 ignoring polylog)")
        print(f"  Connected    log-log slope : {slope_con:.3f}  "
              f"(theory: ~1.5 ignoring polylog)")
        if math.isfinite(slope_dis) and math.isfinite(slope_con):
            both_subquad = slope_dis < 2.0 and slope_con < 2.0
            print(f"  Both sub-quadratic (<2.0)  : "
                  f"{'PASS ' if both_subquad else 'FAIL '}")

    # ── Theory-ratio summary (disconnected) ───────────────────────────────────
    ratios_dis = [r["dis_queries_over_theory"] for r in all_rows
                  if math.isfinite(r["dis_queries_over_theory"])]
    ratios_con = [r["con_queries_over_theory"] for r in all_rows
                  if math.isfinite(r["con_queries_over_theory"])]
    if ratios_dis:
        print()
        print("THEORY-BOUND RATIOS  (actual queries / theoretical bound)")
        print(f"  Disconnected  mean={np.mean(ratios_dis):.4f}  "
              f"min={min(ratios_dis):.4f}  max={max(ratios_dis):.4f}")
    if ratios_con:
        print(f"  Connected     mean={np.mean(ratios_con):.4f}  "
              f"min={min(ratios_con):.4f}  max={max(ratios_con):.4f}")

    print()
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main(verbose=True)