"""
LBL-R vs EXHAUSTIVE-QUERY Comparison on Disconnected Weighted Graphs

This test file compares the query complexity and accuracy of:
- LBL-R (Layer-by-Layer Reconstruction with weight thresholds)
- EXHAUSTIVE-QUERY (Brute-force O(n²) algorithm)

On disconnected graphs with Pareto-distributed weights, which is the
primary use case from the paper.

Paper: "Exact Learning of Weighted Graphs Using Composite Queries"
       Goodrich, Liu, Panageas (arXiv:2511.14882v1)
"""

import time
import csv
import json
import math
import networkx as nx
import numpy as np
from pathlib import Path
from datetime import datetime

from oracle import Oracle
from helper import norm_edge
from lblr_algorithms import lbl_r, exhaustive_query
from graph_generator import (
    generate_disconnected_graph,
    print_graph_info
)


def _safe_loglog(n):
    """Return log(log(n)) with a safe lower bound to avoid invalid values."""
    if n <= 2:
        return 1.0
    return max(math.log(math.log(n)), 1e-12)


def lblr_theory_bound(n, d_bound):
    """Theoretical scale used in reconstruct() query-limit expression."""
    if n <= 1:
        return 1.0
    log2n = max(math.log2(n), 1e-12)
    return (d_bound ** 3) * (n ** 1.5) * (log2n ** 2) * _safe_loglog(n)


def fit_loglog_slope(x_values, y_values):
    """Fit slope in log-log space for empirical complexity estimation."""
    if len(x_values) < 2:
        return float('nan')

    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)

    valid = (x > 0) & (y > 0)
    if np.sum(valid) < 2:
        return float('nan')

    lx = np.log(x[valid])
    ly = np.log(y[valid])
    slope, _ = np.polyfit(lx, ly, 1)
    return float(slope)


def run_comparison(G, graph_name="Test Graph", verbose=False):
    """
    Run LBL-R and EXHAUSTIVE-QUERY on the same graph and compare results.
    
    Args:
        G: networkx.Graph (disconnected, weighted)
        graph_name: Name for display
        verbose: Print detailed info
    
    Returns:
        Dictionary with results
    """
    
    # Convert to adjacency matrix
    adj = nx.to_numpy_array(G, weight='weight')
    
    # Extract ground truth and parameters
    all_nodes = list(G.nodes())
    D_MAX = max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0
    W_MAX = int(max(d.get('weight', 1) for _, _, d in G.edges(data=True))) if G.number_of_edges() > 0 else 1
    
    true_edges = {
        norm_edge(u, v): d.get('weight', 1)
        for u, v, d in G.edges(data=True)
    }
    
    if verbose:
        print("\n" + "=" * 120)
        print(f"COMPARISON: {graph_name}")
        print("=" * 120)
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"Connected components: {nx.number_connected_components(G)}")
        print(f"Parameters: D_MAX={D_MAX}, W_MAX={W_MAX}")
        print(f"True edges to recover: {len(true_edges)}")
        print()
    
    results = {}
    n_nodes = len(all_nodes)
    
    # ── 1. LBL-R (Layer-by-Layer Reconstruction) ────────────────────────────
    if verbose:
        print("Running LBL-R (with weight thresholds)...")
    
    oracle_lblr = Oracle(adj)
    t0 = time.time()
    recovered_lblr = lbl_r(oracle_lblr, all_nodes, W_MAX, D_MAX)
    t1 = time.time()
    
    correct_lblr = sum(
        1 for e, w in true_edges.items()
        if recovered_lblr.get(e) == w
    )
    
    results['LBL-R'] = {
        'queries': oracle_lblr.query_count,
        'edges_recovered': len(recovered_lblr),
        'edges_correct': correct_lblr,
        'accuracy_pct': (correct_lblr / len(true_edges)) * 100 if len(true_edges) > 0 else 0,
        'time_sec': t1 - t0,
    }

    theory_bound = lblr_theory_bound(n_nodes, D_MAX)
    lblr_ratio = results['LBL-R']['queries'] / theory_bound if theory_bound > 0 else float('inf')
    results['LBL-R']['theory_bound'] = theory_bound
    results['LBL-R']['over_theory_bound_ratio'] = lblr_ratio
    
    if verbose:
        print(f"  ✓ Completed in {results['LBL-R']['time_sec']:.3f}s")
        print(f"  ✓ Queries: {results['LBL-R']['queries']}")
        print(f"  ✓ Accuracy: {results['LBL-R']['accuracy_pct']:.1f}%")
    
    # ── 2. EXHAUSTIVE-QUERY (Brute-force O(n²)) ────────────────────────────
    if verbose:
        print("\nRunning EXHAUSTIVE-QUERY (brute-force, expected O(n²) queries)...")
    
    oracle_exhaustive = Oracle(adj)
    t0 = time.time()
    recovered_exhaustive = exhaustive_query(oracle_exhaustive, all_nodes, w_thr=1)
    t1 = time.time()
    
    correct_exhaustive = sum(
        1 for e, w in true_edges.items()
        if recovered_exhaustive.get(e) == w
    )
    
    results['EXHAUSTIVE-QUERY'] = {
        'queries': oracle_exhaustive.query_count,
        'edges_recovered': len(recovered_exhaustive),
        'edges_correct': correct_exhaustive,
        'accuracy_pct': (correct_exhaustive / len(true_edges)) * 100 if len(true_edges) > 0 else 0,
        'time_sec': t1 - t0,
    }

    expected_exhaustive = n_nodes * (n_nodes - 1) // 2
    exhaustive_error = results['EXHAUSTIVE-QUERY']['queries'] - expected_exhaustive
    results['EXHAUSTIVE-QUERY']['theory_expected_queries'] = expected_exhaustive
    results['EXHAUSTIVE-QUERY']['theory_error'] = exhaustive_error
    results['EXHAUSTIVE-QUERY']['theory_exact_match'] = (exhaustive_error == 0)
    
    if verbose:
        print(f"  ✓ Completed in {results['EXHAUSTIVE-QUERY']['time_sec']:.3f}s")
        print(f"  ✓ Queries: {results['EXHAUSTIVE-QUERY']['queries']}")
        print(f"  ✓ Accuracy: {results['EXHAUSTIVE-QUERY']['accuracy_pct']:.1f}%")
    
    # ── Display comparison ────────────────────────────────────────────────────
    if verbose:
        print("\n" + "-" * 120)
        print("RESULTS SUMMARY")
        print("-" * 120)
        print(f"{'Algorithm':<20} | {'Queries':<12} | {'Edges Recovered':<16} | {'Correct':<10} | {'Accuracy':<12} | {'Time (s)':<10}")
        print("-" * 120)
        
        for algo_name, data in results.items():
            print(f"{algo_name:<20} | {data['queries']:<12} | {data['edges_recovered']:<16} | "
                  f"{data['edges_correct']:<10} | {data['accuracy_pct']:>10.1f}% | {data['time_sec']:>9.3f}")
        
        print("-" * 120)
        
        # Calculate improvement
        lblr_queries = results['LBL-R']['queries']
        exhaustive_queries = results['EXHAUSTIVE-QUERY']['queries']
        
        if exhaustive_queries > 0:
            reduction = 100.0 * (exhaustive_queries - lblr_queries) / exhaustive_queries
            speedup = exhaustive_queries / lblr_queries if lblr_queries > 0 else float('inf')
            
            print(f"\nQuery Complexity Improvement:")
            print(f"  LBL-R uses {reduction:.1f}% fewer queries than EXHAUSTIVE-QUERY")
            print(f"  Speedup factor: {speedup:.2f}x")
        
        # Time improvement
        lblr_time = results['LBL-R']['time_sec']
        exhaustive_time = results['EXHAUSTIVE-QUERY']['time_sec']
        
        if exhaustive_time > 0:
            time_reduction = 100.0 * (exhaustive_time - lblr_time) / exhaustive_time
            time_speedup = exhaustive_time / lblr_time if lblr_time > 0 else float('inf')
            
            print(f"\nRuntime Improvement:")
            print(f"  LBL-R is {time_speedup:.2f}x faster than EXHAUSTIVE-QUERY")
    
    return results


def save_results_to_files(summary_data):
    """Save final comparison results to timestamped JSON and CSV files."""
    results_dir = Path(__file__).with_name("saved_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"comparison_results_{timestamp}.json"
    csv_path = results_dir / f"comparison_results_{timestamp}.csv"

    rows = []
    for test_name, n_nodes, results in summary_data:
        lblr_q = results['LBL-R']['queries']
        exh_q = results['EXHAUSTIVE-QUERY']['queries']
        reduction = 100.0 * (exh_q - lblr_q) / exh_q if exh_q > 0 else 0.0
        speedup = exh_q / lblr_q if lblr_q > 0 else float('inf')
        is_scaling_test = test_name.startswith("Scaling graph n=")

        rows.append({
            "test_case": test_name,
            "nodes": n_nodes,
            "is_scaling_test": is_scaling_test,
            "lblr_queries": lblr_q,
            "exhaustive_queries": exh_q,
            "exhaustive_theory_expected_queries": results['EXHAUSTIVE-QUERY']['theory_expected_queries'],
            "exhaustive_theory_error": results['EXHAUSTIVE-QUERY']['theory_error'],
            "exhaustive_theory_exact_match": results['EXHAUSTIVE-QUERY']['theory_exact_match'],
            "lblr_theory_bound": results['LBL-R']['theory_bound'],
            "lblr_over_theory_bound_ratio": results['LBL-R']['over_theory_bound_ratio'],
            "query_reduction_pct": reduction,
            "speedup_x": speedup,
            "lblr_accuracy_pct": results['LBL-R']['accuracy_pct'],
            "exhaustive_accuracy_pct": results['EXHAUSTIVE-QUERY']['accuracy_pct'],
            "lblr_time_sec": results['LBL-R']['time_sec'],
            "exhaustive_time_sec": results['EXHAUSTIVE-QUERY']['time_sec'],
        })

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


test_cases = [
    {
        "title": "TEST 1: SMALL DISCONNECTED GRAPH",
        "graph_name": "Small Test Graph (2 components, 5 nodes each)",
        "num_components": 2,
        "component_size": 5,
        "alpha": 2.0,
        "extra_edges_per_component": 1,
        "max_degree": 4,
        "seed": 42,
    },
    {
        "title": "TEST 2: MEDIUM DISCONNECTED GRAPH",
        "graph_name": "Medium Test Graph (10 components, 50 nodes each)",
        "num_components": 10,
        "component_size": 50,
        "alpha": 2.0,
        "extra_edges_per_component": 2,
        "max_degree": 4,
        "seed": 42,
    },
    {
        "title": "TEST 3: LARGE DISCONNECTED GRAPH",
        "graph_name": "Large Test Graph (50 components, 20 nodes each)",
        "num_components": 50,
        "component_size": 20,
        "alpha": 2.0,
        "extra_edges_per_component": 2,
        "max_degree": 4,
        "seed": 42,
    },
    {
        "title": "TEST 4: VERY LARGE DISCONNECTED GRAPH",
        "graph_name": "Large Test Graph (100 components, 20 nodes each)",
        "num_components": 100,
        "component_size": 20,
        "alpha": 2.0,
        "extra_edges_per_component": 2,
        "max_degree": 4,
        "seed": 42,
    },
    {
        "title": "TEST 5: LARGE COMPONENTS GRAPH",
        "graph_name": "Large Test Graph (50 components, 40 nodes each)",
        "num_components": 50,
        "component_size": 40,
        "alpha": 2.0,
        "extra_edges_per_component": 2,
        "max_degree": 4,
        "seed": 42,
    },

    # degree tests
    {
        "title": "TEST 6: TREE-LIKE LOW DEGREE GRAPH",
        "graph_name": "Sparse Forest",
        "num_components": 20,
        "component_size": 50,
        "alpha": 2.0,
        "extra_edges_per_component": 0,
        "max_degree": 2,
        "seed": 42,
    },
    {
        "title": "TEST 7: HIGHER DEGREE GRAPH",
        "graph_name": "Dense Bounded-Degree Graph",
        "num_components": 5,
        "component_size": 100,
        "alpha": 2.0,
        "extra_edges_per_component": 10,
        "max_degree": 20,
        "seed": 42,
    },

    # alpha sensitivity
    {
        "title": "TEST 8: HEAVY-TAILED WEIGHTS",
        "graph_name": "Pareto alpha = 1.2",
        "num_components": 20,
        "component_size": 50,
        "alpha": 1.2,
        "extra_edges_per_component": 2,
        "max_degree": 4,
        "seed": 42,
    },
    {
        "title": "TEST 9: LIGHT-TAILED WEIGHTS",
        "graph_name": "Pareto alpha = 5.0",
        "num_components": 20,
        "component_size": 50,
        "alpha": 5.0,
        "extra_edges_per_component": 2,
        "max_degree": 4,
        "seed": 42,
    },

    # connectivity / scale variants
    {
        "title": "TEST 10: FEW LARGE COMPONENTS",
        "graph_name": "Few Large Components",
        "num_components": 2,
        "component_size": 250,
        "alpha": 2.0,
        "extra_edges_per_component": 3,
        "max_degree": 6,
        "seed": 42,
    },
    {
        "title": "TEST 11: MANY SMALL COMPONENTS",
        "graph_name": "Many Small Components",
        "num_components": 100,
        "component_size": 10,
        "alpha": 2.0,
        "extra_edges_per_component": 1,
        "max_degree": 4,
        "seed": 42,
    },
    {
        "title": "TEST 12: TRANSITIVE EDGE HEAVY APPROXIMATION",
        "graph_name": "Triangle-Rich Approximation",
        "num_components": 5,
        "component_size": 50,
        "alpha": 2.0,
        "extra_edges_per_component": 20,
        "max_degree": 10,
        "seed": 42,
    },
]

for n in [100, 200, 400, 800, 1600]:
    test_cases.append({
        "title": f"SCALING TEST n={n}",
        "graph_name": f"Scaling graph n={n}",
        "num_components": max(1, n // 20),
        "component_size": 20,
        "alpha": 2.0,
        "extra_edges_per_component": 2,
        "max_degree": 4,
        "seed": 42,
    })
    

test_results = []

for idx, case in enumerate(test_cases, start=1):
    if idx > 1:
        print()
    print("=" * 120)
    print(case["title"])
    print("=" * 120)

    G, metadata = generate_disconnected_graph(
        num_components=case["num_components"],
        component_size=case["component_size"],
        alpha=case["alpha"],
        extra_edges_per_component=case["extra_edges_per_component"],
        max_degree=case["max_degree"],
        seed=case["seed"],
        verbose=False
    )

    print_graph_info(G, metadata)

    results = run_comparison(G, case["graph_name"], verbose=False)

    test_results.append((
        case["graph_name"],
        G.number_of_nodes(),
        results,
    ))


# ── Final Summary ──────────────────────────────────────────────────────────
print("\n\n" + "=" * 120)
print("FINAL SUMMARY ACROSS ALL TESTS")
print("=" * 120)

summary_data = test_results

print(f"\n{'Test Case':<48} | {'Nodes':<7} | {'LBL-R Queries':<15} | {'Exhaustive Queries':<18} | {'Reduction':<12} | {'Speedup':<8}")
print("-" * 120)

for test_name, n_nodes, results in summary_data:
    lblr_q = results['LBL-R']['queries']
    exh_q = results['EXHAUSTIVE-QUERY']['queries']
    reduction = 100.0 * (exh_q - lblr_q) / exh_q if exh_q > 0 else 0
    speedup = exh_q / lblr_q if lblr_q > 0 else float('inf')
    
    print(f"{test_name:<48} | {n_nodes:<7} | {lblr_q:<15} | {exh_q:<18} | {reduction:>10.1f}% | {speedup:>7.2f}x")

print("-" * 120)

print("\n" + "=" * 120)
print("VALIDATION CHECKLIST")
print("=" * 120)

all_pass = True

for test_name, _, results in summary_data:
    lblr_accuracy = results['LBL-R']['accuracy_pct']
    exh_accuracy = results['EXHAUSTIVE-QUERY']['accuracy_pct']
    
    lblr_ok = lblr_accuracy == 100.0
    exh_ok = exh_accuracy == 100.0
    
    status = "✓ PASS" if (lblr_ok and exh_ok) else "✗ FAIL"
    all_pass = all_pass and lblr_ok and exh_ok
    
    print(f"{test_name:<48} | LBL-R: {lblr_accuracy:>6.1f}% | Exhaustive: {exh_accuracy:>6.1f}% | {status}")

print("-" * 120)

if all_pass:
    print("\n✓ ALL TESTS PASSED - Both algorithms achieve 100% accuracy")
else:
    print("\n✗ SOME TESTS FAILED - Check algorithm implementations")

# ── Theory Validation Checklist ─────────────────────────────────────────────
print("\n" + "=" * 120)
print("THEORY VALIDATION CHECKLIST")
print("=" * 120)

exhaustive_exact_all = all(
    results['EXHAUSTIVE-QUERY']['theory_exact_match']
    for _, _, results in summary_data
)

print(
    f"Exhaustive baseline q = n(n-1)/2 exact on all tests: "
    f"{'✓ PASS' if exhaustive_exact_all else '✗ FAIL'}"
)

scaling_points = []
for test_name, n_nodes, results in summary_data:
    if test_name.startswith("Scaling graph n="):
        scaling_points.append((n_nodes, results))

if scaling_points:
    scaling_points.sort(key=lambda x: x[0])
    n_vals = [n for n, _ in scaling_points]
    lblr_q_vals = [r['LBL-R']['queries'] for _, r in scaling_points]
    exh_q_vals = [r['EXHAUSTIVE-QUERY']['queries'] for _, r in scaling_points]
    lblr_ratio_vals = [r['LBL-R']['over_theory_bound_ratio'] for _, r in scaling_points]

    exh_slope = fit_loglog_slope(n_vals, exh_q_vals)
    lblr_slope = fit_loglog_slope(n_vals, lblr_q_vals)

    ratio_finite = [x for x in lblr_ratio_vals if np.isfinite(x) and x > 0]
    if len(ratio_finite) >= 2:
        ratio_spread = max(ratio_finite) / min(ratio_finite)
    else:
        ratio_spread = float('inf')

    exhaustive_slope_ok = np.isfinite(exh_slope) and abs(exh_slope - 2.0) <= 0.10
    lblr_better_slope = np.isfinite(lblr_slope) and np.isfinite(exh_slope) and (lblr_slope < exh_slope)
    lblr_subquadratic = np.isfinite(lblr_slope) and (lblr_slope < 2.0)
    lblr_ratio_stable = np.isfinite(ratio_spread) and (ratio_spread <= 10.0)

    print(f"Scaling tests available: {len(scaling_points)}")
    print(f"Log-log slope (Exhaustive): {exh_slope:.3f}  [target ≈ 2]")
    print(f"Log-log slope (LBL-R):      {lblr_slope:.3f}  [should be < Exhaustive]")
    print(f"LBL-R normalized ratio spread max/min: {ratio_spread:.3f}")
    print(
        f"Exhaustive slope near 2.0: {'✓ PASS' if exhaustive_slope_ok else '✗ FAIL'}"
    )
    print(
        f"LBL-R slope lower than Exhaustive: {'✓ PASS' if lblr_better_slope else '✗ FAIL'}"
    )
    print(
        f"LBL-R empirically subquadratic (<2): {'✓ PASS' if lblr_subquadratic else '✗ FAIL'}"
    )
    print(
        f"LBL-R normalized ratio non-explosive: {'✓ PASS' if lblr_ratio_stable else '✗ FAIL'}"
    )
else:
    print("No scaling tests found (names starting with 'Scaling graph n=').")

print("-" * 120)

json_output_path, csv_output_path = save_results_to_files(summary_data)
print(f"\nSaved results to: {json_output_path}")
print(f"Saved results to: {csv_output_path}")

