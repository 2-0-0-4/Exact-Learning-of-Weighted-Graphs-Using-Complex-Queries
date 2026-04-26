import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('saved_results\\connected_vs_disconnected_20260426_214951.csv')

# --- 1. PRE-PROCESSING ---
# Separate scaling experiments from scenario-based experiments
scaling_df = df[df['label'].str.contains('Scaling')].copy()
scaling_df['n'] = scaling_df['dis_n']  # Extract numeric n for the x-axis
scaling_df = scaling_df.sort_values('n')

scenario_df = df[~df['label'].str.contains('Scaling')].copy()

# --- 2. PLOT: QUERY COMPLEXITY SCALING ---
plt.figure(figsize=(10, 6))
plt.plot(scaling_df['n'], scaling_df['dis_queries'], 'o-', label='Disconnected (Empirical)', linewidth=2)
plt.plot(scaling_df['n'], scaling_df['con_queries'], 's-', label='Connected (Empirical)', linewidth=2)
plt.plot(scaling_df['n'], scaling_df['dis_theory_bound'], '--', label='Disconnected Theory', alpha=0.7)
plt.plot(scaling_df['n'], scaling_df['con_theory_bound'], '--', label='Connected Theory', alpha=0.7)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Nodes ($n$)')
plt.ylabel('Number of Queries')
plt.title('Query Complexity Scaling: Connected vs. Disconnected')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()
plt.savefig('query_complexity_scaling.png')
plt.close()

# --- 3. PLOT: TIME PERFORMANCE OVERHEAD ---
plt.figure(figsize=(12, 6))
x = np.arange(len(scenario_df))
width = 0.4

plt.bar(x - width/2, scenario_df['query_ratio_con_over_dis'], width, label='Query Ratio (Con/Dis)', color='skyblue')
plt.bar(x + width/2, scenario_df['time_ratio_con_over_dis'], width, label='Time Ratio (Con/Dis)', color='salmon')

plt.axhline(1, color='black', linestyle='-', linewidth=0.8) # Baseline 1:1 ratio
plt.xticks(x, scenario_df['label'], rotation=45, ha='right')
plt.ylabel('Ratio (Connected / Disconnected)')
plt.title('Relative Overhead of Connectivity Enforcement across Scenarios')
plt.legend()
plt.tight_layout()
plt.savefig('overhead_ratios.png')
plt.close()

# --- 4. PLOT: EFFICIENCY VS. THEORETICAL BOUND ---
plt.figure(figsize=(10, 6))
plt.scatter(scaling_df['n'], scaling_df['dis_queries_over_theory'], label='Disconnected Efficiency', s=100)
plt.scatter(scaling_df['n'], scaling_df['con_queries_over_theory'], label='Connected Efficiency', marker='s', s=100)

plt.xlabel('Number of Nodes ($n$)')
plt.ylabel('Empirical Queries / Theoretical Bound')
plt.yscale('log')
plt.title('Tightness of Theoretical Bounds')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('efficiency_analysis.png')
plt.close()
print("Plots saved successfully as PNG files.")