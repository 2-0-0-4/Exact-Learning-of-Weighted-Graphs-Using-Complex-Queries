import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
sns.set_theme(style="whitegrid")

# Load data
df = pd.read_csv('saved_results\\comparison_results_20260426_003344.csv')

# Split data into scaling and specific graph types
scaling_df = df[df['is_scaling_test'] == True].sort_values('nodes')
graph_types_df = df[df['is_scaling_test'] == False]

# --- Plot 1: Number of Nodes vs Query Complexity (Scaling) ---
plt.figure(figsize=(10, 6))
plt.plot(scaling_df['nodes'], scaling_df['lblr_queries'], marker='o', label='LBLR Queries', linewidth=2)
plt.plot(scaling_df['nodes'], scaling_df['exhaustive_queries'], marker='s', label='Exhaustive Queries', linewidth=2, linestyle='--')
plt.xlabel('Number of Nodes ($n$)')
plt.ylabel('Number of Queries')
plt.title('Query Complexity Scaling: LBLR vs Exhaustive')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig('scaling_query_complexity.png')
plt.close()

# --- Plot 2: Theoretical Bound vs Actual Queries (Scaling) ---
plt.figure(figsize=(10, 6))
plt.plot(scaling_df['nodes'], scaling_df['lblr_queries'], marker='o', label='Actual LBLR Queries', linewidth=2)
plt.plot(scaling_df['nodes'], scaling_df['lblr_theory_bound'], marker='x', label='Theoretical LBLR Bound', linewidth=2, linestyle=':')
plt.yscale('log')
plt.xlabel('Number of Nodes ($n$)')
plt.ylabel('Number of Queries (Log Scale)')
plt.title('LBLR Efficiency: Actual Queries vs Theoretical Bound')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig('theoretical_vs_actual.png')
plt.close()

# --- Plot 3: Speedup Across Different Graph Types ---
graph_types_df = graph_types_df.sort_values('speedup_x', ascending=False)
plt.figure(figsize=(12, 8))
# Removing hue to avoid legend issues and just use color
sns.barplot(data=graph_types_df, x='speedup_x', y='test_case', color='skyblue')
plt.axvline(x=1, color='red', linestyle='--', label='Break-even (Speedup=1)')
plt.xlabel('Speedup Factor (Exhaustive Queries / LBLR Queries)')
plt.ylabel('Graph Test Case')
plt.title('LBLR Query Speedup across Different Graph Structures')
plt.tight_layout()
plt.savefig('graph_speedup_comparison.png')
plt.close()

# --- Plot 4: Query Reduction Percentage across Scaling ---
plt.figure(figsize=(10, 6))
plt.plot(scaling_df['nodes'], scaling_df['query_reduction_pct'], marker='D', color='green', linewidth=2)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Number of Nodes ($n$)')
plt.ylabel('Query Reduction (%)')
plt.title('Efficiency Gain: Query Reduction (%) as $n$ Increases')
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('scaling_reduction_pct.png')
plt.close()

# --- Plot 5: Time Complexity Comparison (Scaling) ---
plt.figure(figsize=(10, 6))
plt.plot(scaling_df['nodes'], scaling_df['lblr_time_sec'], marker='o', label='LBLR Time', linewidth=2)
plt.plot(scaling_df['nodes'], scaling_df['exhaustive_time_sec'], marker='s', label='Exhaustive Time', linewidth=2, linestyle='--')
plt.xlabel('Number of Nodes ($n$)')
plt.ylabel('Execution Time (seconds)')
plt.title('Runtime Comparison: LBLR vs Exhaustive')
plt.legend()
plt.savefig('scaling_time_comparison.png')
plt.close()

print("Plots generated successfully.")