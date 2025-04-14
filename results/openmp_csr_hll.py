import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("openmp/performance.csv")
df.columns = df.columns.str.strip()
df['type'] = df['type'].str.strip()

# Filter to keep only the parallel types (CSR and HLL)
df_parallel = df[df['type'].str.upper().isin(['CSR', 'HLL'])]

# Define the nz threshold and split the data into two groups
nz_threshold = 1_000_000
df_few = df_parallel[df_parallel['nz'] < nz_threshold]
df_many = df_parallel[df_parallel['nz'] >= nz_threshold]

# Create a single figure with two subplots (stacked vertically)
fig, axes = plt.subplots(2, 1, figsize=(24, 16), sharex=False)

# Plot for matrices with few nz
sns.barplot(ax=axes[0], data=df_few, x='matrix', y='avgGFlops', hue='type')
axes[0].set_title("OpenMP AvgGFlops Comparison (CSR vs HLL) - Few nz (< 1e6)")
axes[0].set_xlabel("Matrix")
axes[0].set_ylabel("AvgGFlops")
axes[0].legend(title="Type", loc='upper right')

# Plot for matrices with many nz
sns.barplot(ax=axes[1], data=df_many, x='matrix', y='avgGFlops', hue='type')
axes[1].set_title("OpenMP AvgGFlops Comparison (CSR vs HLL) - Many nz (>= 1e6)")
axes[1].set_xlabel("Matrix")
axes[1].set_ylabel("AvgGFlops")
axes[1].legend(title="Type", loc='upper right')

# Adjust layout and save
plt.tight_layout()
plt.savefig("openmp/graphs/csr_hll_gflops.png", dpi=300, bbox_inches='tight')
print("Combined plot saved to: openmp/graphs/csr_hll_gflops.png")
plt.close()
