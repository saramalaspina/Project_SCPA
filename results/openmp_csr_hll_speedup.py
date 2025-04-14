import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("openmp/speedup.csv")
df.columns = df.columns.str.strip()
df['matrix'] = df['matrix'].str.strip()

# Melt the DataFrame to have a single column for speedup and a corresponding type
df_melt = pd.melt(
    df,
    id_vars=['matrix', 'nz', 'time_serial', 'nThreads'],
    value_vars=['speedup_csr', 'speedup_hll'],
    var_name='type',
    value_name='speedup'
)

# Remove the "speedup_" prefix and convert the type to uppercase
df_melt['type'] = df_melt['type'].str.replace('speedup_', '', regex=False).str.upper()

# Define the nz threshold and split the data
nz_threshold = 1_000_000
df_few = df_melt[df_melt['nz'] < nz_threshold]
df_many = df_melt[df_melt['nz'] >= nz_threshold]

# Create a single figure with two subplots (vertical)
fig, axes = plt.subplots(2, 1, figsize=(24, 16), sharex=False)

# Plot for matrices with few nz
sns.barplot(ax=axes[0], data=df_few, x='matrix', y='speedup', hue='type')
axes[0].set_title("OpenMP Speedup Comparison (CSR vs HLL) - Few nz (< 1e6)")
axes[0].set_xlabel("Matrix")
axes[0].set_ylabel("Speedup")
axes[0].legend(title="Type", loc='upper right')

# Plot for matrices with many nz
sns.barplot(ax=axes[1], data=df_many, x='matrix', y='speedup', hue='type')
axes[1].set_title("OpenMP Speedup Comparison (CSR vs HLL) - Many nz (>= 1e6)")
axes[1].set_xlabel("Matrix")
axes[1].set_ylabel("Speedup")
axes[1].legend(title="Type", loc='upper right')

# Adjust layout and save
plt.tight_layout()
plt.savefig("openmp/graphs/csr_hll_speedup.png", dpi=300, bbox_inches='tight')
print("Combined plot saved to: openmp/graphs/csr_hll_speedup.png")
plt.close()
