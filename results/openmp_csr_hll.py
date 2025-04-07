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

def plot_bar_comparison(data, title, save_path):
    plt.figure(figsize=(20, 10))
    sns.barplot(data=data, x='matrix', y='avgGFlops', hue='type')
    plt.title(title)
    plt.xlabel("Matrix")
    plt.ylabel("AvgGFlops")
    plt.legend(title="Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()

plot_bar_comparison(df_few, "OpenMP AvgGFlops Comparison (CSR vs HLL) for Matrices with Few nz (<1e6)", "openmp/graphs/csr_hll_few_nz.png")
plot_bar_comparison(df_many, "OpenMP AvgGFlops Comparison (CSR vs HLL) for Matrices with Many nz (>=1e6)", "openmp/graphs/csr_hll_many_nz.png")