import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

df = pd.read_csv("openmp/performance_threads.csv")
df.columns = df.columns.str.strip()
df['type'] = df['type'].str.strip()

# Define the nz threshold and split the data into two groups
nz_threshold = 1_000_000
df_few = df[df['nz'] < nz_threshold]
df_many = df[df['nz'] >= nz_threshold]

def plot_type_performance(data, type, title, save_path):
    # Filter the data by the specified type
    data_type = data[data['type'].str.upper() == type.upper()]
    
    if data_type.empty:
        print(f"No data for type {type} in group {title}")
        return

    plt.figure(figsize=(16, 8))
    sns.lineplot(data=data_type, x='nThreads', y='avgGFlops', hue='matrix', marker='o')
    plt.title(f"{title} - {type.upper()} Performance")
    plt.xlabel("Number of Threads")
    plt.ylabel("GFlops")
    plt.legend(title="Matrix", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()

plot_type_performance(df_few, 'CSR', "OpenMP Gflops matrices with few nz (<1e6)", "openmp/graphs/csr_few_nz.png")
plot_type_performance(df_few, 'HLL', "OpenMP Gflops matrices with few nz (<1e6)", "openmp/graphs/hll_few_nz.png")

plot_type_performance(df_many, 'CSR', "OpenMP Gflops matrices with many nz (>=1e6)", "openmp/graphs/csr_many_nz.png")
plot_type_performance(df_many, 'HLL', "OpenMP Gflops matrices with many nz (>=1e6)", "openmp/graphs/hll_many_nz.png")
