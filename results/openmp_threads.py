import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# Read the CSV file and clean the column names
df = pd.read_csv("openmp/performance_threads.csv")
df.columns = df.columns.str.strip()

# Clean the values in the 'type' column
df['type'] = df['type'].str.strip()

# Remove rows with NaN values in key columns
df = df.dropna(subset=['nThreads', 'avgGFlops', 'nz'])

# Define the threshold for nz
nz_threshold = 1_000_000

# Split the data into two groups based on nz
df_few = df[df['nz'] < nz_threshold]
df_many = df[df['nz'] >= nz_threshold]

def plot_type_performance(data, type, title, save_path):
    # Filter the data by the specified type (case insensitive)
    data_type = data[data['type'].str.upper() == type.upper()]
    
    if data_type.empty:
        print(f"No data for type {type} in group {title}")
        return

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data_type, x='nThreads', y='avgGFlops', hue='matrix', marker='o')
    plt.title(f"{title} - {type.upper()} Performance")
    plt.xlabel("Number of Threads")
    plt.ylabel("GFlops")
    plt.legend(title="Matrix", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the plot (this will overwrite the file if it exists)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()

# Save plots for matrices with few nz (<1e6)
plot_type_performance(df_few, 'CSR', "Gflops matrices with few nz (<1e6)", "openmp/graphs/few_nz_CSR.png")
plot_type_performance(df_few, 'HLL', "Gflops matrices with few nz (<1e6)", "openmp/graphs/few_nz_HLL.png")

# Save plots for matrices with many nz (>=1e6)
plot_type_performance(df_many, 'CSR', "Gflops matrices with many nz (>=1e6)", "openmp/graphs/many_nz_CSR.png")
plot_type_performance(df_many, 'HLL', "Gflops matrices with many nz (>=1e6)", "openmp/graphs/many_nz_HLL.png")
