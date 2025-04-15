import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set seaborn theme
sns.set_theme(style="whitegrid")

# Load CSV and clean column/type names
df = pd.read_csv("openmp/performance_threads.csv")
df.columns = df.columns.str.strip()
df['type'] = df['type'].str.strip()

# Define threshold and split dataset
nz_threshold = 1_000_000
df_few = df[df['nz'] < nz_threshold]
df_many = df[df['nz'] >= nz_threshold]

def plot_performance(df_few, df_many, algo_type, save_path):
    # Filter data for the given algorithm type
    data_few = df_few[df_few['type'].str.upper() == algo_type.upper()]
    data_many = df_many[df_many['type'].str.upper() == algo_type.upper()]

    if data_few.empty and data_many.empty:
        print(f"No data available for type {algo_type}")
        return

    # Create the figure with two subplots (vertically stacked)
    fig, axs = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

    # Top subplot: matrices with few non-zeros
    if not data_few.empty:
        sns.lineplot(data=data_few, x='nThreads', y='avgGFlops', hue='matrix', marker='o', ax=axs[0])
        axs[0].set_title(f"{algo_type.upper()} Threads Performance - Matrices with Few Non-Zeros (< 1e6)")
        axs[0].set_xlabel("Number of Threads")
        axs[0].set_ylabel("GFlops")
        axs[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    else:
        axs[0].set_visible(False)

    # Bottom subplot: matrices with many non-zeros
    if not data_many.empty:
        sns.lineplot(data=data_many, x='nThreads', y='avgGFlops', hue='matrix', marker='o', ax=axs[1])
        axs[1].set_title(f"{algo_type.upper()} Threads Performance - Matrices with Many Non-Zeros (≥ 1e6)")
        axs[1].set_xlabel("Number of Threads")
        axs[1].set_ylabel("GFlops")
        axs[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    else:
        axs[1].set_visible(False)

    # Adjust layout and save to file
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


# Ensure the output directory exists
os.makedirs("openmp/graphs", exist_ok=True)

# Generate combined plots
plot_performance(df_few, df_many, "CSR", "openmp/graphs/csr_threads_gflops.png")
plot_performance(df_few, df_many, "HLL", "openmp/graphs/hll_threads_gflops.png")
