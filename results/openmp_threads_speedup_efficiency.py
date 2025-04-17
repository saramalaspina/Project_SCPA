import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set seaborn theme for consistent plot styling
sns.set_theme(style="whitegrid")

# Load CSV file and clean whitespace from column and matrix names
df = pd.read_csv("openmp/speedup_threads.csv")
df.columns = df.columns.str.strip()
df['matrix'] = df['matrix'].str.strip()

# Threshold to distinguish between sparse and dense matrices (based on non-zeros)
nz_threshold = 1_000_000

# Function to compute efficiency from speedup and thread count
def add_efficiency_columns(df, algo_type):
    speedup_col = f"speedup_{algo_type.lower()}"
    eff_col = f"efficiency_{algo_type.lower()}"
    df[eff_col] = df[speedup_col] / df['nThreads']
    return df

# Add efficiency columns to the dataset
df = add_efficiency_columns(df, "CSR")
df = add_efficiency_columns(df, "HLL")

# Re-split dataset after adding new columns
df_few = df[df['nz'] < nz_threshold]
df_many = df[df['nz'] >= nz_threshold]

# Function to plot speedup
def plot_speedup(df_few, df_many, algo_type, save_path):
    """
    Plots speedup vs number of threads
    """
    if algo_type.upper() == "CSR":
        y_col = "speedup_csr"
    elif algo_type.upper() == "HLL":
        y_col = "speedup_hll"
    else:
        print("Invalid algorithm type. Please use 'CSR' or 'HLL'.")
        return

    # Create the figure with two subplots (few non-zeros and many non-zeros)
    fig, axs = plt.subplots(2, 1, figsize=(18, 12))

    # Plot for sparse matrices
    sns.lineplot(data=df_few, x='nThreads', y=y_col, hue='matrix', marker='o', ax=axs[0])
    axs[0].set_title(f"{algo_type.upper()} Threads Speedup - Matrices with Few Non-Zeros (< 1e6)")
    axs[0].set_xlabel("Number of Threads")
    axs[0].set_ylabel("Speedup")
    axs[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # Plot for dense matrices
    sns.lineplot(data=df_many, x='nThreads', y=y_col, hue='matrix', marker='o', ax=axs[1])
    axs[1].set_title(f"{algo_type.upper()} Threads Speedup - Matrices with Many Non-Zeros (≥ 1e6)")
    axs[1].set_xlabel("Number of Threads")
    axs[1].set_ylabel("Speedup")
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # Adjust layout and save to file
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Speedup plot saved to: {save_path}")
    plt.close()

# Function to plot efficiency
def plot_efficiency(df_few, df_many, algo_type, save_path):
    """
    Plots efficiency vs number of threads
    """
    eff_col = f"efficiency_{algo_type.lower()}"

    # Create the figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(18, 12))

    # Plot for sparse matrices
    sns.lineplot(data=df_few, x='nThreads', y=eff_col, hue='matrix', marker='o', ax=axs[0])
    axs[0].axhline(1.0, color='gray', linestyle='--', linewidth=1)  # Reference line at ideal efficiency
    axs[0].set_title(f"{algo_type.upper()} Threads Efficiency - Matrices with Few Non-Zeros (< 1e6)")
    axs[0].set_xlabel("Number of Threads")
    axs[0].set_ylabel("Efficiency")
    axs[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # Plot for dense matrices
    sns.lineplot(data=df_many, x='nThreads', y=eff_col, hue='matrix', marker='o', ax=axs[1])
    axs[1].axhline(1.0, color='gray', linestyle='--', linewidth=1)  # Reference line at ideal efficiency
    axs[1].set_title(f"{algo_type.upper()} Threads Efficiency - Matrices with Many Non-Zeros (≥ 1e6)")
    axs[1].set_xlabel("Number of Threads")
    axs[1].set_ylabel("Efficiency")
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # Adjust layout and save to file
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Efficiency plot saved to: {save_path}")
    plt.close()

# Create output directory if it doesn't exist
os.makedirs("openmp/graphs", exist_ok=True)

# Generate and save speedup plots
plot_speedup(df_few, df_many, "CSR", "openmp/graphs/csr_threads_speedup.png")
plot_speedup(df_few, df_many, "HLL", "openmp/graphs/hll_threads_speedup.png")

# Generate and save efficiency plots
plot_efficiency(df_few, df_many, "CSR", "openmp/graphs/csr_threads_efficiency.png")
plot_efficiency(df_few, df_many, "HLL", "openmp/graphs/hll_threads_efficiency.png")
