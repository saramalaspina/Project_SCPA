import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Seaborn theme for consistent and clean visuals
sns.set_theme(style="whitegrid")

# Load the two CSV files
df_base = pd.read_csv("cuda/speedup.csv")
df_warp = pd.read_csv("cuda/speedup_warp.csv")

# Clean up column names (remove leading/trailing spaces)
df_base.columns = df_base.columns.str.strip()
df_warp.columns = df_warp.columns.str.strip()

# Add a 'mode' column to distinguish between normal and warp implementations
df_base['mode'] = 'normal'
df_warp['mode'] = 'warp'

# Combine both datasets into one
df = pd.concat([df_base, df_warp], ignore_index=True)

# Clean and normalize text columns
df['matrix'] = df['matrix'].str.strip()
df['mode'] = df['mode'].str.strip()

# Define threshold to categorize matrices based on number of non-zeros
nz_threshold = 1_000_000
df['nz_category'] = df['nz'].apply(lambda x: 'few' if x < nz_threshold else 'many')

# Create output directory for plots
os.makedirs("graphs", exist_ok=True)

# Function to filter data and generate a plot for one algorithm + mode
def plot_single_speedup(df, algo_col, mode_val, save_path, algo_label):
    # Filter data by selected mode (normal or warp)
    data = df[df['mode'] == mode_val].copy()

    # Choose a label for the plot based on algorithm
    if algo_col == 'speedup_csr':
        title_algo = 'CSR'
    else:
        title_algo = 'HLL'

    # Rename selected speedup column for easier plotting
    data['speedup'] = data[algo_col]

    # Split data into two subsets based on nz_category
    data_few = data[data['nz_category'] == 'few']
    data_many = data[data['nz_category'] == 'many']

    # Create a figure with 2 subplots (for 'few' and 'many')
    fig, axs = plt.subplots(2, 1, figsize=(20, 12))

    # Helper function to plot each subset
    def plot_subset(ax, subset, subtitle):
        if subset.empty:
            ax.set_visible(False)
            return
        sns.lineplot(
            data=subset,
            x='nThreads',
            y='speedup',
            hue='matrix',
            marker='o',
            ax=ax
        )
        ax.set_xticks([128, 256, 512, 1024])
        ax.set_title(f"CUDA {algo_label} Threads per Block Speedup — {subtitle}", fontsize=16)
        ax.set_xlabel("Threads Per Block")
        ax.set_ylabel("Speedup")
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=14, title_fontsize=16)
        ax.grid(True)

    # Plot both subsets: few and many non-zeros
    plot_subset(axs[0], data_few, "Matrices with Few Non-Zeros (< 1e6)")
    plot_subset(axs[1], data_many, "Matrices with Many Non-Zeros (≥ 1e6)")

    # Adjust layout, save figure, and close
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

# Generate four plots for each combination of algorithm and mode
plot_single_speedup(df, 'speedup_csr', 'normal', "cuda/graphs/csr_block.png", "CSR")
plot_single_speedup(df, 'speedup_csr', 'warp', "cuda/graphs/csr_warp_block.png", "CSR (warp)")
plot_single_speedup(df, 'speedup_hll', 'normal', "cuda/graphs/hll_block.png", "HLL")
plot_single_speedup(df, 'speedup_hll', 'warp', "cuda/graphs/hll_warp_block.png", "HLL (warp)")
