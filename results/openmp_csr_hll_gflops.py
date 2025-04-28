import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load and clean data
df = pd.read_csv("openmp/performance.csv")
df.columns = df.columns.str.strip()
df['type'] = df['type'].str.strip()

# Filter to keep only parallel implementations
df_parallel = df[df['type'].str.upper().isin(['CSR', 'HLL'])]

# Define threshold and split into groups
nz_threshold = 1_000_000
df_few = df_parallel[df_parallel['nz'] < nz_threshold]
df_many = df_parallel[df_parallel['nz'] >= nz_threshold]

def plot_bar_gflops_comparison(df_few, df_many, save_path):
    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(24, 16), sharex=False)

    # Define custom color palette with high contrast (yellow and blue)
    custom_palette = {
        "CSR": "#38bdf8",  # bright light blue
        "HLL": "#f9a8d4",  # light pink (rosa chiaro)
    }
    # Few nz
    sns.barplot(ax=axes[0], data=df_few, x='matrix', y='avgGFlops', hue='type', palette=custom_palette)
    axes[0].set_title("OpenMP CSR vs HLL GigaFlops Comparison - Matrices with Few Non-Zeros (< 1e6)", fontsize = 20)
    axes[0].set_xlabel("Matrix")
    axes[0].set_ylabel("GigaFlops")
    axes[0].legend(loc='upper right', fontsize=16, title_fontsize=18)
    axes[0].grid(True, axis='y', linestyle='--', linewidth=0.7)
    axes[0].grid(False, axis='x')
    for label in axes[0].get_xticklabels():
        label.set_rotation(0)
        label.set_fontsize(14)

    # Many nz
    sns.barplot(ax=axes[1], data=df_many, x='matrix', y='avgGFlops', hue='type', palette=custom_palette)
    axes[1].set_title("OpenMP CSR vs HLL GigaFlops Comparison - Matrices with Many Non-Zeros (≥ 1e6)", fontsize = 20)
    axes[1].set_xlabel("Matrix")
    axes[1].set_ylabel("GigaFlops")
    axes[1].legend(loc='upper right', fontsize=16, title_fontsize=18)
    axes[1].grid(True, axis='y', linestyle='--', linewidth=0.7)
    axes[1].grid(False, axis='x')
    for label in axes[1].get_xticklabels():
        label.set_rotation(0)
        label.set_fontsize(14)

    # Save plot
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()

# Call the function
plot_bar_gflops_comparison(df_few, df_many, "openmp/graphs/csr_hll_gflops.png")
