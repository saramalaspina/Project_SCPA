import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_bar_comparison(save_path, csv_path, nz_threshold=1_000_000):
    sns.set_theme(style="whitegrid")

    # Load and clean data
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df['type'] = df['type'].str.strip()

    # Filter to keep only parallel implementations
    df_parallel = df[df['type'].str.upper().isin(['CSR', 'HLL'])]

    # Define threshold and split into groups
    df_few = df_parallel[df_parallel['nz'] < nz_threshold]
    df_many = df_parallel[df_parallel['nz'] >= nz_threshold]
    
    palette = { "CSR": "#38bdf8", "HLL": "#f9a8d4"}

    # Create subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(24, 14))

    # Few nz
    sns.barplot(ax=axes[0], data=df_few, x='matrix', y='avgGFlops', hue='type', palette=palette)
    axes[0].set_title("Cuda CSR vs HLL GigaFlops Comparison - Matrices with Few Non-Zeros (< 1e6)", fontsize = 20)
    axes[0].grid(True, axis='y', linestyle='--', linewidth=0.7)
    axes[0].grid(False, axis='x')
    axes[0].set_xlabel("Matrix")
    axes[0].set_ylabel("GigaFlops")
    axes[0].legend(loc='upper right', fontsize=16, title_fontsize=18)
    for label in axes[0].get_xticklabels():
        label.set_rotation(0)
        label.set_fontsize(14)

    # Many nz
    sns.barplot(ax=axes[1], data=df_many, x='matrix', y='avgGFlops', hue='type', palette=palette)
    axes[1].set_title("Cuda CSR vs HLL GigaFlops Comparison - Matrices with Many Non-Zeros (≥ 1e6)", fontsize = 20)
    axes[1].grid(True, axis='y', linestyle='--', linewidth=0.7)
    axes[1].grid(False, axis='x')
    axes[1].set_xlabel("Matrix")
    axes[1].set_ylabel("GigaFlops")
    axes[1].legend(loc='upper right', fontsize=16, title_fontsize=18)
    for label in axes[1].get_xticklabels():
        label.set_rotation(0)
        label.set_fontsize(14)

    plt.tight_layout()

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()

plot_bar_comparison("cuda/graphs/csr_hll_gflops.png", "cuda/best_performance.csv")
