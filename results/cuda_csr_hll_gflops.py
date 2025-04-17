import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_bar_comparison(save_path, csv_path, nz_threshold=1_000_000):
    sns.set_theme(style="whitegrid")

    # Caricamento dati
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df['type'] = df['type'].str.strip()

    # Filtra solo i tipi paralleli (CSR e HLL)
    df_parallel = df[df['type'].str.upper().isin(['CSR', 'HLL'])]

    # Soglia nz per separare i dataset
    df_few = df_parallel[df_parallel['nz'] < nz_threshold]
    df_many = df_parallel[df_parallel['nz'] >= nz_threshold]
    
    palette = { "CSR": "#38bdf8", "HLL": "#f9a8d4"}

    # Crea la figura con due subplot
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(24, 14))

    # Primo grafico: pochi nz
    sns.barplot(ax=axes[0], data=df_few, x='matrix', y='avgGFlops', hue='type', palette=palette)
    axes[0].set_title("Cuda AvgGFlops Comparison (CSR vs HLL) - Matrices with Few Non-Zeros (< 1e6)")
    axes[0].grid(True, axis='y', linestyle='--', linewidth=0.7)
    axes[0].grid(False, axis='x')
    axes[0].set_xlabel("Matrix")
    axes[0].set_ylabel("AvgGFlops")
    axes[0].legend(loc='upper right')
    for label in axes[0].get_xticklabels():
        label.set_rotation(0)

    # Secondo grafico: molti nz
    sns.barplot(ax=axes[1], data=df_many, x='matrix', y='avgGFlops', hue='type', palette=palette)
    axes[1].set_title("Cuda AvgGFlops Comparison (CSR vs HLL) - Matrices with Many Non-Zeros (≥ 1e6)")
    axes[1].grid(True, axis='y', linestyle='--', linewidth=0.7)
    axes[1].grid(False, axis='x')
    axes[1].set_xlabel("Matrix")
    axes[1].set_ylabel("AvgGFlops")
    axes[1].legend(loc='upper right')
    for label in axes[1].get_xticklabels():
        label.set_rotation(0)

    plt.tight_layout()

    # Salva la figura unica
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()

plot_bar_comparison("cuda/graphs/csr_hll_gflops.png", "cuda/best_performance.csv")
