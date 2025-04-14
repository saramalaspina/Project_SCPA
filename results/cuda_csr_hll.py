import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_bar_comparison(csv_path, save_path, nz_threshold=1_000_000):
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

    palette = {'CSR': '#1f77b4', 'HLL': '#ff69b4'}

    # Crea la figura con due subplot
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(24, 14))

    # Primo grafico: pochi nz
    sns.barplot(ax=axes[0], data=df_few, x='matrix', y='avgGFlops', hue='type', palette=palette)
    axes[0].set_title("Cuda AvgGFlops Comparison (CSR vs HLL) - Few nz (<1e6)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("AvgGFlops")
    axes[0].legend(title="Type", bbox_to_anchor=(1.01, 1), loc='upper left')
    axes[0].tick_params(axis='x', rotation=90)

    # Secondo grafico: molti nz
    sns.barplot(ax=axes[1], data=df_many, x='matrix', y='avgGFlops', hue='type', palette=palette)
    axes[1].set_title("Cuda AvgGFlops Comparison (CSR vs HLL) - Many nz (>=1e6)")
    axes[1].set_xlabel("Matrix")
    axes[1].set_ylabel("AvgGFlops")
    axes[1].legend(title="Type", bbox_to_anchor=(1.01, 1), loc='upper left')
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()

    # Salva la figura unica
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to: {save_path}")
    plt.close()

plot_bar_comparison("cuda/performance.csv", "cuda/graphs/csr_hll.png")
plot_bar_comparison("cuda/performance_warp.csv", "cuda/graphs/csr_hll_warp.png")