import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Nuova funzione combinata
def plot_speedup(save_path, filename):
    sns.set_theme(style="whitegrid")

    # Caricamento e pulizia dati
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    df['matrix'] = df['matrix'].str.strip()

    # Melt del DataFrame per una singola colonna di speedup
    df_melt = pd.melt(df, id_vars=['matrix', 'nz', 'time_serial'], value_vars=['speedup_csr', 'speedup_hll'], var_name='type', value_name='speedup')

    # Pulizia della colonna 'type'
    df_melt['type'] = df_melt['type'].str.replace('speedup_', '', regex=False).str.upper()

    # Soglia nz
    nz_threshold = 1_000_000
    data_few = df_melt[df_melt['nz'] < nz_threshold]
    data_many = df_melt[df_melt['nz'] >= nz_threshold]

    palette = { "CSR": "#38bdf8", "HLL": "#f9a8d4"}

    fig, axes = plt.subplots(2, 1, figsize=(24, 14), sharey=False)

    # Few nz
    sns.barplot(data=data_few, x='matrix', y='speedup', hue='type', ax=axes[0], palette=palette)
    axes[0].set_title("Cuda Speedup Comparison (CSR vs HLL) - Matrices with Few Non-Zeros (< 1e6)")
    axes[0].grid(True, axis='y', linestyle='--', linewidth=0.7)
    axes[0].grid(False, axis='x')
    axes[0].set_xlabel("Matrix")
    axes[0].set_ylabel("Speedup")
    axes[0].legend(loc='upper right')
    for label in axes[0].get_xticklabels():
        label.set_rotation(0)

    # Many nz
    sns.barplot(data=data_many, x='matrix', y='speedup', hue='type', ax=axes[1], palette=palette)
    axes[1].set_title("Cuda Speedup Comparison (CSR vs HLL) - Matrices with Many Non-Zeros (≥ 1e6)")
    axes[1].grid(True, axis='y', linestyle='--', linewidth=0.7)
    axes[1].grid(False, axis='x')
    axes[1].set_xlabel("Matrix")
    axes[1].set_ylabel("Speedup")
    axes[1].legend(loc='upper right')
    for label in axes[1].get_xticklabels():
        label.set_rotation(0)

    # Layout e salvataggio
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()

plot_speedup("cuda/graphs/csr_hll_speedup.png", "cuda/best_speedup.csv")

