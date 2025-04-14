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

    palette = {'CSR': '#1f77b4', 'HLL': '#ff69b4'}

    fig, axes = plt.subplots(2, 1, figsize=(24, 14), sharey=False)

    # Few nz
    sns.barplot(data=data_few, x='matrix', y='speedup', hue='type', ax=axes[0], palette=palette)
    axes[0].set_title("Cuda Speedup Comparison (CSR vs HLL) for Matrices with Few nz (<1e6)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Speedup")
    axes[0].legend(title="Type", bbox_to_anchor=(1.01, 1), loc='upper left')

    # Many nz
    sns.barplot(data=data_many, x='matrix', y='speedup', hue='type', ax=axes[1], palette=palette)
    axes[1].set_title("Cuda Speedup Comparison (CSR vs HLL) for Matrices with Many nz (≥1e6)")
    axes[1].set_xlabel("Matrix")
    axes[1].set_ylabel("Speedup")
    axes[1].legend(title="Type", bbox_to_anchor=(1.01, 1), loc='upper left')

    # Layout e salvataggio
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to: {save_path}")
    plt.close()

plot_speedup("cuda/graphs/csr_hll_speedup.png", "cuda/speedup.csv")
plot_speedup("cuda/graphs/csr_hll_speedup_warp.png", "cuda/speedup_warp.csv")
