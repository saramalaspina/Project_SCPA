import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_no_warp = pd.read_csv("cuda/performance.csv")
df_warp = pd.read_csv("cuda/performance_warp.csv")

df_no_warp.columns = df_no_warp.columns.str.strip()
df_no_warp['type'] = df_no_warp['type'].str.strip()

df_warp.columns = df_warp.columns.str.strip()
df_warp['type'] = df_warp['type'].str.strip()

# Aggiungi colonna di classificazione
def classify_nz(df):
    df['group'] = df['nz'].apply(lambda x: '<1M' if x < 1_000_000 else '≥1M')
    return df

df_no_warp = classify_nz(df_no_warp)
df_warp = classify_nz(df_warp)

def plot_combined(type_filter, output_filename, threads_per_block=None):
    sns.set_theme(style="whitegrid")

    palette = {
        "warp": "#a7f3d0",  # light green
        "no_warp": "#c084fc",  # vibrant lilac
    }

    fig, axs = plt.subplots(2, 1, figsize=(20, 12), sharex=False)
    groups = ['<1M', '≥1M']
    titles = ['Matrices with Few Non-Zeros (< 1e6)', 'Matrices with Many Non-Zeros (≥ 1e6)']

    for i, group in enumerate(groups):
        df1 = df_no_warp[(df_no_warp['type'] == type_filter) & (df_no_warp['group'] == group)]
        df2 = df_warp[(df_warp['type'] == type_filter) & (df_warp['group'] == group)]

        # Filtro opzionale su threadsPerBlock
        if threads_per_block is not None:
            df1 = df1[df1['threadsPerBlock'] == threads_per_block]
            df2 = df2[df2['threadsPerBlock'] == threads_per_block]

        df1 = df1[['matrix', 'avgGFlops']].rename(columns={'avgGFlops': 'no_warp'})
        df2 = df2[['matrix', 'avgGFlops']].rename(columns={'avgGFlops': 'warp'})

        merged = pd.merge(df1, df2, on='matrix')
        if merged.empty:
            axs[i].text(0.5, 0.5, f"Nessun dato ({titles[i]})", ha='center', va='center')
            axs[i].axis('off')
            continue

        melted = pd.melt(merged, id_vars='matrix', value_vars=['no_warp', 'warp'], var_name='method', value_name='GFlops')
        sns.barplot(data=melted, x='matrix', y='GFlops', hue='method', palette=palette, ax=axs[i])
        axs[i].set_title(f"{type_filter} CUDA Scheduling Comparison - {titles[i]} (TPB={threads_per_block})")
        axs[i].grid(True, axis='y', linestyle='--', linewidth=0.7)
        axs[i].grid(False, axis='x')
        axs[i].set_ylabel("GFlops")
        axs[i].set_xlabel("Matrix")
        axs[i].legend(loc='upper right')
        axs[i].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved to: {output_filename}")


# Genera i due grafici
plot_combined('HLL', 'cuda/graphs/hll_scheduling.png', threads_per_block=128)
plot_combined('CSR', 'cuda/graphs/csr_scheduling.png', threads_per_block=128)
