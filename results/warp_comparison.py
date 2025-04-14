import pandas as pd
import matplotlib.pyplot as plt

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

# Funzione per creare subplot combinati
def plot_combined(type_filter, output_filename):
    fig, axs = plt.subplots(2, 1, figsize=(20, 12), sharex=False)
    groups = ['<1M', '≥1M']
    titles = ['nz < 1M', 'nz ≥ 1M']

    for i, group in enumerate(groups):
        df1 = df_no_warp[(df_no_warp['type'] == type_filter) & (df_no_warp['group'] == group)][['matrix', 'avgGFlops']]
        df2 = df_warp[(df_warp['type'] == type_filter) & (df_warp['group'] == group)][['matrix', 'avgGFlops']]
        
        df1.rename(columns={'avgGFlops': 'no_warp'}, inplace=True)
        df2.rename(columns={'avgGFlops': 'warp'}, inplace=True)
        
        merged = pd.merge(df1, df2, on='matrix')
        if merged.empty:
            axs[i].text(0.5, 0.5, f"Nessun dato ({titles[i]})", ha='center', va='center')
            axs[i].axis('off')
            continue

        x = merged['matrix']
        bar_width = 0.35
        index = range(len(x))

        axs[i].bar(index, merged['no_warp'], bar_width, label='No Warp')
        axs[i].bar([j + bar_width for j in index], merged['warp'], bar_width, label='Warp', alpha=0.7)
        axs[i].set_title(f"{type_filter} - {titles[i]}")
        axs[i].set_ylabel("GFlops")
        axs[i].set_xticks([j + bar_width / 2 for j in index])
        axs[i].set_xticklabels(x, rotation=45)

    axs[0].legend()
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Grafico salvato: {output_filename}")

# Genera i due grafici
plot_combined('HLL', 'cuda/graphs/hll_warp_comparison.png')
plot_combined('CSR', 'cuda/graphs/csr_warp_comparison.png')
