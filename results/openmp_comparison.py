import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caricamento e pulizia dati
df_guided = pd.read_csv("openmp/performance_guided.csv")
df_bound = pd.read_csv("openmp/performance.csv")

for df in [df_guided, df_bound]:
    df.columns = df.columns.str.strip()
    df['type'] = df['type'].str.strip()

# Classificazione nz
def classify_nz(df):
    df['group'] = df['nz'].apply(lambda x: '<1M' if x < 1_000_000 else '≥1M')
    return df

df_guided = classify_nz(df_guided)
df_bound = classify_nz(df_bound)

# Nuova funzione per confrontare scheduling guided vs bound per tipo (CSR o HLL)
def plot_sched_comparison(type_filter, output_filename):
    fig, axs = plt.subplots(2, 1, figsize=(20, 12), sharex=False)
    groups = ['<1M', '≥1M']
    titles = ['nz < 1M', 'nz ≥1M']
    palette = {'guided': '#1f77b4', 'bound': '#ff69b4'}

    for i, group in enumerate(groups):
        df1 = df_guided[(df_guided['type'] == type_filter) & (df_guided['group'] == group)][['matrix', 'avgGFlops']]
        df2 = df_bound[(df_bound['type'] == type_filter) & (df_bound['group'] == group)][['matrix', 'avgGFlops']]
        
        df1.rename(columns={'avgGFlops': 'guided'}, inplace=True)
        df2.rename(columns={'avgGFlops': 'bound'}, inplace=True)

        merged = pd.merge(df1, df2, on='matrix')
        if merged.empty:
            axs[i].text(0.5, 0.5, f"Nessun dato ({titles[i]})", ha='center', va='center')
            axs[i].axis('off')
            continue

        # Converte in formato "long" per seaborn
        melted = pd.melt(merged, id_vars='matrix', value_vars=['guided', 'bound'],
                         var_name='method', value_name='GFlops')

        sns.barplot(data=melted, x='matrix', y='GFlops', hue='method',
                    palette=palette, ax=axs[i])

        axs[i].set_title(f"{type_filter} - {titles[i]}")
        axs[i].set_ylabel("GFlops")
        axs[i].set_xlabel("")
        axs[i].tick_params(axis='x', rotation=45)

    axs[0].legend(title='Scheduling')
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Grafico salvato: {output_filename}")


# Esecuzione per i due tipi
plot_sched_comparison('HLL', 'openmp/graphs/hll_comparison.png')
plot_sched_comparison('CSR', 'openmp/graphs/csr_comparison.png')
