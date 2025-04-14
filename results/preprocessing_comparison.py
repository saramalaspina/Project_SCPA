import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load and clean the data
df = pd.read_csv("openmp/preprocessing.csv")
df.columns = df.columns.str.strip()
df['type'] = df['type'].str.strip()

# Classify matrices by number of non-zeros
def classify_nz(df):
    df['group'] = df['nz'].apply(lambda x: '<1M' if x < 1_000_000 else '≥1M')
    return df

df = classify_nz(df)

# Function to plot comparison between guided and bound times
def plot_time_comparison(type_filter, output_filename):
    fig, axs = plt.subplots(2, 1, figsize=(20, 12), sharex=False)
    groups = ['<1M', '≥1M']
    titles = ['nz < 1M', 'nz ≥ 1M']
    palette = {'guided': '#1f77b4', 'bound': '#ff69b4'}

    for i, group in enumerate(groups):
        subset = df[(df['type'] == type_filter) & (df['group'] == group)]

        if subset.empty:
            axs[i].text(0.5, 0.5, f"No data ({titles[i]})", ha='center', va='center')
            axs[i].axis('off')
            continue

        # Compute total time for each method
        subset = subset.copy()
        subset['guided'] = subset['avgTimeGuided']
        subset['bound'] = subset['preTimeBound'] + subset['avgTimeBound']

        # Reshape to long format
        melted = pd.melt(subset, id_vars='matrix', value_vars=['guided', 'bound'], var_name='method', value_name='Time (ms)')

        sns.barplot(data=melted, x='matrix', y='Time (ms)', hue='method', palette=palette, ax=axs[i])
        
        axs[i].set_yscale('log')
        axs[i].set_title(f"{type_filter} - {titles[i]}")
        axs[i].set_ylabel("Execution Time (ms)")
        axs[i].set_xlabel("")
        axs[i].tick_params(axis='x', rotation=45)

    axs[0].legend(title='Method')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename)
    plt.close()
    print(f"Chart saved: {output_filename}")

# Generate charts
plot_time_comparison('HLL', 'openmp/graphs/hll_preprocessing_comparison.png')
plot_time_comparison('CSR', 'openmp/graphs/csr_preprocessing_comparison.png')

