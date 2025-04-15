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
    # Set seaborn style
    sns.set_theme(style="whitegrid")

    fig, axs = plt.subplots(2, 1, figsize=(20, 12), sharex=False)
    groups = ['<1M', '≥1M']
    titles = ['Matrices with Few Non-Zeros (< 1e6)', 'Matrices with Many Non-Zeros (≥ 1e6)']
    palette = {'guided': '#ffa557', 'bound': '#1f78b4'}

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
        axs[i].set_title(f"{type_filter} OpenMP Scheduling Comparison - {titles[i]}")
        axs[i].grid(True, axis='y', linestyle='--', linewidth=0.7)
        axs[i].grid(False, axis='x')
        axs[i].set_ylabel("Execution Time (ms)")
        axs[i].set_xlabel("Matrix")
        axs[i].legend(loc='upper right')
        for label in axs[i].get_xticklabels():
            label.set_rotation(0)

    axs[0].legend(loc='upper left')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved to: {output_filename}")

# Generate charts
plot_time_comparison('HLL', 'openmp/graphs/hll_scheduling_totaltime.png')
plot_time_comparison('CSR', 'openmp/graphs/csr_scheduling_totaltime.png')

