import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot speedup comparison
def plot_speedup_comparison(df, nz_threshold, save_path):

    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # Melt the DataFrame to have a single column for speedup and a corresponding type
    df_melt = pd.melt(
        df,
        id_vars=['matrix', 'nz', 'time_serial', 'nThreads'],
        value_vars=['speedup_csr', 'speedup_hll'],
        var_name='type',
        value_name='speedup'
    )

    # Clean the 'type' column (remove 'speedup_' prefix and make it uppercase)
    df_melt['type'] = df_melt['type'].str.replace('speedup_', '', regex=False).str.upper()

    # Split data into two groups based on the nz threshold
    df_few = df_melt[df_melt['nz'] < nz_threshold]
    df_many = df_melt[df_melt['nz'] >= nz_threshold]

    # Create a single figure with two subplots (vertical)
    fig, axes = plt.subplots(2, 1, figsize=(24, 16), sharex=False)

    # Define custom color palette with high contrast (yellow and blue)
    custom_palette = {
        "CSR": "#38bdf8",  # bright light blue
        "HLL": "#f9a8d4",  # light pink (rosa chiaro)
    }

    # Plot for matrices with few nz
    sns.barplot(ax=axes[0], data=df_few, x='matrix', y='speedup', hue='type', palette=custom_palette)
    axes[0].set_title("OpenMP CSR vs HLL Speedup Comparison - Matrices with Few Non-Zeros (< 1e6)", fontsize = 20)
    axes[0].set_xlabel("Matrix")
    axes[0].set_ylabel("Speedup")
    axes[0].legend(loc='upper right', fontsize=16, title_fontsize=18)
    axes[0].grid(True, axis='y', linestyle='--', linewidth=0.7)  # Add horizontal grid lines
    for label in axes[0].get_xticklabels():
        label.set_rotation(0)
        label.set_fontsize(14)

    # Plot for matrices with many nz
    sns.barplot(ax=axes[1], data=df_many, x='matrix', y='speedup', hue='type', palette=custom_palette)
    axes[1].set_title("OpenMP CSR vs HLL Speedup Comparison - Matrices with Many Non-Zeros (≥ 1e6)", fontsize = 20)
    axes[1].set_xlabel("Matrix")
    axes[1].set_ylabel("Speedup")
    axes[1].legend(loc='upper right', fontsize=16, title_fontsize=18)
    axes[1].grid(True, axis='y', linestyle='--', linewidth=0.7)  # Add horizontal grid lines
    for label in axes[1].get_xticklabels():
        label.set_rotation(0)
        label.set_fontsize(14)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()

# Read data
df = pd.read_csv("openmp/speedup.csv")
df.columns = df.columns.str.strip()
df['matrix'] = df['matrix'].str.strip()

# Define the nz threshold and save path
nz_threshold = 1_000_000
save_path = "openmp/graphs/csr_hll_speedup.png"

# Call the function to create and save the plot
plot_speedup_comparison(df, nz_threshold, save_path)
