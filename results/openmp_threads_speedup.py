import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set seaborn theme
sns.set_theme(style="whitegrid")

# Load CSV and clean column/matrix names
df = pd.read_csv("openmp/speedup_threads.csv")
df.columns = df.columns.str.strip()
df['matrix'] = df['matrix'].str.strip()

# Define threshold and split dataset
nz_threshold = 1_000_000
df_few = df[df['nz'] < nz_threshold]
df_many = df[df['nz'] >= nz_threshold]

def plot_speedup(df_few, df_many, algo_type, save_path):
    # Choose the correct column based on algorithm type
    if algo_type.upper() == "CSR":
        y_col = "speedup_csr"
    elif algo_type.upper() == "HLL":
        y_col = "speedup_hll"
    else:
        print("Invalid algorithm type. Please use 'CSR' or 'HLL'.")
        return

    # Create the figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

    # First subplot: matrices with few non-zeros
    sns.lineplot(data=df_few, x='nThreads', y=y_col, hue='matrix', marker='o', ax=axs[0])
    axs[0].set_title(f"{algo_type.upper()} Threads Speedup - Matrices with Few Non-Zeros (< 1e6)")
    axs[0].set_ylabel("Speedup")
    axs[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # Second subplot: matrices with many non-zeros
    sns.lineplot(data=df_many, x='nThreads', y=y_col, hue='matrix', marker='o', ax=axs[1])
    axs[1].set_title(f"{algo_type.upper()} Threads Speedup - Matrices with Many Non-Zeros (≥ 1e6)")
    axs[1].set_xlabel("Number of Threads")
    axs[1].set_ylabel("Speedup")
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # Adjust layout and save to file
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


# Create output directory if it doesn't exist
os.makedirs("openmp/graphs", exist_ok=True)

# Generate and save plots
plot_speedup(df_few, df_many, "CSR", "openmp/graphs/csr_threads_speedup.png")
plot_speedup(df_few, df_many, "HLL", "openmp/graphs/hll_threads_speedup.png")
