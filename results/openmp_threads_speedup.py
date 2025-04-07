import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# Read the CSV file
df = pd.read_csv("openmp/speedup_threads.csv")
df.columns = df.columns.str.strip()  # Clean column names

# Clean values in the 'matrix' column (if needed)
df['matrix'] = df['matrix'].str.strip()

# Remove rows with missing values in key columns
df = df.dropna(subset=['nThreads', 'speedup_csr', 'speedup_hll'])

# Define the threshold for nz
nz_threshold = 1_000_000

# Split the data into two groups based on nz
df_few = df[df['nz'] < nz_threshold]
df_many = df[df['nz'] >= nz_threshold]

def plot_type_speedup(data, type, title, save_path):
    # Choose the correct column based on the algorithm type
    if type.upper() == "CSR":
        y_col = "speedup_csr"
    elif type.upper() == "HLL":
        y_col = "speedup_hll"
    else:
        print("Invalid algorithm type. Please use 'CSR' or 'HLL'.")
        return

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x='nThreads', y=y_col, hue='matrix', marker='o')
    plt.title(f"{title} - {type.upper()} Speedup")
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.legend(title="Matrix", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot (this will overwrite the file if it exists)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


# Plot and save speedup for CSR few nz
plot_type_speedup(df_few, "CSR", "Speedup matrices with few nz (<1e6) ", "openmp/graphs/few_nz_speedup_CSR.png")
# Plot and save speedup for HLL few nz
plot_type_speedup(df_few, "HLL", "Speedup matrices with few nz (<1e6)", "openmp/graphs/few_nz_speedup_HLL.png")

# Plot and save speedup for CSR many nz
plot_type_speedup(df_many, "CSR", "Speedup matrices with many nz (>=1e6)", "openmp/graphs/many_nz_speedup_CSR.png")
# Plot and save speedup for HLL many nz
plot_type_speedup(df_many, "HLL", "Speedup matrices with many nz (>=1e6)", "openmp/graphs/many_nz_speedup_HLL.png")
