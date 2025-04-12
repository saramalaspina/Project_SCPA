import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

df = pd.read_csv("openmp/speedup_threads.csv")
df.columns = df.columns.str.strip()
df['matrix'] = df['matrix'].str.strip()

# Define the nz threshold and split the data into two groups
nz_threshold = 1_000_000
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

    plt.figure(figsize=(16, 8))
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


plot_type_speedup(df_few, "CSR", "OpenMP Speedup matrices with few nz (<1e6) ", "openmp/graphs/speedup_csr_few_nz.png")
plot_type_speedup(df_few, "HLL", "OpenMP Speedup matrices with few nz (<1e6)", "openmp/graphs/speedup_hll_few_nz.png")

plot_type_speedup(df_many, "CSR", "OpenMP Speedup matrices with many nz (>=1e6)", "openmp/graphs/speedup_csr_many_nz.png")
plot_type_speedup(df_many, "HLL", "OpenMP Speedup matrices with many nz (>=1e6)", "openmp/graphs/speedup_hll_many_nz.png")
