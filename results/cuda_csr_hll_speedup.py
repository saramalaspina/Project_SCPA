import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Read the CSV file and clean the column names
df = pd.read_csv("cuda/speedup.csv")
df.columns = df.columns.str.strip()

# Clean the 'matrix' column (if necessary)
df['matrix'] = df['matrix'].str.strip()

# Melt the DataFrame to have a single column for speedup and a corresponding type
df_melt = pd.melt(df,
                  id_vars=['matrix', 'nz', 'time_serial'],
                  value_vars=['speedup_csr', 'speedup_hll'],
                  var_name='type',
                  value_name='speedup')

# Remove the "speedup_" prefix and convert the type to uppercase
df_melt['type'] = df_melt['type'].str.replace('speedup_', '', regex=False).str.upper()

# Define the nz threshold and split the data into two groups
nz_threshold = 1_000_000
df_few = df_melt[df_melt['nz'] < nz_threshold]
df_many = df_melt[df_melt['nz'] >= nz_threshold]

def plot_bar_speedup(data, title, save_path):
    plt.figure(figsize=(20, 10))
    sns.barplot(data=data, x='matrix', y='speedup', hue='type')
    plt.title(title)
    plt.xlabel("Matrix")
    plt.ylabel("Speedup")
    plt.legend(title="Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot (this overwrites the file if it exists)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()

# Plot and save for matrices with fewer than 1e6 nz
plot_bar_speedup(df_few,
                 "Speedup Comparison (CSR vs HLL) for Matrices with Few nz (<1e6)",
                 "cuda/graphs/csr_hll_few_nz_speedup.png")

# Plot and save for matrices with 1e6 or more nz
plot_bar_speedup(df_many,
                 "Speedup Comparison (CSR vs HLL) for Matrices with Many nz (>=1e6)",
                 "cuda/graphs/csr_hll_many_nz_speedup.png")

