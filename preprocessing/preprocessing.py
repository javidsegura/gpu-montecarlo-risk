import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

def monte_carlo_crash_simulation(x, N, k, mu, Sigma, M):
    """
    Pure Monte Carlo simulation following Cate's pseudocode exactly.
    """
    L = np.linalg.cholesky(Sigma)
    
    print(f"Starting Monte Carlo simulation with M = {M:,} trials...")
    print(f"Parameters: N={N}, k={k}, x={x}%")
    print("-" * 60)
    
    count = 0 
    S_values = np.zeros(M) 
    
    for j in range(M):
        Z = np.random.standard_normal(N)
        R = mu + L @ Z
        S = np.sum(R < -x)
        S_values[j] = S
        
        if S >= k:
            count += 1 
        
        if (j + 1) % (M // 10) == 0:
            print(f"Progress: {(j+1)/M*100:.0f}% - Current estimate: {count/(j+1):.6f}")
    
    P_hat = count / M
    
    print("-" * 60)
    print(f"Simulation complete.")
    
    return P_hat, count, S_values

def compute_accuracy(P_hat, M):
    """
    Compute accuracy and confidence interval.
    """
    std_error = np.sqrt(P_hat * (1 - P_hat) / M)
    z_score = 1.96
    margin = z_score * std_error
    ci_lower = max(0, P_hat - margin)
    ci_upper = min(1, P_hat + margin)
    
    return std_error, ci_lower, ci_upper


# LOAD & PROCESS MULTIPLE .csv FILES
csv_files = [
    "PreprocessingFile1.csv",
    "PreprocessingFile2.csv",
    "PreprocessingFile3.csv",
    "PreprocessingFile4.csv",
    "PreprocessingFile5.csv",
    "PreprocessingFile6.csv",
    "PreprocessingFile7.csv",
    "PreprocessingFile8.csv",
    "PreprocessingFile9.csv",
    "PreprocessingFile10.csv"
]

# Dictionary to store the DataFrames
dataframes = {}
all_returns = []

print("=" * 60)
print("Loading Datasets")
print("=" * 60)

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Clean and convert Price column to numérico
        # Delete thousand comas if there exist (e.g.: "1,234.56" -> "1234.56")
        if df['Price'].dtype == 'object':
            df['Price'] = df['Price'].astype(str).str.replace(',', '')
        
        # Convert to float
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
        # Delete rows with invalid prices
        df = df.dropna(subset=['Price'])
        
        # Order by date (oldest first)
        df = df.sort_values('Date')
        
        # Calculate daily returns (log returns)
        df['Return'] = np.log(df['Price'] / df['Price'].shift(1))
        
        # Delete the first value (NaN) and any infinite value
        df = df.dropna(subset=['Return'])
        df = df[~np.isinf(df['Return'])]
        
        # Save to the dictionary
        index_name = csv_file.replace("_Historical_Data.csv", "").replace("_", " ")
        dataframes[index_name] = df
        all_returns.append(df['Return'].values)
        
        print(f"\n✓ {index_name}:")
        print(f"  Period: {df['Date'].min().date()} a {df['Date'].max().date()}")
        print(f"  Observations: {len(df)}")
        print(f"  Mean return: {df['Return'].mean():.6f}")
        print(f"  Standard deviation: {df['Return'].std():.6f}")
        
    except FileNotFoundError:
        print(f"\n✗ ERROR: Couldn't find the file {csv_file}")

if len(dataframes) == 0:
    print("\n" + "=" * 60)
    print("ERROR: Couldn't load any dataset")
    print("=" * 60)
    exit()


# PREPARE DATA FOR MONTE CARLO SIMULATION
print("\n" + "=" * 60)
print("Preparing data for simulation")
print("=" * 60)

# N = number of loaded indices
N = len(dataframes)
index_names = list(dataframes.keys())

print(f"Number of inidices (N): {N}")
print(f"Indices: {', '.join(index_names)}")

# Find the common date range for all datasets
common_start = max([df['Date'].min() for df in dataframes.values()])
common_end = min([df['Date'].max() for df in dataframes.values()])

print(f"\nPeríodo común: {common_start.date()} a {common_end.date()}")

# Filter all dataframes to the common period
aligned_returns = []
for name, df in dataframes.items():
    df_filtered = df[(df['Date'] >= common_start) & (df['Date'] <= common_end)].copy()
    df_filtered = df_filtered.set_index('Date')
    aligned_returns.append(df_filtered['Return'])

# Create a DataFrame with all returns aligned by date
returns_df = pd.concat(aligned_returns, axis=1, join='inner')
returns_df.columns = index_names

print(f"Observations from common period: {len(returns_df)}")

# Convert to a numpy matrix (each column is an index)
returns_matrix = returns_df.values

# Calculate the mean of each index
mu = returns_matrix.mean(axis=0)

# Calculate the covariance matrix
Sigma = np.cov(returns_matrix.T)

print("\n" + "-" * 60)
print("CALCULATED PATAMETERS")
print(f"\nMean vector (μ):")
for i, name in enumerate(index_names):
    print(f"  {name}: {mu[i]:.6f}")

print(f"\nCovariance matrix (Σ):")
print(Sigma)

print(f"\nStandard deviations:")
for i, name in enumerate(index_names):
    print(f"  {name}: {np.sqrt(Sigma[i,i]):.6f}")

print(f"\nCorrelation matrix:")
corr_matrix = np.corrcoef(returns_matrix.T)
print(corr_matrix)
for i in range(N):
    for j in range(i+1, N):
        print(f"  {index_names[i]} vs {index_names[j]}: {corr_matrix[i,j]:.4f}")


# EXECUTE MONTE CARLO SIMULATION
# Simulation parameters
k = 2           # At least 2 indices must drop (needs to be <= N)
x = 0.02        # 2% drop (0.02 = 2%)
M = 100_000     # Number of simulations

# Parameter validation
if k > N:
    print("\n" + "-" * 60)
    print("WARNING: k > N")
    print(f"k (minimum number of indices that must drop) = {k}")
    print(f"N (total number of indices) = {N}")
    print(f"\nIt is not possible that {k} indices drop of there are only {N} indices.")
    print(f"Adjusting k a {N} (all indices must drop)")
    k = N

print("\n" + "" * 60)
print("SIMULATION CONFIGURATION")
print(f"N (number of indices): {N}")
print(f"k (minimum number of indices that must drop): {k}")
print(f"x (drop threshold): {x*100}%")
print(f"M (number of simulations): {M:,}")
print(f"\nInterpretation:")
print(f"  Calculating the probability that at lest {k} out of {N}")
print(f"  indices drop more than {x*100}% in the same day.")

# Execute simulation
P_hat, count, S_values = monte_carlo_crash_simulation(x, N, k, mu, Sigma, M)

# Calculate precision
std_error, ci_lower, ci_upper = compute_accuracy(P_hat, M)


# RESULTS
print("\n" + "-" * 60)
print("FINAL RESULTS")
print(f"Estimated probability (P̂_MIN): {P_hat:.6f} ({P_hat*100:.4f}%)")
print(f"Extreme events: {count:,} out of {M:,} simulations")
print(f"Standard error: {std_error:.6f}")
print(f"95% confidence interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
print(f"  In percentage format: [{ci_lower*100:.4f}%, {ci_upper*100:.4f}%]")


# SAVE RESULTS IN CSV FILES
# 1. Save summary of results
results_summary = {
    'Parameter': [
        'N_indices',
        'k_minimum_drops',
        'x_drop_threshold',
        'M_simulations',
        'P_hat_probability',
        'P_hat_percentage',
        'extreme_events',
        'standard_error',
        'IC_95_lower',
        'IC_95_upper',
        'IC_95_lower_pct',
        'IC_95_upper_pct'
    ],
    'Value': [
        N,
        k,
        x,
        M,
        P_hat,
        P_hat * 100,
        count,
        std_error,
        ci_lower,
        ci_upper,
        ci_lower * 100,
        ci_upper * 100
    ],
    'Description': [
        'Number of analyzed indices',
        'Minimum number of indices that must drop',
        'Drop threshold (float)',
        'Number of Monte Carlo simulations',
        'Estimated probability (float)',
        'Estimated probability (percentage)',
        'Number of extreme events observed',
        'Standard error of the estimation',
        'Lower bound IC 95% (float)',
        'Lower bound IC 95% (float)',
        'Upper bound IC 95% (percentage)',
        'Upper bound IC 95% (percentage)'
    ]
}

df_summary = pd.DataFrame(results_summary)
df_summary.to_csv('monte_carlo_summary.csv', index=False)
print("\nSummary saved in: monte_carlo_summary.csv")

# 2. Save the S_values distribution (number of drops per simulation)
s_distribution = pd.Series(S_values).value_counts().sort_index()
df_s_dist = pd.DataFrame({
    'Number_of_drops': s_distribution.index,
    'Frequence': s_distribution.values,
    'Percentage': (s_distribution.values / M) * 100
})
df_s_dist.to_csv('monte_carlo_distribution.csv', index=False)
print("Drop distribution saved in: monte_carlo_distribution.csv")

# 3. Save model parameters (mu and Sigma)
df_params = pd.DataFrame({
    'Index': index_names,
    'Return_average': mu,
    'Stardard_deviation': np.sqrt(np.diag(Sigma))
})
df_params.to_csv('monte_carlo_parameters.csv', index=False)
print("Model parameters saved in: monte_carlo_parameters.csv")

# 4. Save covariance matrix
df_cov = pd.DataFrame(Sigma, columns=index_names, index=index_names)
df_cov.to_csv('monte_carlo_covariance_matrix.csv')
print("Covariance matrix saved in: monte_carlo_covariance_matrix.csv")

# 5. Save correlation matrix
df_corr = pd.DataFrame(corr_matrix, columns=index_names, index=index_names)
df_corr.to_csv('monte_carlo_correlation_matrix.csv')
print("Correlation matrix saved in: monte_carlo_correlation_matrix.csv")

# 6. Save all S values (the file can be too big, it can be taken out, not needet)
df_all_sims = pd.DataFrame({'Simulacion': range(M), 'Number_of_drops': S_values})
df_all_sims.to_csv('monte_carlo_all_simulations.csv', index=False)
print("All simulations saved in: monte_carlo_all_simulations.csv")


# VISUALIZATION
plt.figure(figsize=(12, 8))

# Subplot 1: S_values distribution
plt.subplot(2, 2, 1)
plt.hist(S_values, bins=range(0, N+2), edgecolor='black', alpha=0.7)
plt.axvline(k, color='red', linestyle='--', linewidth=2, label=f'Threshold k={k}')
plt.xlabel('Number of dropped indices')
plt.ylabel('Frequence')
plt.title('Distribution of number of drops per simulation')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Temporal series of all indices
plt.subplot(2, 2, 2)
for name, df in dataframes.items():
    # Normalize to 100 in the beginning to compare
    normalized = (df['Price'] / df['Price'].iloc[0]) * 100
    plt.plot(df['Date'], normalized, label=name, alpha=0.7, linewidth=1.5)
plt.xlabel('Date')
plt.ylabel('Normalized price (base 100)')
plt.title('Temporal series of the indices')
plt.legend(fontsize=8)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Subplot 3: Histogramas of superposed returns
plt.subplot(2, 2, 3)
for name in index_names:
    plt.hist(returns_df[name], bins=50, alpha=0.5, label=name, density=True)
plt.axvline(-x, color='red', linestyle='--', linewidth=2, label=f'Threshold -{x*100}%')
plt.xlabel('Daily reutrn')
plt.ylabel('Density')
plt.title('Daily return distribution')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# Subplot 4: Correlation matrix with names
plt.subplot(2, 2, 4)
im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im)
plt.title('Correlation matrix between indices')
plt.xticks(range(N), [name.replace(' ', '\n') for name in index_names], 
           rotation=45, ha='right', fontsize=8)
plt.yticks(range(N), index_names, fontsize=8)
for i in range(N):
    for j in range(N):
        text = plt.text(j, i, f'{corr_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.tight_layout()
plt.savefig('monte_carlo_results.png', dpi=300, bbox_inches='tight')
print("\nGraph saved as 'monte_carlo_results.png'")
plt.show()
