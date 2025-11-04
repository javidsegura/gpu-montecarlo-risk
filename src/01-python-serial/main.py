import numpy as np
import matplotlib.pyplot as plt
"""

Informal goal: estimating financial doomsday
Formal Goal: inferencing change of k assets (out of N assets -- assets being here stock market indices)
       dropping (their return) by more than a percentage x


"""



def monte_carlo_crash_simulation(x, N, k, mu, Sigma, M):
    """
    Pure Monte Carlo simulation following the pseudocode exactly.
    
    Parameters:
    - x : float
        Drop percentage threshold
    - N : int
        Number of indices
    - k : int
        Number of crashes needed for extreme event
    - mu : array, shape (N,)
        Mean return vector (mu_i = assets return mea)
    - Sigma : array, shape (N, N)
        Covariance matrix
    - M : int
        Number of Monte Carlo trials
    
    Returns:
    - P_hat : float
        Estimated probability P̂_MIN
    - count : int
        Count C of extreme events
    - S_values : array
        Array of S^(j) values for each trial
    """
    
    # BEFORE SIMULATION: Cholesky decomposition of Sigma
    # Find lower triangular matrix L such that Σ = LL'
    L = np.linalg.cholesky(Sigma)
    
    print(f"Starting Monte Carlo simulation with M = {M:,} trials...")
    print(f"Parameters: N={N}, k={k}, x={x}%")
    print("-" * 60)
    
    # Initialize crash counter and S_values storage for each trial
    count = 0 
    S_values = np.zeros(M) 
    # SIMULATION
    for j in range(M):
        # (i) Get random sample vector R^(j): Generate vector of independent N(0,1) variables
        Z = np.random.standard_normal(N)
        
        # R = μ + L*Z  (gives one sample from N(μ, Σ))
        R = mu + L @ Z
        
        # (ii) Compute the count: S^(j) = Σ I(Ri^(j) < -x)
        S = np.sum(R < -x)
        S_values[j] = S
        
        # (iii) Check if S^(j) ≥ k
        if S >= k:
            count += 1 
        
        # Progress indicator
        if (j + 1) % (M // 10) == 0:
            print(f"Progress: {(j+1)/M*100:.0f}% - Current estimate: {count/(j+1):.6f}")
    
    # Compute probability: P̂_MIN = C/M
    P_hat = count / M
    
    print("-" * 60)
    print(f"Simulation complete!")
    
    return P_hat, count, S_values

def compute_accuracy(P_hat, M):
    """
    Compute accuracy and confidence interval.
    The estimation error is on the order of √(β(1-β)/M)
    
    Parameters:
    - P_hat : float
        Estimated probability
    - M : int
        Number of trials
    
    Returns:
    - std_error : float
        Standard error
    - ci_lower : float
        Lower bound of 95% confidence interval
    - ci_upper : float
        Upper bound of 95% confidence interval
    """
    # Standard error: √(p(1-p)/M)
    std_error = np.sqrt(P_hat * (1 - P_hat) / M)
    
    z_score = 1.96  # for 95% confidence
    margin = z_score * std_error
    
    ci_lower = max(0, P_hat - margin)
    ci_upper = min(1, P_hat + margin)
    
    return std_error, ci_lower, ci_upper
    
if __name__ == "__main__":
      # =============================================================================
      # EXECUTION EXAMPLE
      # =============================================================================

      # Define parameters
      N = 10          # Number of indices
      k = 5           # At least 5 must crash
      x = 0.2         # 2% drop threshold
      M = 100_000     # Number of Monte Carlo trials
      # Define mean returns (assume zero for simplicity)
      mu = np.zeros(N)
      # Define covariance matrix (with some correlation)
      rho = 0.3       # Correlation coefficient
      variance = 0.04 # 4% variance (2% std dev)
      Sigma = np.full((N, N), rho * variance) + np.eye(N) * variance * (1 - rho)

      # Run simulation
      P_hat, count, S_values = monte_carlo_crash_simulation(x, N, k, mu, Sigma, M)
      # Compute accuracy
      std_error, ci_lower, ci_upper = compute_accuracy(P_hat, M)

      # Print results
      print("\n" + "="*60)
      print("FINAL RESULTS")
      print("="*60)
      print(f"Estimated Probability (P̂_MIN): {P_hat:.6f}")
      print(f"Extreme Events Count: {count:,} out of {M:,} trials")
      print(f"Standard Error: {std_error:.6f}")
      print(f"95% Confidence Interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
      print("="*60)