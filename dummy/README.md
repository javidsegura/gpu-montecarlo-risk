## From Python to C: Monte Carlo Crash Simulation (with GSL)

This guide shows how to go from the Python implementation in `python-demo/main.py` to the C implementation in `dummy/main.c` using GSL (GNU Scientific Library). It is written for readers with little practical C experience and focuses on concrete steps and side‑by‑side mappings.

### What the program does
- Simulates correlated asset returns R ~ N(μ, Σ) using a Cholesky factor L of Σ
- Counts, per trial, how many assets fall below −x
- Estimates P̂_MIN = C / M where C is the number of trials with at least k such “crashes”
- Reports a 95% confidence interval using a normal approximation

## Prerequisites

### Install GSL
- macOS (Homebrew):
```bash
brew install gsl
```
- Ubuntu/Debian:
```bash
sudo apt-get update && sudo apt-get install -y libgsl-dev
```
- Fedora:
```bash
sudo dnf install gsl-devel
```

### Check your C toolchain
```bash
gcc --version
```
If you don’t have `gcc`, install Xcode Command Line Tools on macOS (`xcode-select --install`) or install `build-essential` on Debian/Ubuntu.

## Build and run the C version
From the project root:
```bash
cd dummy
gcc -O2 -Wall -Wextra -o mc_crash main.c -lgsl -lgslcblas -lm
./mc_crash
```
If `gcc` cannot find GSL, you may need to add include/library paths, for example on macOS with Homebrew (replace prefix if different):
```bash
gcc -O2 -I/opt/homebrew/include -L/opt/homebrew/lib -o mc_crash main.c -lgsl -lgslcblas -lm
```

## Quick parameter consistency note (x units)
- In C, `x` is a fraction (e.g., 0.02 means 2%). It prints as a percentage via `x * 100`.
- Ensure the Python script uses the same scale. For a 2% drop, set `x = 0.02`.

## Side-by-side: Python → C mapping

### Function signature and return values
Python returns `(P_hat, count, S_values)` from a function.
```14:39:/Users/javierdominguezsegura/Programming/Junior/HPC/final_project/python-demo/main.py
def monte_carlo_crash_simulation(x, N, k, mu, Sigma, M):
    """
    Pure Monte Carlo simulation following the pseudocode exactly.
    ...
    Returns:
    - P_hat : float
    - count : int
    - S_values : array
    """
```
C returns a `SimulationResult` struct with the same data.
```11:15:/Users/javierdominguezsegura/Programming/Junior/HPC/final_project/dummy/main.c
typedef struct {
    double P_hat;
    int count;
    double *S_values;
} SimulationResult;
```

### Cholesky decomposition (Σ = L Lᵀ)
Python uses NumPy’s Cholesky:
```41:44:/Users/javierdominguezsegura/Programming/Junior/HPC/final_project/python-demo/main.py
# BEFORE SIMULATION: Cholesky decomposition of Sigma
# Find lower triangular matrix L such that Σ = LL'
L = np.linalg.cholesky(Sigma)
```
C uses GSL and decomposes in place, storing the factor into `L`:
```49:53:/Users/javierdominguezsegura/Programming/Junior/HPC/final_project/dummy/main.c
// BEFORE SIMULATION: Cholesky decomposition
// Copy Sigma to L (gsl_linalg_cholesky_decomp modifies in place)
gsl_matrix_memcpy(L, Sigma);
gsl_linalg_cholesky_decomp1(L);
```

### Random normal vector Z and correlated returns R = μ + L Z
Python:
```54:59:/Users/javierdominguezsegura/Programming/Junior/HPC/final_project/python-demo/main.py
Z = np.random.standard_normal(N)
# R = μ + L @ Z
R = mu + L @ Z
```
C with GSL:
```64:73:/Users/javierdominguezsegura/Programming/Junior/HPC/final_project/dummy/main.c
for (int i = 0; i < N; i++) {
    gsl_vector_set(Z, i, gsl_ran_gaussian(rng, 1.0));
}
// R = L*Z then add μ
gsl_blas_dgemv(CblasNoTrans, 1.0, L, Z, 0.0, R);
gsl_vector_add(R, mu_vec);
```

### Counting crashes and updating the estimate
Python:
```60:67:/Users/javierdominguezsegura/Programming/Junior/HPC/final_project/python-demo/main.py
S = np.sum(R < -x)
S_values[j] = S
if S >= k:
    count += 1 
```
C:
```75:87:/Users/javierdominguezsegura/Programming/Junior/HPC/final_project/dummy/main.c
int S = 0;
for (int i = 0; i < N; i++) {
    if (gsl_vector_get(R, i) < -x) {
        S++;
    }
}
result.S_values[j] = (double)S;
if (S >= k) {
    result.count++;
}
```

### Final probability and confidence interval
Python:
```72:78:/Users/javierdominguezsegura/Programming/Junior/HPC/final_project/python-demo/main.py
P_hat = count / M
print("-" * 60)
print(f"Simulation complete!")
return P_hat, count, S_values
```
C:
```96:109:/Users/javierdominguezsegura/Programming/Junior/HPC/final_project/dummy/main.c
// Compute probability: P̂_MIN = C/M
result.P_hat = (double)result.count / M;
printf("------------------------------------------------------------\n");
printf("Simulation complete!\n");
return result;
```
Confidence interval helper in C:
```115:125:/Users/javierdominguezsegura/Programming/Junior/HPC/final_project/dummy/main.c
void compute_accuracy(double P_hat, int M, double *std_error, 
                     double *ci_lower, double *ci_upper) {
    *std_error = sqrt(P_hat * (1.0 - P_hat) / M);
    double z_score = 1.96; // for 95% confidence
    double margin = z_score * (*std_error);
    *ci_lower = fmax(0.0, P_hat - margin);
    *ci_upper = fmin(1.0, P_hat + margin);
}
```

## Running the Python version (for comparison)
From the project root:
```bash
cd python-demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Common pitfalls and tips
- x scale: Use `x = 0.02` for a 2% threshold in both languages.
- RNG seed: C version seeds with a fixed value (`42`) for reproducibility; adjust or remove for variability.
- Memory management: In C, every allocation must be freed (already handled in `main.c`).
- BLAS/LAPACK backend: GSL uses its own CBLAS; ensure you link `-lgsl -lgslcblas -lm`.
- Performance: Build with `-O2` or higher. For very large M, consider `-O3`.

## Minimal Makefile (optional)
If you prefer a Makefile in `dummy/`:
```make
mc_crash: main.c
	gcc -O2 -Wall -Wextra -o mc_crash main.c -lgsl -lgslcblas -lm

run: mc_crash
	./mc_crash
```
Use with:
```bash
make -C dummy run
```

## Where to look in the code
- Python entry point and parameters: see `python-demo/main.py` bottom `if __name__ == "__main__":`.
- C entry point and parameters: see `dummy/main.c` `main()` where N, k, x, M, and Σ are defined.

With these steps and mappings, you can replicate the Python simulation in C, compile it locally, and understand how each Python operation translates into GSL-based C code.


