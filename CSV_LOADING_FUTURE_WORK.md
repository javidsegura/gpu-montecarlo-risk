# CSV Loading - Changes Needed for Cluster

## Simple Changes to Make When Cluster is Online

### 1. File Paths
**Current**: Hard-coded relative paths assume files are in tests/ directory
```python
# preprocessing/load_csv.py
mu, Sigma = read_csv_parameters('mu.csv', 'Sigma.csv', 9)
```

**Change Needed**: Accept full file paths as parameters, or read from config
```python
mu, Sigma = read_csv_parameters('/cluster/data/mu.csv', '/cluster/data/Sigma.csv', 9)
```

---

### 2. Data Directory Structure
**Current**: Test CSV files in `/tests/` directory

**Change Needed**: Create `/data/` directory for cluster
```
project/
├── data/
│   ├── mu.csv
│   ├── Sigma.csv
│   └── README.md (document format)
├── tests/
│   └── (mock data stays here)
```

---

### 3. Real Data Generation
**Current**: `generate_test_csv.py` creates mock 9x9 random data

**Change Needed**: Download real financial data for 9 European indices
- Use actual historical returns from 2015-2025
- Compute real mu (mean returns) and Sigma (covariance matrix)
- Store in `/data/mu.csv` and `/data/Sigma.csv`

---

### 4. Configuration File
**Current**: N=9 hardcoded in multiple places

**Change Needed**: Create config file to specify data parameters
```yaml
# config.yaml
data:
  n_indices: 9
  indices: [STOXX50E, CAC40, DAX, FTSE100, FTSEMIB, IBEX35, SMI, AEX, BEL20]
  csv_path: /data/
  mu_file: mu.csv
  sigma_file: Sigma.csv
```

Update code to read from config instead of hardcoding values.

---

### 5. Update preprocessing/main.py
**Current**: Creates CSV files with random data

**Change Needed**:
- Load CSV files from `/data/` instead of generating them
- Remove the `save_parameters_to_csv()` function (or repurpose it for results)
- Integration: Fetch real data → Compute stats → Run Monte Carlo

---

### 6. Validation for Real Data
**Current**: Only checks file existence and dimensions

**Change Needed**: Add basic validation for realistic financial data
```python
# In load_csv.py
if not np.all(np.isfinite(mu)):
    raise ValueError("mu contains NaN or inf values")

# Check Sigma is approximately symmetric
if not np.allclose(Sigma, Sigma.T, rtol=1e-5):
    raise ValueError("Sigma matrix is not symmetric")
```

---

## Summary of Changes

| File | Change | Why |
|------|--------|-----|
| `preprocessing/load_csv.py` | Accept config/path parameters | Cluster has different file locations |
| `preprocessing/main.py` | Load from `/data/` instead of generating | Use real financial data |
| `src/csv_io.c` | Add path parameter | Cluster file locations differ |
| Create `config.yaml` | Define data paths and parameters | Avoid hardcoding cluster paths |
| Create `/data/` directory | Store real CSV files | Separate real data from test data |

All existing tests remain the same - they continue to use mock data in `/tests/`.
