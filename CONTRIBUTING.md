# Contributing to GPU-Accelerated Monte Carlo Securities

Welcome to the GPU-accelerated Monte Carlo securities project! This document provides guidelines for contributing to our high-performance computing implementation using C, MPI, OpenMP, and CUDA.

## Table of Contents
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Style Guidelines](#code-style-guidelines)
- [Performance Standards](#performance-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Commit Message Format](#commit-message-format)
- [Documentation Requirements](#documentation-requirements)

## Getting Started

### Prerequisites
- **C Compiler**: GCC 9.0+ or Clang 10.0+
- **MPI**: OpenMPI 4.0+ or MPICH 3.4+
- **CUDA Toolkit**: 11.0+ (for GPU acceleration)
- **OpenMP**: Usually included with GCC/Clang
- **Development Tools**: Git, CMake 3.16+, Valgrind, GDB

### Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/javidsegura/gpu-montecarlo-risk
   cd gpu-montecarlo-risk
   ```

2. Build the project [still undefined]:
   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ```

3. Run tests [still undefined]:
   ```bash
   make test
   ```

## Development Environment

### Recommended Setup
- **IDE**: VS Code with C/C++ extensions, or CLion
- **Profiling**: NVIDIA Nsight Systems, Intel VTune
- **Debugging**: GDB, CUDA-GDB, Valgrind
- **Code Analysis**: Clang Static Analyzer, Cppcheck

### GPU Development
- Use `nvidia-smi` to monitor GPU usage
- Profile with `nvprof` or Nsight Compute
- Ensure CUDA memory is properly managed (allocated/freed)

## Code Style Guidelines

### C Coding Standards
- Follow **GNU C Style** with modifications for HPC
- Use **4 spaces** for indentation (no tabs)
- Maximum line length: **100 characters**
- Use `snake_case` for variables and functions
- Use `UPPER_CASE` for constants and macros

### Example Code Style
```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#define MAX_SIMULATIONS 1000000
#define DEFAULT_THREADS 4

typedef struct {
    double spot_price;
    double strike_price;
    double time_to_maturity;
    double risk_free_rate;
    double volatility;
} option_params_t;

int run_monte_carlo_simulation(const option_params_t* params, 
                               int num_simulations,
                               int num_threads) {
    // Implementation here
    return 0;
}
```

### MPI Guidelines
- Always check MPI return codes
- Use collective operations when possible
- Minimize communication overhead
- Use appropriate data types for MPI calls

```c
int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

// Good: Check return codes
if (MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, 
                  MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Allreduce failed\n");
    return -1;
}
```

### OpenMP Guidelines
- Use `#pragma omp parallel` for parallel regions
- Specify data sharing attributes explicitly
- Use reduction clauses for accumulations
- Be careful with thread safety

```c
double sum = 0.0;
#pragma omp parallel for reduction(+:sum) num_threads(num_threads)
for (int i = 0; i < num_simulations; i++) {
    sum += calculate_payoff(i);
}
```

### CUDA Guidelines
- Use proper error checking with `cudaError_t`
- Implement proper memory management
- Use shared memory for frequently accessed data
- Optimize memory access patterns (coalescing)

```c
// Good: Error checking
cudaError_t err = cudaMalloc(&d_data, size);
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA malloc failed: %s\n", 
            cudaGetErrorString(err));
    return -1;
}

// Good: Memory cleanup
if (d_data != NULL) {
    cudaFree(d_data);
}
```

## Performance Standards

### Benchmarking Requirements
- All performance-critical code must include benchmarks
- Report performance in simulations per second
- Include memory usage (CPU and GPU)
- Test scalability with different core/thread counts

### Performance Targets
- **CPU-only**: > 10,000 simulations/second per core
- **GPU acceleration**: > 100,000 simulations/second
- **MPI scaling**: > 80% efficiency up to 32 processes
- **Memory efficiency**: < 1GB RAM per million simulations

### Profiling Requirements [undefined]

## Testing Requirements

### Unit Tests
- Test individual functions with known inputs/outputs
- Test edge cases and error conditions
- Use CMake's CTest framework

### Integration Tests
- Test MPI communication patterns
- Test OpenMP thread safety
- Test CUDA kernel correctness

### Performance Tests
- Benchmark against baseline performance
- Test with various input sizes
- Verify scaling behavior

### Example Test Structure
```c
#include <assert.h>
#include "monte_carlo.h"

void test_option_pricing() {
    option_params_t params = {
        .spot_price = 100.0,
        .strike_price = 105.0,
        .time_to_maturity = 1.0,
        .risk_free_rate = 0.05,
        .volatility = 0.2
    };
    
    double price = calculate_option_price(&params, 1000000);
    
    // Price should be within reasonable range
    assert(price > 0.0 && price < 200.0);
}
```

## Pull Request Process

### Before Submitting
1. **Create an issue** if it doesn't exist
2. **Fork and branch** from main
3. **Write tests** for new functionality
4. **Update documentation** if needed
5. **Run all tests** and ensure they pass
6. **Profile performance** if applicable

### PR Requirements
- **Clear description** of changes and motivation
- **Performance impact** analysis
- **Test results** and coverage
- **Breaking changes** clearly marked

### Review Process
- At least **2 approvals** required
- **Performance PRs** require performance team review
- **Core algorithm changes** require team lead approval

## Commit Message Format

Use the following format for commit messages:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `docs`: Documentation changes
- `style`: Code style changes
- `build`: Build system changes

### Examples
```
feat(cuda): add GPU-accelerated random number generation

perf(mpi): optimize communication pattern for large simulations

fix(memory): resolve CUDA memory leak in option pricing kernel
```

## Documentation Requirements

### Code Documentation
- Document all public functions with Doxygen-style comments
- Include parameter descriptions and return values
- Provide usage examples for complex functions

### Performance Documentation
- Document algorithmic complexity
- Include scaling characteristics
- Provide benchmark results

### Example Documentation
```c
/**
 * @brief Calculate option price using Monte Carlo simulation
 * 
 * This function performs a Monte Carlo simulation to price European options
 * using GPU acceleration when available.
 * 
 * @param params Pointer to option parameters structure
 * @param num_simulations Number of Monte Carlo simulations to run
 * @param use_gpu If true, use GPU acceleration (requires CUDA)
 * @return Calculated option price, or -1.0 on error
 * 
 * @note Thread-safe when using different parameter structures
 * @warning num_simulations must be positive
 * 
 * @example
 * ```c
 * option_params_t params = {100.0, 105.0, 1.0, 0.05, 0.2};
 * double price = calculate_option_price(&params, 1000000, true);
 * printf("Option price: %.4f\n", price);
 * ```
 */
double calculate_option_price(const option_params_t* params,
                              int num_simulations,
                              bool use_gpu);
```

## Performance Optimization Guidelines

### CPU Optimization
- Use compiler optimization flags (`-O3`, `-march=native`)
- Profile to identify hotspots
- Use appropriate data structures (cache-friendly)
- Minimize function call overhead in hot loops

### GPU Optimization
- Optimize memory access patterns
- Use appropriate block/thread configurations
- Minimize host-device memory transfers
- Use shared memory for frequently accessed data

### MPI Optimization
- Use appropriate communication patterns
- Minimize communication frequency
- Use non-blocking operations when possible
- Consider communication-computation overlap

## Security and Financial Considerations

### Numerical Stability
- Use appropriate random number generators
- Handle edge cases (zero volatility, extreme parameters)
- Implement proper error handling

### Validation
- Compare results with analytical solutions when available
- Use multiple random seeds for verification
- Implement convergence criteria

---

Thank you for contributing to our GPU-accelerated Monte Carlo securities project! 🚀
