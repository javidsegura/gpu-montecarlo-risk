---
name: Performance Issue
about: Report performance problems or optimization opportunities
title: '[PERFORMANCE] '
labels: performance
assignees: ''
---

## Performance Issue Description
A clear description of the performance problem or optimization opportunity.

## Current Performance
- **Execution Time**: [e.g., 45.2 seconds]
- **Throughput**: [e.g., 10,000 simulations/second]
- **Memory Usage**: [e.g., 8GB RAM, 2GB GPU memory]
- **CPU Utilization**: [e.g., 75% across 16 cores]
- **GPU Utilization**: [e.g., 85% utilization, 60% memory usage]

## Expected Performance
- **Target Execution Time**: [e.g., < 20 seconds]
- **Target Throughput**: [e.g., > 25,000 simulations/second]
- **Memory Budget**: [e.g., < 4GB RAM, < 1GB GPU memory]

## Test Configuration
### Hardware
- **CPU**: [e.g., Intel Xeon E5-2686 v4, 16 cores]
- **GPU**: [e.g., NVIDIA Tesla V100, 32GB memory]
- **Memory**: [e.g., 64GB DDR4]
- **Network**: [e.g., InfiniBand, Ethernet]

### Software
- **OS**: [e.g., Ubuntu 20.04 LTS]
- **CUDA**: [e.g., 11.8]
- **MPI**: [e.g., OpenMPI 4.1.0]
- **Compiler**: [e.g., GCC 9.4.0 with -O3]

### Test Parameters
- **Simulations**: [e.g., 1,000,000]
- **Time Steps**: [e.g., 252 (daily for 1 year)]
- **Processes**: [e.g., 8 MPI processes]
- **Threads per Process**: [e.g., 4 OpenMP threads]
- **GPU Blocks/Threads**: [e.g., 256 blocks, 512 threads/block]

## Bottleneck Analysis
### Profiling Results
- **Hotspots**: [e.g., Random number generation: 40%, Monte Carlo loop: 35%]
- **Memory Bandwidth**: [e.g., 150 GB/s achieved, 300 GB/s theoretical]
- **GPU Kernel Performance**: [e.g., Occupancy: 75%, Registers per thread: 32]

### Identified Issues
- [ ] **CPU Bottleneck**: High CPU utilization, low GPU utilization
- [ ] **Memory Bottleneck**: High memory bandwidth usage
- [ ] **Communication Overhead**: High MPI communication time
- [ ] **Load Imbalance**: Uneven work distribution
- [ ] **Cache Issues**: Poor cache locality
- [ ] **GPU Memory**: Memory transfer bottlenecks

## Optimization Suggestions
### Potential Improvements
- [ ] **Algorithm**: [e.g., Variance reduction techniques]
- [ ] **Data Structures**: [e.g., Structure of Arrays vs Array of Structures]
- [ ] **Memory Access**: [e.g., Coalesced memory access patterns]
- [ ] **Parallelization**: [e.g., Better work distribution]
- [ ] **Communication**: [e.g., Reduce MPI communication]

### Code Changes
Describe any specific code changes or optimizations you've tried:

```c
// Example optimization
```

## Benchmarking Data
### Baseline Performance
```
Time: 45.2s
Throughput: 10,000 sim/s
Memory: 8GB RAM, 2GB GPU
```

### After Optimization (if any)
```
Time: 38.1s
Throughput: 12,500 sim/s
Memory: 7.5GB RAM, 1.8GB GPU
```

## Additional Context
Add any other context about the performance issue here.

## Checklist
- [ ] I have provided detailed performance metrics
- [ ] I have identified potential bottlenecks
- [ ] I have specified the test configuration
- [ ] I have included profiling results if available
- [ ] I have suggested optimization approaches
