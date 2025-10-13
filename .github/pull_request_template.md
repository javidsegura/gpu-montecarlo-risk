## Description
Brief description of the changes made and the motivation behind them.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Performance improvement (optimization without changing functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Build system changes

## Technical Details

### Implementation Approach
- [ ] **MPI**: Distributed computing changes
- [ ] **OpenMP**: Thread-level parallelism
- [ ] **CUDA**: GPU acceleration
- [ ] **Hybrid**: Combination of parallelization techniques

### Performance Impact
<!-- REQUIRED: Performance impact must be specified for all changes -->
- **Expected Speedup**: [MANDATORY: e.g., 2x faster, 50% memory reduction, or "No performance impact"]
- **Memory Usage**: [MANDATORY: e.g., No change, +100MB RAM, -200MB GPU memory]
- **Scalability**: [MANDATORY: e.g., Better scaling up to 32 processes, or "No scalability impact"]

### Algorithm Changes
- **Monte Carlo Method**: [e.g., Variance reduction, Quasi-random numbers]
- **Security Type**: [e.g., Options, Bonds, Complex derivatives]
- **Risk Metrics**: [e.g., VaR, Expected Shortfall, Greeks]

## Testing

### Test Coverage
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Performance benchmarks included
- [ ] Memory leak testing completed

### Test Results
<!-- REQUIRED: Test results must be provided -->
```
Test Results:
- Unit tests: PASSED (X/Y tests)
- Integration tests: PASSED (X/Y tests)
- Performance benchmarks: [Results here]
- Memory usage: [Results here]
- CUDA compilation: PASSED/FAILED
- MPI communication tests: PASSED/FAILED
```

### Performance Benchmarks
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Execution Time | X.Xs | Y.Ys | Z% |
| Throughput | X sim/s | Y sim/s | Z% |
| Memory Usage | X MB | Y MB | Z% |
| GPU Utilization | X% | Y% | Z% |

## Code Quality

### Code Style
- [ ] Follows project coding standards
- [ ] Proper error handling implemented
- [ ] Memory management is correct (no leaks)
- [ ] CUDA memory properly allocated/freed
- [ ] MPI error codes checked

### Documentation
- [ ] Code is properly commented
- [ ] Public functions documented
- [ ] README updated if needed
- [ ] Performance characteristics documented

## Breaking Changes
If this PR includes breaking changes, describe them here:
- [ ] API changes
- [ ] Configuration changes
- [ ] Dependencies changes
- [ ] Performance characteristics changes

## Dependencies
- [ ] No new dependencies added
- [ ] New dependencies documented
- [ ] Version compatibility verified

## Security/Financial Validation
- [ ] Numerical stability verified
- [ ] Edge cases handled
- [ ] Results validated against known benchmarks
- [ ] Random number generation is appropriate

## Screenshots/Logs
If applicable, add screenshots or logs to help explain your changes.

## Checklist
<!-- REQUIRED: All checkboxes must be checked before PR can be merged -->
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published
- [ ] I have checked performance impact and documented results
- [ ] I have verified memory management (no leaks, proper CUDA cleanup)
- [ ] I have tested with different MPI configurations
- [ ] I have tested with different OpenMP thread counts
- [ ] I have tested GPU functionality (if applicable)
- [ ] I have run static analysis (cppcheck) without critical issues
- [ ] I have validated numerical results against known benchmarks (if applicable)
- [ ] I have checked for memory leaks with Valgrind (if applicable)

## Related Issues
Fixes #(issue number)
Related to #(issue number)

## Additional Notes
Any additional information that reviewers should know.
