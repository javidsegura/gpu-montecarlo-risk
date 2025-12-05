# HOW TO EXECUTE
1. For Cuda stuff make sure comment out all models that arent 'cuda'. Then run `sbatch gpu_script.sh`
2. For the CPU stuff comment out 'cuda' model and run `sbatch cpu_script.sh`

Accelerated a multi-asset risk engine by 192×, boosting throughput from ~188k to 36 million simulations per second on NVIDIA GPUs; reduced runtime for 100M iterations from 9 minutes (Python) to under 3 seconds (CUDA).
Engineered a hybrid C++/CUDA pipeline with parallel Cholesky decomposition and cuRAND-based path generation for pricing correlated assets, ensuring numerical consistency across CPU and GPU backends.
Designed a comparative benchmarking suite profiling Serial, OpenMP, MPI, and CUDA implementations; identified and resolved memory bottlenecks in RNG and reduction kernels to achieve optimal GPU occupancy.
