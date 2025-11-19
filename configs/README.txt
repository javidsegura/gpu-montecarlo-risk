Configuration Files
===================

config.yaml         - Default configuration (CPU: serial/openmp)
config_gpu.yaml     - GPU configuration template (CUDA with large M)

Usage:
------
1. Edit config.yaml directly for your run, or
2. Copy a template: cp configs/config_gpu.yaml configs/config.yaml

All scripts automatically read from: configs/config.yaml

