#!/bin/bash
# Quick test to demonstrate SAACT logging

echo "=========================================="
echo "Testing SAACT Logging in OpenMP"
echo "=========================================="
echo ""

# Update config to use openmp with a small M value
cat > config.yaml.test << 'EOF'
models:
  - openmp

start: "2015-01-01"
end: "2025-12-31"

indices:
  AEX: "AEX.csv"
  Bel20: "Bel20.csv"

train_ratio: 0.8
M: 1000000  # Small M for quick test
random_seed: 67
x: 0.02
k: 5
comment: "SAACT logging test"
EOF

# Backup original config
cp config.yaml config.yaml.backup

# Use test config
cp config.yaml.test config.yaml

echo "Running simulation with SAACT logging enabled..."
echo "Look for lines starting with [openmp_simulate] in stderr"
echo ""

# Run simulation and capture stderr
./bin/monte_carlo 2>&1 | tee /tmp/saact_output.log

echo ""
echo "=========================================="
echo "SAACT Log Entries:"
echo "=========================================="
grep "\[openmp" /tmp/saact_output.log

echo ""
echo "=========================================="
echo "Restoring original config..."
mv config.yaml.backup config.yaml
rm config.yaml.test

echo "Done! Full output saved to /tmp/saact_output.log"
