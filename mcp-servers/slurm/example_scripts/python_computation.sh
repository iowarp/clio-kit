#!/bin/bash
#SBATCH --job-name=python_computation
#SBATCH --output=logs/slurm_output/slurm_%j.out
#SBATCH --error=logs/slurm_output/slurm_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB

echo "=== Python Computation ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start Time: $(date)"

python3 - <<'PY'
import math
import random
import time

start = time.perf_counter()

primes = [n for n in range(2, 50_000) if all(n % d for d in range(2, math.isqrt(n) + 1))]
print(f"Primes below 50000: {len(primes)} (largest {primes[-1]})")

random.seed(42)
samples = 1_000_000
inside = 0
for step in range(1, samples + 1):
    x, y = random.random(), random.random()
    inside += x * x + y * y <= 1.0
    if step % 250_000 == 0:
        print(f"progress: {step}/{samples} samples, pi ~ {4 * inside / step:.6f}", flush=True)

print(f"Monte Carlo pi estimate: {4 * inside / samples:.6f} (error {abs(4 * inside / samples - math.pi):.2e})")
print(f"Elapsed: {time.perf_counter() - start:.2f} s")
PY

echo "End Time: $(date)"
