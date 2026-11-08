#!/bin/bash
#SBATCH --job-name=hello_world
#SBATCH --output=logs/slurm_output/slurm_%j.out
#SBATCH --error=logs/slurm_output/slurm_%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB

echo "Hello from Slurm job ${SLURM_JOB_ID:-unknown} on $(hostname)"
echo "Job name: ${SLURM_JOB_NAME:-unknown}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-1}"
echo "Working directory: $(pwd)"
echo "Date: $(date)"
