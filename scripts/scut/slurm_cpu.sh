#!/bin/bash
#SBATCH --job-name=scut_cpu_task
#SBATCH --partition=cpuHygon7380
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16           # number of cores per tasks
#SBATCH --mem=50000MB                # memory
#SBATCH --output=outputs/%x-%j.out   # output file name
#SBATCH --time=30-00:00:00           # max time

set -x -e

export PYTHONPATH=${PWD}

# Add task here
srun --jobid $SLURM_JOBID bash -c 'bash scripts/scut/litdata_unzip.sh'

echo "DONE"
