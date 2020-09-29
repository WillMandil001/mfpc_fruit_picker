#!/bin/bash
# file submit.sh
#SBATCH --job-name=RCNLSTM
#SBATCH -N 1
#SBATCH --output=mfpc_all_models_002_OUPUT.out




## Run two jobs in parallel on the SAME node (e.g. if each job requires a single GPU)
## Currently, creates an additional empty output file "slurm-jobid.out", not sure
## how to avoid.

module add cuda/7.5

srun python3 mfpc_all_models.py