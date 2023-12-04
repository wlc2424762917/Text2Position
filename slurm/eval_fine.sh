#!/bin/bash
#SBATCH --job-name="Eval fine"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:31G
#SBATCH --mem=10G
#SBATCH --time=0:59:00
#SBATCH --mail-type=NONE
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

source /usr/stud/kolmet/venv/pyg/bin/activate
module load cuda/10.2
srun python3 -m evaluation.fine "$@"
