#!/bin/bash
#SBATCH --job-name=convert_h5
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/data/pathology/projects/lohith/logs/slurm-convert_atlaspatch_to_clam-%j.out
#SBATCH --error=/data/pathology/projects/lohith/logs/slurm-convert_atlaspatch_to_clam-%j.err
#SBATCH --no-container-entrypoint
#SBATCH --container-image="dockerdex.umcn.nl:5005/lohith-ashwa/bone:v1"
#SBATCH --container-mounts=/data/pathology:/data/pathology

python3 ~/CLAM/convert_atlaspatch_to_clam.py
