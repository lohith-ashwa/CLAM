#!/bin/bash
#SBATCH --job-name=heatmap
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/pathology/projects/lohith/logs/slurm-generate_heatmap_process-%j.out
#SBATCH --error=/data/pathology/projects/lohith/logs/slurm-generate_heatmap_process-%j.err
#SBATCH --no-container-entrypoint
#SBATCH --container-image="dockerdex.umcn.nl:5005/lohith-ashwa/bone:v1"
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/pathology:/data/pathology

cd ~/CLAM

python3 generate_heatmap_process_list.py --fold 0
