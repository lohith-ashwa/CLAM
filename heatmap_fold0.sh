#!/bin/bash
#SBATCH --job-name=heatmap
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --nodelist=dlc-arceus
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/pathology/projects/lohith/logs/slurm-heatmap-%j.out
#SBATCH --error=/data/pathology/projects/lohith/logs/slurm-heatmap-%j.err
#SBATCH --no-container-entrypoint
#SBATCH --container-image="dockerdex.umcn.nl:5005/lohith-ashwa/bone:v1"
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/pathology:/data/pathology

pip3 install scipy --break-system-packages 2>/dev/null || pip3 install scipy

# Hugging Face token for Virchow2 model download
export HF_TOKEN=$HF_TOKEN
cd ~/CLAM

# Pipe 'Y' to bypass the interactive confirmation prompt
echo "Y" | python3 create_heatmaps.py --config bone_sex_fold0.yaml
