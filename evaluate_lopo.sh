#!/bin/bash
#SBATCH --job-name=clam_eval_lopo
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/pathology/projects/lohith/logs/slurm-eval-lopo-%j.out
#SBATCH --error=/data/pathology/projects/lohith/logs/slurm-eval-lopo-%j.err
#SBATCH --no-container-entrypoint
#SBATCH --container-image="dockerdex.umcn.nl:5005/lohith-ashwa/bone:v1"
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/pathology:/data/pathology

cd ~/CLAM
 
python3 evaluate_lopo.py \
    --eval_dir eval_results/EVAL_bone_sex_eval_6 \
    --save_dir /data/pathology/projects/lohith/evaluation/lopo_6 \
    --n_folds 15
