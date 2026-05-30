#!/bin/bash
#SBATCH --job-name=clam_eval_5fold
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=0
#Have reduced the specs
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/pathology/projects/lohith/logs/slurm-eval-5fold-%j.out
#SBATCH --error=/data/pathology/projects/lohith/logs/slurm-eval-5fold-%j.err
#SBATCH --no-container-entrypoint
#SBATCH --container-image="dockerdex.umcn.nl:5005/lohith-ashwa/bone:v1"
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/pathology:/data/pathology

cd ~/CLAM

python3 evaluate_5fold.py \
    --eval_dir eval_results/EVAL_bone_sex_eval \
    --save_dir /data/pathology/projects/lohith/evaluation/5fold_best \
    --n_folds 5
