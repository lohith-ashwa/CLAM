#!/bin/bash
#SBATCH --job-name=clam_eval
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/pathology/projects/lohith/logs/slurm-eval-all-%j.out
#SBATCH --error=/data/pathology/projects/lohith/logs/slurm-eval-all-%j.err
#SBATCH --no-container-entrypoint
#SBATCH --container-image="dockerdex.umcn.nl:5005/lohith-ashwa/bone:v1"
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/pathology:/data/pathology


cd ~/CLAM

CUDA_VISIBLE_DEVICES=0 python3 eval.py \
    --k 5 \
    --models_exp_code bone_sex_clam_sb_virchow2_1_s1 \
    --save_exp_code bone_sex_eval \
    --task bone_sex \
    --model_type clam_sb \
    --data_root_dir /data/pathology/projects/lohith \
    --embed_dim 2560 \
    --results_dir /data/pathology/projects/lohith/clam_results \
    --splits_dir /data/pathology/projects/lohith/clam_results/bone_sex_clam_sb_virchow2_1_s1
