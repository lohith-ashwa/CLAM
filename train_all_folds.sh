#!/bin/bash
#SBATCH --job-name=clam_all
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/pathology/projects/lohith/logs/slurm-clam-all-%j.out
#SBATCH --error=/data/pathology/projects/lohith/logs/slurm-clam-all-%j.err
#SBATCH --no-container-entrypoint
#SBATCH --container-image="dockerdex.umcn.nl:5005/lohith-ashwa/bone:v1"
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/pathology:/data/pathology

pip3 install tensorboardX
pip3 install future
pip3 install git+https://github.com/oval-group/smooth-topk.git

cd ~/CLAM

# ── Train CLAM-SB on all Folds ── 
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --drop_out 0.25 \
    --early_stopping \
    --lr 2e-4 \
    --k 5 \
    --exp_code bone_sex_clam_sb_virchow2_2 \
    --weighted_sample \
    --bag_loss ce \
    --inst_loss svm \
    --task bone_sex \
    --model_type clam_sb \
    --log_data \
    --data_root_dir /data/pathology/projects/lohith \
    --embed_dim 2560 \
    --split_dir bone_sex_classification \
    --results_dir /data/pathology/projects/lohith/clam_results
