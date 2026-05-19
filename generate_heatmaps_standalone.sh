#!/bin/bash
#SBATCH --job-name=heatmap
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/pathology/projects/lohith/logs/slurm-heatmap-%j.out
#SBATCH --error=/data/pathology/projects/lohith/logs/slurm-heatmap-%j.err
#SBATCH --no-container-entrypoint
#SBATCH --container-image="dockerdex.umcn.nl:5005#lohith-ashwa/bone:v1"
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/pathology:/data/pathology

pip3 install scipy 2>/dev/null

cd ~/CLAM

# ── Fold 0 test slides only ──
python3 generate_heatmaps_standalone.py \
    --ckpt_path /data/pathology/projects/lohith/clam_results/bone_sex_clam_sb_virchow2_s1/s_0_checkpoint.pt \
    --fold_split ~/CLAM/splits/bone_sex_classification/splits_0.csv \
    --save_dir /data/pathology/projects/lohith/heatmaps/fold_0_inferno_a0.4 \
    --vis_level 3 \
    --alpha 0.4 \
    --cmap inferno \
    --top_k 15 \
    --device cpu
