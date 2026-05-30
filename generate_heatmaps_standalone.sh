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
#python3 generate_heatmaps_standalone.py \
#    --ckpt_path /data/pathology/projects/lohith/clam_results/bone_sex_clam_sb_virchow2_s1/s_0_checkpoint.pt \
#    --fold_split ~/CLAM/splits/bone_sex_classification/splits_0.csv \
#    --save_dir /data/pathology/projects/lohith/heatmaps/fold_0_inferno_a0.4 \
#    --vis_level 3 \
#    --alpha 0.4 \
#    --cmap inferno \
#    --top_k 15 \
#    --device cpu

CKPT_BASE="/data/pathology/projects/lohith/clam_results/bone_sex_clam_sb_virchow2_6_s1"
H5_DIR="/data/pathology/projects/lohith/clam_data/h5_files"
WSI_DIR="/data/pa_cpgarchive/archives/bone/OsteoVision/public/images"
SAVE_BASE="/data/pathology/projects/lohith/heatmaps/selected"
 
COMMON_ARGS="--h5_dir $H5_DIR --wsi_dir $WSI_DIR --wsi_ext .mrxs \
             --model_type clam_sb --embed_dim 2560 \
             --vis_level 3 --alpha 0.4 --cmap inferno --top_k 15 --device cpu"

# ── BONE_6 M63 100% → fold 5 ──
echo "=== BONE_6 (fold 5) ==="
python3 generate_heatmaps_standalone.py $COMMON_ARGS \
    --ckpt_path $CKPT_BASE/s_5_checkpoint.pt \
    --save_dir $SAVE_BASE/BONE_6_fold5 \
    --slide_ids \
        BONE_6_M63_1_1 BONE_6_M63_1_2 BONE_6_M63_1_3 \
        BONE_6_M63_2_1 BONE_6_M63_2_2 BONE_6_M63_2_3 \
        BONE_6_M63_3_1 BONE_6_M63_3_2 BONE_6_M63_3_3 \
        BONE_6_M63_4_1 BONE_6_M63_4_2 BONE_6_M63_4_3 \
        BONE_6_M63_5_1 BONE_6_M63_5_2 \
        BONE_6_M63_6_1 BONE_6_M63_6_2 BONE_6_M63_6_3 \
        BONE_6_M63_7_1 BONE_6_M63_7_2 BONE_6_M63_7_3 \
        BONE_6_M63_8_1 BONE_6_M63_8_2 BONE_6_M63_8_3
 
# ── BONE_17 F52 100% → fold 14 ──
echo "=== BONE_17 (fold 14) ==="
python3 generate_heatmaps_standalone.py $COMMON_ARGS \
    --ckpt_path $CKPT_BASE/s_14_checkpoint.pt \
    --save_dir $SAVE_BASE/BONE_17_fold14 \
    --slide_ids \
        BONE_17_F52_1_1 BONE_17_F52_1_2 BONE_17_F52_1_3 \
        BONE_17_F52_2_1 BONE_17_F52_2_2 BONE_17_F52_2_3 \
        BONE_17_F52_3_1 BONE_17_F52_3_2 BONE_17_F52_3_3 \
        BONE_17_F52_4_1 BONE_17_F52_4_2 BONE_17_F52_4_3 \
        BONE_17_F52_5_1 BONE_17_F52_5_2 BONE_17_F52_5_3 \
        BONE_17_F52_6_1 BONE_17_F52_6_2 BONE_17_F52_6_3 \
        BONE_17_F52_7_1 BONE_17_F52_7_2 BONE_17_F52_7_3
 
# ── BONE_16 F63 100% → fold 13 ──
echo "=== BONE_16 (fold 13) ==="
python3 generate_heatmaps_standalone.py $COMMON_ARGS \
    --ckpt_path $CKPT_BASE/s_13_checkpoint.pt \
    --save_dir $SAVE_BASE/BONE_16_fold13 \
    --slide_ids \
        BONE_16_F63_1_1 BONE_16_F63_1_2 \
        BONE_16_F63_2_1 BONE_16_F63_2_2 BONE_16_F63_2_3 \
        BONE_16_F63_3_1 BONE_16_F63_3_2 BONE_16_F63_3_3 BONE_16_F63_3_4 \
        BONE_16_F63_4_1 BONE_16_F63_4_2 \
        BONE_16_F63_5_1 \
        BONE_16_F63_6_1 BONE_16_F63_6_2 BONE_16_F63_6_3 \
        BONE_16_F63_7_1 BONE_16_F63_7_2 BONE_16_F63_7_3
 
# ── BONE_9 M72 0% → fold 8 ──
echo "=== BONE_9 (fold 8) ==="
python3 generate_heatmaps_standalone.py $COMMON_ARGS \
    --ckpt_path $CKPT_BASE/s_8_checkpoint.pt \
    --save_dir $SAVE_BASE/BONE_9_fold8 \
    --slide_ids \
        BONE_9_M72_1_1 BONE_9_M72_1_2 BONE_9_M72_1_3 BONE_9_M72_1_4 \
        BONE_9_M72_2_1 BONE_9_M72_2_2 BONE_9_M72_2_3 BONE_9_M72_2_4 \
        BONE_9_M72_4_1 BONE_9_M72_4_2 \
        BONE_9_M72_5_1 BONE_9_M72_5_2 BONE_9_M72_5_3 BONE_9_M72_5_4 \
        BONE_9_M72_6_1 BONE_9_M72_6_2 \
        BONE_9_M72_7_1 BONE_9_M72_7_2 BONE_9_M72_7_3
 
# ── BONE_1 F51 0% → fold 0 ──
echo "=== BONE_1 (fold 0) ==="
python3 generate_heatmaps_standalone.py $COMMON_ARGS \
    --ckpt_path $CKPT_BASE/s_0_checkpoint.pt \
    --save_dir $SAVE_BASE/BONE_1_fold0 \
    --slide_ids \
        BONE_1_F51_1_1 BONE_1_F51_1_2 \
        BONE_1_F51_2_1 BONE_1_F51_2_2 BONE_1_F51_2_3 BONE_1_F51_2_4
 
# ── BONE_3 F56 0% → fold 2 ──
echo "=== BONE_3 (fold 2) ==="
python3 generate_heatmaps_standalone.py $COMMON_ARGS \
    --ckpt_path $CKPT_BASE/s_2_checkpoint.pt \
    --save_dir $SAVE_BASE/BONE_3_fold2 \
    --slide_ids \
        BONE_3_F56_1_1 BONE_3_F56_1_2 \
        BONE_3_F56_8_1 BONE_3_F56_8_2 \
        BONE_3_F56_9_1 BONE_3_F56_9_2 \
        BONE_3_F56_11_1 BONE_3_F56_11_2 BONE_3_F56_11_3
 
# ── BONE_11 M72 25% → fold 9 ──
echo "=== BONE_11 (fold 9) ==="
python3 generate_heatmaps_standalone.py $COMMON_ARGS \
    --ckpt_path $CKPT_BASE/s_9_checkpoint.pt \
    --save_dir $SAVE_BASE/BONE_11_fold9 \
    --slide_ids \
        BONE_11_M72_2_1 BONE_11_M72_2_2 BONE_11_M72_2_3 \
        BONE_11_M72_3_1 BONE_11_M72_3_2 BONE_11_M72_3_3 \
        BONE_11_M72_4_1 BONE_11_M72_4_2 BONE_11_M72_4_3 \
        BONE_11_M72_5_1 BONE_11_M72_5_2 BONE_11_M72_5_3 BONE_11_M72_5_4 \
        BONE_11_M72_6_1 BONE_11_M72_6_2 BONE_11_M72_6_3 BONE_11_M72_6_4 BONE_11_M72_6_5 \
        BONE_11_M72_7_1 BONE_11_M72_7_2 BONE_11_M72_7_3 \
        BONE_11_M72_8_1 BONE_11_M72_8_2 BONE_11_M72_8_3
 
# ── BONE_5 M59 52% → fold 4 ──
echo "=== BONE_5 (fold 4) ==="
python3 generate_heatmaps_standalone.py $COMMON_ARGS \
    --ckpt_path $CKPT_BASE/s_4_checkpoint.pt \
    --save_dir $SAVE_BASE/BONE_5_fold4 \
    --slide_ids \
        BONE_5_M59_1_1 BONE_5_M59_1_2 BONE_5_M59_1_3 BONE_5_M59_1_4 \
        BONE_5_M59_2_1 BONE_5_M59_2_2 BONE_5_M59_2_3 BONE_5_M59_2_4 BONE_5_M59_2_5 \
        BONE_5_M59_3_1 BONE_5_M59_3_2 BONE_5_M59_3_3 BONE_5_M59_3_4 BONE_5_M59_3_5 BONE_5_M59_3_6 \
        BONE_5_M59_4_1 BONE_5_M59_4_2 BONE_5_M59_4_3 \
        BONE_5_M59_5_1 BONE_5_M59_5_2 BONE_5_M59_5_3 \
        BONE_5_M59_6_1 BONE_5_M59_6_2 BONE_5_M59_6_3 BONE_5_M59_6_4 BONE_5_M59_6_5 BONE_5_M59_6_6
 
# ── BONE_12 F69 54% → fold 10 ──
echo "=== BONE_12 (fold 10) ==="
python3 generate_heatmaps_standalone.py $COMMON_ARGS \
    --ckpt_path $CKPT_BASE/s_10_checkpoint.pt \
    --save_dir $SAVE_BASE/BONE_12_fold10 \
    --slide_ids \
        BONE_12_F69_1_1 BONE_12_F69_1_2 BONE_12_F69_1_3 \
        BONE_12_F69_2_1 BONE_12_F69_2_2 BONE_12_F69_2_3 \
        BONE_12_F69_3_1 BONE_12_F69_3_2 BONE_12_F69_3_3 \
        BONE_12_F69_4_1 BONE_12_F69_4_2 BONE_12_F69_4_3 \
        BONE_12_F69_5_1 BONE_12_F69_5_2 BONE_12_F69_5_3 \
        BONE_12_F69_6_1 BONE_12_F69_6_2 BONE_12_F69_6_3 \
        BONE_12_F69_7_1 BONE_12_F69_7_2 BONE_12_F69_7_3 \
        BONE_12_F69_8_1 BONE_12_F69_8_2 BONE_12_F69_8_3
 
echo "=== All done ==="
