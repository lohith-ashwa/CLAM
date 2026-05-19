#!/bin/bash
#SBATCH --job-name=heatmap
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/pathology/projects/lohith/logs/slurm-heatmap_bug-%j.out
#SBATCH --error=/data/pathology/projects/lohith/logs/slurm-heatmap_bug-%j.err
#SBATCH --no-container-entrypoint
#SBATCH --container-image="dockerdex.umcn.nl:5005/lohith-ashwa/bone:v1"
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/pathology:/data/pathology

cd ~/CLAM
python3 -c "
import h5py
import numpy as np
import glob

# Check a blockmap file
bmap_files = glob.glob('/data/pathology/projects/lohith/heatmaps/heatmap_raw_results/bone_sex_heatmaps_fold0/*/*_blockmap.h5')
print(f'Found {len(bmap_files)} blockmap files')

for f in bmap_files[:3]:
    print(f'\n--- {f} ---')
    with h5py.File(f, 'r') as h:
        scores = h['attention_scores'][:]
        coords = h['coords'][:]
        print(f'Scores shape: {scores.shape}')
        print(f'Scores min: {scores.min():.6f}, max: {scores.max():.6f}, mean: {scores.mean():.6f}, std: {scores.std():.6f}')
        print(f'Unique scores (first 10): {np.unique(scores.flatten())[:10]}')
        print(f'Coords shape: {coords.shape}')
        print(f'Coords range: x=[{coords[:,0].min()}, {coords[:,0].max()}], y=[{coords[:,1].min()}, {coords[:,1].max()}]')

# Also check the h5 features file
feat_files = glob.glob('/data/pathology/projects/lohith/heatmaps/heatmap_raw_results/bone_sex_heatmaps_fold0/*/*[!blockmap].h5')
feat_files = [f for f in feat_files if 'blockmap' not in f and 'roi' not in f]
for f in feat_files[:2]:
    print(f'\n--- {f} ---')
    with h5py.File(f, 'r') as h:
        print(f'Keys: {list(h.keys())}')
        if 'features' in h:
            feats = h['features'][:]
            print(f'Features shape: {feats.shape}, dtype: {feats.dtype}')
            print(f'Features min: {feats.min():.4f}, max: {feats.max():.4f}, mean: {feats.mean():.4f}')
            # Check for NaN/Inf
            print(f'NaN count: {np.isnan(feats).sum()}, Inf count: {np.isinf(feats).sum()}')
"
