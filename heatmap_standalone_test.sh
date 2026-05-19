#!/bin/bash
#SBATCH --job-name=heatmap
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/pathology/projects/lohith/logs/slurm-standalone_heatmap_test-%j.out
#SBATCH --error=/data/pathology/projects/lohith/logs/slurm-standalone_heatmap_test-%j.err
#SBATCH --no-container-entrypoint
#SBATCH --container-image="dockerdex.umcn.nl:5005/lohith-ashwa/bone:v1"
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/pathology:/data/pathology
cd ~/CLAM
python3 -c "
import h5py, torch, numpy as np
f = h5py.File('/data/pathology/projects/lohith/clam_data/h5_files/BONE_13_M47_1_1.h5','r')
feats = torch.tensor(f['features'][:])
coords = f['coords'][:]
f.close()

import sys; sys.path.insert(0,'.')
from models.model_clam import CLAM_SB
model = CLAM_SB(embed_dim=2560, n_classes=2, dropout=0.25)
ckpt = torch.load('/data/pathology/projects/lohith/clam_results/bone_sex_clam_sb_virchow2_s1/s_0_checkpoint.pt', map_location='cpu')
ckpt_clean = {k.replace('.module',''):v for k,v in ckpt.items() if 'instance_loss_fn' not in k}
model.load_state_dict(ckpt_clean)
model.eval()

with torch.inference_mode():
    logits, Y_prob, Y_hat, A_raw, _ = model(feats)
    A = A_raw.cpu().numpy().flatten()
print(f'Prediction: {Y_hat.item()}, Probs: {Y_prob.cpu().numpy().flatten()}')
print(f'Attention: min={A.min():.4f}, max={A.max():.4f}, std={A.std():.4f}')
print(f'Patches: {len(coords)}, Feature dim: {feats.shape[1]}')
print('Everything works!')
"
