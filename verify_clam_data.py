"""
Verify that all converted H5 and PT files are CLAM-compatible.
Checks:
  - Every source H5 has a corresponding H5 + PT file
  - H5 files open cleanly with required datasets
  - H5 features shape matches PT tensor shape
  - Coords are (N, 2) int, features are (N, 2560) float32
  - Feature and coord row counts match
"""
import os
import glob
import h5py
import torch
import numpy as np

ATLASPATCH_DIR = "/data/pathology/projects/lohith/outputs/patches"
CLAM_H5_DIR    = "/data/pathology/projects/lohith/clam_data/h5_files"
CLAM_PT_DIR    = "/data/pathology/projects/lohith/clam_data/pt_files"
EXPECTED_DIM   = 2560

src_files = sorted(glob.glob(os.path.join(ATLASPATCH_DIR, "*.h5")))
print(f"Source files:    {len(src_files)}")
print(f"Converted H5:    {len(glob.glob(os.path.join(CLAM_H5_DIR, '*.h5')))}")
print(f"Converted PT:    {len(glob.glob(os.path.join(CLAM_PT_DIR, '*.pt')))}\n")

errors = []
total_patches = 0
min_patches, max_patches = float("inf"), 0

for src in src_files:
    slide_id = os.path.splitext(os.path.basename(src))[0]
    h5_path  = os.path.join(CLAM_H5_DIR, f"{slide_id}.h5")
    pt_path  = os.path.join(CLAM_PT_DIR, f"{slide_id}.pt")

    # Existence
    if not os.path.exists(h5_path):
        errors.append(f"{slide_id}: H5 missing")
        continue
    if not os.path.exists(pt_path):
        errors.append(f"{slide_id}: PT missing")
        continue

    # Open and check H5
    try:
        with h5py.File(h5_path, "r") as f:
            if "features" not in f:
                errors.append(f"{slide_id}: H5 missing 'features'")
                continue
            if "coords" not in f:
                errors.append(f"{slide_id}: H5 missing 'coords'")
                continue

            feats = f["features"]
            coords = f["coords"]

            # Feature shape
            if feats.ndim != 2 or feats.shape[1] != EXPECTED_DIM:
                errors.append(f"{slide_id}: features shape {feats.shape}, expected (N, {EXPECTED_DIM})")
                continue
            if feats.dtype != np.float32:
                errors.append(f"{slide_id}: features dtype {feats.dtype}, expected float32")
                continue

            # Coord shape
            if coords.ndim != 2 or coords.shape[1] != 2:
                errors.append(f"{slide_id}: coords shape {coords.shape}, expected (N, 2)")
                continue
            if not np.issubdtype(coords.dtype, np.integer):
                errors.append(f"{slide_id}: coords dtype {coords.dtype}, expected integer")
                continue

            # Row alignment
            if feats.shape[0] != coords.shape[0]:
                errors.append(f"{slide_id}: feat rows {feats.shape[0]} != coord rows {coords.shape[0]}")
                continue

            # Coord attrs
            if "patch_size" not in coords.attrs:
                errors.append(f"{slide_id}: coords missing 'patch_size' attr")
                continue
            if "patch_level" not in coords.attrs:
                errors.append(f"{slide_id}: coords missing 'patch_level' attr")
                continue

            n_patches = feats.shape[0]
            total_patches += n_patches
            min_patches = min(min_patches, n_patches)
            max_patches = max(max_patches, n_patches)
            feat_shape = feats.shape

    except Exception as e:
        errors.append(f"{slide_id}: H5 open failed — {e}")
        continue

    # Open and check PT
    try:
        pt_tensor = torch.load(pt_path, map_location="cpu", weights_only=True)
        if not isinstance(pt_tensor, torch.Tensor):
            errors.append(f"{slide_id}: PT not a tensor")
            continue
        if tuple(pt_tensor.shape) != feat_shape:
            errors.append(f"{slide_id}: PT shape {tuple(pt_tensor.shape)} != H5 shape {feat_shape}")
            continue
        if pt_tensor.dtype != torch.float32:
            errors.append(f"{slide_id}: PT dtype {pt_tensor.dtype}, expected float32")
            continue
    except Exception as e:
        errors.append(f"{slide_id}: PT load failed — {e}")
        continue

# Report
print(f"{'='*60}")
if errors:
    print(f"FAILED: {len(errors)} errors\n")
    for e in errors:
        print(f"  {e}")
else:
    print(f"ALL {len(src_files)} SLIDES OK")
    print(f"  Total patches:     {total_patches:,}")
    print(f"  Patches per slide: min={min_patches}, max={max_patches}, mean={total_patches // len(src_files)}")
    print(f"  Feature dim:       {EXPECTED_DIM}")
