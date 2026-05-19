"""
Convert AtlasPatch H5 files to CLAM-compatible format.
Creates:
  - h5_files/<slide_id>.h5  with datasets 'features' and 'coords'
  - pt_files/<slide_id>.pt  with just the feature tensor
"""
import os
import glob
import h5py
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

# ── Configuration ──
ATLASPATCH_DIR = "/data/pathology/projects/lohith/outputs/patches"
CLAM_H5_DIR    = "/data/pathology/projects/lohith/clam_data/h5_files"
CLAM_PT_DIR    = "/data/pathology/projects/lohith/clam_data/pt_files"
FEATURE_KEY    = "features/virchow_v2"
NUM_WORKERS    = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

os.makedirs(CLAM_H5_DIR, exist_ok=True)
os.makedirs(CLAM_PT_DIR, exist_ok=True)


def convert_one(src_path: str) -> tuple[str, str]:
    """Convert a single AtlasPatch H5 file."""
    slide_id = os.path.splitext(os.path.basename(src_path))[0]
    dst_h5   = os.path.join(CLAM_H5_DIR, f"{slide_id}.h5")
    dst_pt   = os.path.join(CLAM_PT_DIR, f"{slide_id}.pt")

    if os.path.exists(dst_h5) and os.path.exists(dst_pt):
        return ("skipped", f"{slide_id}: already exists")

    try:
        with h5py.File(src_path, "r") as f:
            if FEATURE_KEY not in f:
                return ("skipped", f"{slide_id}: missing key {FEATURE_KEY}")
            features    = f[FEATURE_KEY][:].astype(np.float32, copy=False)  # (N, 2560)
            coords_full = f["coords"][:]                                    # (N, 5)
            coords      = coords_full[:, :2].astype(np.int32)               # (N, 2)
            patch_size  = int(f.attrs.get("patch_size", 224))

        with h5py.File(dst_h5, "w") as out:
            out.create_dataset("features", data=features)
            coord_dset = out.create_dataset("coords", data=coords)
            coord_dset.attrs["patch_size"]  = patch_size
            coord_dset.attrs["patch_level"] = 0

        torch.save(torch.from_numpy(features), dst_pt)
        return ("converted", slide_id)

    except Exception as e:
        return ("skipped", f"{slide_id}: ERROR — {e}")


def main():
    h5_files = sorted(glob.glob(os.path.join(ATLASPATCH_DIR, "*.h5")))
    print(f"Found {len(h5_files)} AtlasPatch H5 files")
    print(f"Using {NUM_WORKERS} workers\n")

    converted, skipped = 0, 0
    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap_unordered(convert_one, h5_files),
            total=len(h5_files)
        ))

    for status, message in results:
        if status == "converted":
            converted += 1
        else:
            skipped += 1
            if "already exists" not in message:
                print(f"  SKIPPED {message}")

    print(f"\nDone! Converted: {converted}, Skipped: {skipped}")
    print(f"H5 files → {CLAM_H5_DIR}")
    print(f"PT files → {CLAM_PT_DIR}")


if __name__ == "__main__":
    main()
