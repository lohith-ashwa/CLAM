"""
Standalone attention heatmap generator for CLAM.
Uses pre-extracted Virchow2 features — no encoder or GPU-heavy re-extraction needed.

Usage:
    python3 generate_heatmaps_standalone.py \
        --ckpt_path /path/to/s_0_checkpoint.pt \
        --h5_dir /path/to/clam_data/h5_files \
        --wsi_dir /path/to/images \
        --save_dir /path/to/heatmaps
"""
import os
import sys
import re
import argparse
import h5py
import torch
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from scipy.stats import rankdata
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.model_clam import CLAM_SB, CLAM_MB


def load_model(ckpt_path, model_type="clam_sb", embed_dim=2560, n_classes=2, drop_out=0.25):
    model_dict = {"dropout": drop_out, "n_classes": n_classes,
                  "embed_dim": embed_dim, "size_arg": "small"}
    if model_type == "clam_sb":
        model = CLAM_SB(**model_dict)
    elif model_type == "clam_mb":
        model = CLAM_MB(**model_dict)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_clean = {k.replace('.module', ''): v for k, v in ckpt.items()
                  if 'instance_loss_fn' not in k}
    model.load_state_dict(ckpt_clean, strict=True)
    model.eval()
    return model


def find_wsi(wsi_dir, slide_id, wsi_ext=".mrxs"):
    """Search for WSI handling nested folder structure."""
    # Try flat: wsi_dir/slide_id.mrxs
    flat = os.path.join(wsi_dir, slide_id + wsi_ext)
    if os.path.exists(flat):
        return flat
    # Try nested: wsi_dir/slide_id/slide_id.mrxs
    nested = os.path.join(wsi_dir, slide_id, slide_id + wsi_ext)
    if os.path.exists(nested):
        return nested
    # Recursive search as fallback
    for root, dirs, files in os.walk(wsi_dir):
        if slide_id + wsi_ext in files:
            return os.path.join(root, slide_id + wsi_ext)
    return None


def generate_heatmap(slide_id, wsi_path, h5_path, model, save_dir,
                     label_dict=None, vis_level=2, cmap_name="coolwarm",
                     alpha=0.4, top_k_save=15, device="cpu"):
    """Generate attention heatmap for a single slide."""
    import openslide

    if label_dict is None:
        label_dict = {0: "M", 1: "F"}

    # ── Load features and coords ──
    with h5py.File(h5_path, "r") as f:
        features = torch.tensor(f["features"][:], dtype=torch.float32)
        coords = f["coords"][:]  # (N, 2)
        patch_size = int(f["coords"].attrs.get("patch_size", 224))

    # ── Run CLAM inference ──
    with torch.inference_mode():
        logits, Y_prob, Y_hat, A_raw, _ = model(features.to(device))
        pred_class = Y_hat.item()
        pred_prob = Y_prob.cpu().numpy().flatten()
        attention = A_raw.cpu().numpy().flatten()

    # ── Normalize attention to percentiles [0, 100] ──
    attention_pct = rankdata(attention, 'average') / len(attention) * 100.0

    # ── Open WSI ──
    wsi = openslide.open_slide(wsi_path)
    vis_dim = wsi.level_dimensions[vis_level]
    downsample = wsi.level_downsamples[vis_level]

    # Read downsampled WSI
    img = np.array(wsi.read_region((0, 0), vis_level, vis_dim).convert("RGB"))

    # Scale coords and patch size
    sx, sy = 1.0 / downsample, 1.0 / downsample
    scaled_coords = np.column_stack([
        (coords[:, 0] * sx).astype(int),
        (coords[:, 1] * sy).astype(int)
    ])
    pw = max(1, int(patch_size * sx))
    ph = max(1, int(patch_size * sy))
    h_img, w_img = img.shape[:2]

    # ── Build attention overlay ──
    overlay = np.zeros((h_img, w_img), dtype=np.float64)
    counter = np.zeros((h_img, w_img), dtype=np.uint16)

    for i in range(len(scaled_coords)):
        x, y = scaled_coords[i]
        x_end = min(x + pw, w_img)
        y_end = min(y + ph, h_img)
        x = max(x, 0)
        y = max(y, 0)
        if x_end > x and y_end > y:
            overlay[y:y_end, x:x_end] += attention_pct[i]
            counter[y:y_end, x:x_end] += 1

    # Average overlapping regions
    mask = counter > 0
    overlay[mask] /= counter[mask]
    overlay /= 100.0  # Normalize to [0, 1]

    # Apply colormap
    colormap = cm.get_cmap(cmap_name)
    heatmap_rgba = (colormap(overlay) * 255).astype(np.uint8)[:, :, :3]

    # Blend: only where tissue patches exist
    blended = img.copy()
    blended[mask] = (alpha * heatmap_rgba[mask] + (1 - alpha) * img[mask]).astype(np.uint8)

    # ── Save results ──
    pred_label = label_dict[pred_class]
    slide_save_dir = os.path.join(save_dir, pred_label, slide_id)
    os.makedirs(slide_save_dir, exist_ok=True)

    # Heatmap overlay
    heatmap_path = os.path.join(slide_save_dir,
        f"{slide_id}_heatmap_{pred_label}_M{pred_prob[0]:.3f}_F{pred_prob[1]:.3f}.jpg")
    Image.fromarray(blended).save(heatmap_path, quality=95)

    # Heatmap on blank canvas (no tissue underneath)
    canvas = np.full_like(img, 255, dtype=np.uint8)  # white background
    canvas[mask] = heatmap_rgba[mask]
    canvas_path = os.path.join(slide_save_dir,
        f"{slide_id}_heatmap_canvas_{pred_label}.jpg")
    Image.fromarray(canvas).save(canvas_path, quality=95)

    # Original WSI (downsampled)
    orig_path = os.path.join(slide_save_dir, f"{slide_id}_original.jpg")
    Image.fromarray(img).save(orig_path, quality=95)

    # Tissue mask visualization
    mask_img = (mask.astype(np.uint8) * 255)
    mask_path = os.path.join(slide_save_dir, f"{slide_id}_tissue_mask.jpg")
    Image.fromarray(mask_img).save(mask_path, quality=95)

    # ── Save top-k and bottom-k attention patches ──
    if top_k_save > 0:
        sorted_idx = np.argsort(attention)  # ascending
        top_k_idx = sorted_idx[-top_k_save:][::-1]  # highest attention
        bot_k_idx = sorted_idx[:top_k_save]          # lowest attention

        for tag, indices in [("topk_high", top_k_idx), ("topk_low", bot_k_idx)]:
            patch_dir = os.path.join(slide_save_dir, tag)
            os.makedirs(patch_dir, exist_ok=True)
            for rank, idx in enumerate(indices):
                cx, cy = int(coords[idx, 0]), int(coords[idx, 1])
                score = attention_pct[idx]
                try:
                    patch = wsi.read_region((cx, cy), 0, (patch_size, patch_size)).convert("RGB")
                    patch.save(os.path.join(patch_dir,
                        f"{rank:02d}_x{cx}_y{cy}_attn{score:.1f}.png"))
                except Exception:
                    pass

    wsi.close()

    return {
        "slide_id": slide_id,
        "pred_class": pred_class,
        "pred_label": pred_label,
        "prob_M": pred_prob[0],
        "prob_F": pred_prob[1],
        "n_patches": len(coords),
        "attn_std": attention.std(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone CLAM attention heatmaps")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--h5_dir", type=str,
                        default="/data/pathology/projects/lohith/clam_data/h5_files")
    parser.add_argument("--wsi_dir", type=str,
                        default="/data/pa_cpgarchive/archives/bone/OsteoVision/public/images")
    parser.add_argument("--save_dir", type=str,
                        default="/data/pathology/projects/lohith/heatmaps")
    parser.add_argument("--slide_ids", nargs="+", default=None,
                        help="Specific slides to process (default: all)")
    parser.add_argument("--fold_split", type=str, default=None,
                        help="Path to splits CSV — if given, only processes test slides")
    parser.add_argument("--wsi_ext", type=str, default=".mrxs")
    parser.add_argument("--model_type", type=str, default="clam_sb")
    parser.add_argument("--embed_dim", type=int, default=2560)
    parser.add_argument("--vis_level", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--cmap", type=str, default="coolwarm")
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.ckpt_path}")
    model = load_model(args.ckpt_path, args.model_type, args.embed_dim)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Determine which slides to process
    if args.slide_ids:
        slide_ids = args.slide_ids
    elif args.fold_split:
        import pandas as pd
        split_df = pd.read_csv(args.fold_split)
        slide_ids = split_df["test"].dropna().tolist()
        print(f"Processing {len(slide_ids)} test slides from {args.fold_split}")
    else:
        slide_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(args.h5_dir)
                           if f.endswith(".h5")])

    print(f"Processing {len(slide_ids)} slides")

    # Process each slide
    results = []
    for slide_id in tqdm(slide_ids, desc="Generating heatmaps"):
        h5_path = os.path.join(args.h5_dir, f"{slide_id}.h5")
        if not os.path.exists(h5_path):
            print(f"  SKIP: H5 not found for {slide_id}")
            continue

        wsi_path = find_wsi(args.wsi_dir, slide_id, args.wsi_ext)
        if wsi_path is None:
            print(f"  SKIP: WSI not found for {slide_id}")
            continue

        try:
            result = generate_heatmap(
                slide_id=slide_id,
                wsi_path=wsi_path,
                h5_path=h5_path,
                model=model,
                save_dir=args.save_dir,
                vis_level=args.vis_level,
                cmap_name=args.cmap,
                alpha=args.alpha,
                top_k_save=args.top_k,
                device=device,
            )
            results.append(result)
            print(f"  {slide_id}: pred={result['pred_label']} "
                  f"(M={result['prob_M']:.3f}, F={result['prob_F']:.3f}), "
                  f"{result['n_patches']} patches")
        except Exception as e:
            print(f"  ERROR on {slide_id}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary CSV
    if results:
        import pandas as pd
        summary_df = pd.DataFrame(results)
        summary_path = os.path.join(args.save_dir, "heatmap_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
        print(f"Processed {len(results)}/{len(slide_ids)} slides successfully")
