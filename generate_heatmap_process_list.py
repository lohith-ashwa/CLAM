"""
Generate a process list CSV for create_heatmaps.py.
Lists test slides from a specific fold with their labels.
"""
import os
import re
import glob
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4)")
parser.add_argument("--split_dir", type=str,
                    default=os.path.expanduser("~/CLAM/splits/bone_sex_classification"))
parser.add_argument("--output_dir", type=str,
                    default=os.path.expanduser("~/CLAM/heatmaps/process_lists"))
parser.add_argument("--max_slides", type=int, default=None,
                    help="Limit number of slides (for quick testing)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Read the fold split
split_df = pd.read_csv(os.path.join(args.split_dir, f"splits_{args.fold}.csv"))
test_slides = split_df["test"].dropna().tolist()

# Parse sex from filename
records = []
for slide_id in test_slides:
    match = re.match(r"BONE_(\d+)_([MF])(\d+)_(\d+)_(\d+)", slide_id)
    if match:
        sex = match.group(2)
        label = sex  # M or F
        records.append({"slide_id": slide_id, "label": label})

df = pd.DataFrame(records)

if args.max_slides:
    df = df.head(args.max_slides)

out_path = os.path.join(args.output_dir, f"heatmap_fold{args.fold}.csv")
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} test slides to {out_path}")
print(df.head(10))
