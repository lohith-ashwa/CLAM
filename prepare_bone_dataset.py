"""
Generate dataset CSV and 5-fold cross-validation splits for bone sex classification.
Parses filenames like BONE_15_M71_5_1 to extract patient ID and sex.
"""
import os
import glob
import re
import pandas as pd
import numpy as np

# ── Configuration ──
PT_DIR = "/data/pathology/projects/lohith/clam_data/pt_files"
CSV_DIR = os.path.expanduser("~/CLAM/dataset_csv")
SPLIT_DIR = os.path.expanduser("~/CLAM/splits/bone_sex_classification")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

# ── Parse filenames ──
pt_files = sorted(glob.glob(os.path.join(PT_DIR, "BONE_*.pt")))
print(f"Found {len(pt_files)} PT files")

records = []
for pt_path in pt_files:
    fname = os.path.splitext(os.path.basename(pt_path))[0]
    # Parse: BONE_[patient#]_[sex+age]_[block#]_[slice#]
    match = re.match(r"BONE_(\d+)_([MF])(\d+)_(\d+)_(\d+)", fname)
    if not match:
        print(f"WARNING: could not parse {fname}, skipping")
        continue
    patient_num = int(match.group(1))
    sex = match.group(2)
    age = int(match.group(3))
    case_id = f"BONE_{patient_num}"
    records.append({
        "slide_id": fname,
        "case_id": case_id,
        "patient_num": patient_num,
        "sex": sex,
        "age": age,
        "label": sex  # M or F
    })

df = pd.DataFrame(records)
print(f"\nParsed {len(df)} slides from {df['case_id'].nunique()} patients")
print(f"Sex distribution:\n{df.groupby('case_id')['sex'].first().value_counts()}")
print(f"\nSlides per patient:\n{df['case_id'].value_counts().to_string()}")

# ── Save dataset CSV ──
csv_path = os.path.join(CSV_DIR, "bone_sex_classification.csv")
df[["slide_id", "case_id", "label"]].to_csv(csv_path, index=False)
print(f"\nSaved dataset CSV to {csv_path}")

# ── Define 5-fold patient-level splits ──
# Fold definitions (test patients per fold)
fold_test_patients = {
    0: [13, 2, 16],   # Fold 1
    1: [5, 3, 12],    # Fold 2
    2: [6, 1, 8],     # Fold 3
    3: [4, 15, 17],   # Fold 4
    4: [9, 11, 7],    # Fold 5
}

all_patients = sorted(df["patient_num"].unique())
print(f"\nAll patients: {all_patients}")

for fold_idx, test_nums in fold_test_patients.items():
    test_cases = {f"BONE_{n}" for n in test_nums}
    train_val_cases = [c for c in df["case_id"].unique() if c not in test_cases]

    # Split train_val into train (10 patients) and val (2 patients, ~1M + 1F)
    train_val_df = df[df["case_id"].isin(train_val_cases)].groupby("case_id")["sex"].first()

    # Pick 1 male and 1 female for validation
    males = [c for c, s in train_val_df.items() if s == "M"]
    females = [c for c, s in train_val_df.items() if s == "F"]

    np.random.seed(fold_idx + 42)
    val_m = np.random.choice(males, 1).tolist()
    val_f = np.random.choice(females, 1).tolist()
    val_cases = set(val_m + val_f)
    train_cases = set(train_val_cases) - val_cases

    # Get slide IDs for each split
    train_slides = df[df["case_id"].isin(train_cases)]["slide_id"].tolist()
    val_slides = df[df["case_id"].isin(val_cases)]["slide_id"].tolist()
    test_slides = df[df["case_id"].isin(test_cases)]["slide_id"].tolist()

    print(f"\nFold {fold_idx}:")
    print(f"  Train: {len(train_slides)} slides from {len(train_cases)} patients ({train_cases})")
    print(f"  Val:   {len(val_slides)} slides from {len(val_cases)} patients ({val_cases})")
    print(f"  Test:  {len(test_slides)} slides from {len(test_cases)} patients ({test_cases})")

    # Create splits CSV in CLAM format
    max_len = max(len(train_slides), len(val_slides), len(test_slides))
    split_df = pd.DataFrame({
        "train": pd.Series(train_slides),
        "val": pd.Series(val_slides),
        "test": pd.Series(test_slides),
    })
    split_path = os.path.join(SPLIT_DIR, f"splits_{fold_idx}.csv")
    split_df.to_csv(split_path, index=True)
    print(f"  Saved to {split_path}")

print("\nDone! All splits saved.")
