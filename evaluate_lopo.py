"""
LOPO Evaluation for Bone Sex Classification (CLAM-SB + Virchow2)
================================================================
Pools predictions across 15 leave-one-patient-out folds and computes:
  - Patient-level metrics (primary)
  - Slide-level metrics (secondary)
  - Bootstrap confidence intervals
  - ROC curves, confusion matrices, probability distributions
  - Intra-patient consistency analysis
  - Age-stratified analysis

Run from ~/CLAM/:
    python evaluate_lopo.py --eval_dir eval_results/EVAL_bone_sex_eval_5 \
                            --save_dir /data/pathology/projects/lohith/lopo_evaluation
"""

import os
import re
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from scipy import stats

warnings.filterwarnings('ignore')

# ── Plotting defaults ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})
MALE_COLOR = '#2E86C1'
FEMALE_COLOR = '#C0392B'
CORRECT_COLOR = '#27AE60'
WRONG_COLOR = '#E74C3C'


def parse_slide_id(slide_id):
    m = re.match(r'BONE_(\d+)_([MF])(\d+)_(\d+)_(\d+)', slide_id)
    if not m:
        return None
    return {
        'patient_num': int(m.group(1)),
        'case_id': f"BONE_{m.group(1)}",
        'sex': m.group(2),
        'age': int(m.group(3)),
        'block': int(m.group(4)),
        'slice': int(m.group(5)),
    }


def load_all_folds(eval_dir, n_folds=15):
    all_rows = []
    for fold in range(n_folds):
        csv_path = os.path.join(eval_dir, f'fold_{fold}.csv')
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        df['fold'] = fold
        parsed = df['slide_id'].apply(parse_slide_id).apply(pd.Series)
        df = pd.concat([df, parsed], axis=1)
        all_rows.append(df)
    combined = pd.concat(all_rows, ignore_index=True)
    print(f"Loaded {len(combined)} slide predictions across {combined['fold'].nunique()} folds")
    print(f"  Patients: {combined['case_id'].nunique()}")
    print(f"  Males: {(combined.groupby('case_id')['sex'].first() == 'M').sum()}")
    print(f"  Females: {(combined.groupby('case_id')['sex'].first() == 'F').sum()}")
    return combined


def bootstrap_auc(y_true, y_prob, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    aucs = np.array(aucs)
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(aucs, [100 * alpha, 100 * (1 - alpha)])
    return np.mean(aucs), lo, hi


def compute_metrics(y_true, y_prob, y_pred, label, n_boot=1000):
    metrics = {'level': label}
    n = len(y_true)
    metrics['n'] = n
    if len(np.unique(y_true)) < 2:
        metrics['auc'] = np.nan
        metrics['auc_ci_lo'] = np.nan
        metrics['auc_ci_hi'] = np.nan
    else:
        auc_mean, auc_lo, auc_hi = bootstrap_auc(y_true, y_prob, n_boot=n_boot)
        metrics['auc'] = roc_auc_score(y_true, y_prob)
        metrics['auc_boot_mean'] = auc_mean
        metrics['auc_ci_lo'] = auc_lo
        metrics['auc_ci_hi'] = auc_hi
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_F'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['recall_F'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['specificity_M'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics['tn'] = cm[0, 0]
    metrics['fp'] = cm[0, 1]
    metrics['fn'] = cm[1, 0]
    metrics['tp'] = cm[1, 1]
    return metrics


def plot_roc_curves(slide_df, patient_df, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Slide-level ROC
    ax = axes[0]
    fpr, tpr, _ = roc_curve(slide_df['Y'].values, slide_df['p_1'].values)
    auc_val = roc_auc_score(slide_df['Y'].values, slide_df['p_1'].values)
    ax.plot(fpr, tpr, color=MALE_COLOR, lw=2, label=f'AUC = {auc_val:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Slide-level ROC (secondary)')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')

    # Patient-level ROC
    ax = axes[1]
    fpr, tpr, _ = roc_curve(patient_df['Y'].values, patient_df['mean_p1'].values)
    auc_val = roc_auc_score(patient_df['Y'].values, patient_df['mean_p1'].values)
    ax.plot(fpr, tpr, color=FEMALE_COLOR, lw=2, label=f'AUC = {auc_val:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4)

    # Mark individual patients on the curve
    thresholds = np.linspace(0, 1, 200)
    for _, row in patient_df.iterrows():
        pred_correct = (row['mean_p1'] >= 0.5) == row['Y']
        color = CORRECT_COLOR if pred_correct else WRONG_COLOR
        marker = 'o' if row['sex'] == 'M' else 's'
        # Place patient label near a sensible spot — use their probability
        # Just annotate the ROC isn't ideal; we'll do it on the probability plot instead

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Patient-level ROC (primary)')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'))
    plt.close()


def plot_confusion_matrices(slide_df, patient_df, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Slide-level
    cm_slide = confusion_matrix(slide_df['Y'], slide_df['Y_hat'], labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm_slide, display_labels=['Male', 'Female'])
    disp.plot(ax=axes[0], cmap='Blues', colorbar=False, values_format='d')
    axes[0].set_title(f'Slide-level (n={len(slide_df)})')

    # Patient-level
    cm_pat = confusion_matrix(patient_df['Y'], patient_df['Y_hat'], labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm_pat, display_labels=['Male', 'Female'])
    disp.plot(ax=axes[1], cmap='Reds', colorbar=False, values_format='d')
    axes[1].set_title(f'Patient-level (n={len(patient_df)})')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'))
    plt.close()


def plot_probability_distribution(slide_df, patient_df, save_dir):
    """Strip plot with patient-level means and individual slide probabilities."""
    fig, ax = plt.subplots(figsize=(8, 6))

    patient_df_sorted = patient_df.sort_values(['Y', 'mean_p1']).reset_index(drop=True)
    y_positions = {}
    for i, (_, row) in enumerate(patient_df_sorted.iterrows()):
        y_positions[row['case_id']] = i

    # Plot individual slides as small transparent points
    for _, slide_row in slide_df.iterrows():
        y_pos = y_positions[slide_row['case_id']]
        color = MALE_COLOR if slide_row['sex'] == 'M' else FEMALE_COLOR
        ax.scatter(slide_row['p_1'], y_pos, color=color, alpha=0.15, s=12, zorder=1, edgecolors='none')

    # Plot patient means as larger points
    for _, row in patient_df_sorted.iterrows():
        y_pos = y_positions[row['case_id']]
        color = MALE_COLOR if row['sex'] == 'M' else FEMALE_COLOR
        pred_correct = (row['mean_p1'] >= 0.5) == row['Y']
        edgecolor = CORRECT_COLOR if pred_correct else WRONG_COLOR
        ax.scatter(row['mean_p1'], y_pos, color=color, s=100, zorder=3,
                   edgecolors=edgecolor, linewidths=2.0,
                   marker='o' if row['sex'] == 'M' else 's')

    # Decision boundary
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    ax.text(0.48, len(patient_df_sorted) - 0.3, '← Male', ha='right', fontsize=9, color='gray')
    ax.text(0.52, len(patient_df_sorted) - 0.3, 'Female →', ha='left', fontsize=9, color='gray')

    # Separator between M and F groups
    n_males = (patient_df_sorted['Y'] == 0).sum()
    ax.axhline(y=n_males - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    labels = [f"{row['case_id']} ({row['sex']}{row['age']})" for _, row in patient_df_sorted.iterrows()]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('P(Female)')
    ax.set_title('Prediction Probabilities per Patient\n(large markers = patient mean, small dots = individual slides)')
    ax.set_xlim(-0.02, 1.02)

    legend_elements = [
        mpatches.Patch(facecolor=MALE_COLOR, label='Male'),
        mpatches.Patch(facecolor=FEMALE_COLOR, label='Female'),
        plt.Line2D([0], [0], marker='o', color='w', markeredgecolor=CORRECT_COLOR,
                   markerfacecolor='gray', markersize=8, markeredgewidth=2, label='Correct'),
        plt.Line2D([0], [0], marker='o', color='w', markeredgecolor=WRONG_COLOR,
                   markerfacecolor='gray', markersize=8, markeredgewidth=2, label='Incorrect'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'probability_distribution.png'))
    plt.close()


def plot_intra_patient_consistency(patient_df, save_dir):
    """Box plot of slide-level p_1 std within each patient, grouped by sex."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    males = patient_df[patient_df['sex'] == 'M']['std_p1'].values
    females = patient_df[patient_df['sex'] == 'F']['std_p1'].values

    bp = ax.boxplot([males, females], labels=['Male', 'Female'],
                    patch_artist=True, widths=0.4,
                    medianprops={'color': 'black', 'linewidth': 1.5})
    bp['boxes'][0].set_facecolor(MALE_COLOR + '44')
    bp['boxes'][0].set_edgecolor(MALE_COLOR)
    bp['boxes'][1].set_facecolor(FEMALE_COLOR + '44')
    bp['boxes'][1].set_edgecolor(FEMALE_COLOR)

    # Overlay individual points
    for i, (vals, color) in enumerate([(males, MALE_COLOR), (females, FEMALE_COLOR)]):
        x_jitter = 1 + i + np.random.RandomState(42).uniform(-0.1, 0.1, len(vals))
        ax.scatter(x_jitter, vals, color=color, alpha=0.7, s=40, zorder=3, edgecolors='white', linewidths=0.5)

    ax.set_ylabel('Std of P(Female) across slides')
    ax.set_title('Intra-Patient Slide Consistency\n(lower = more consistent predictions)')

    # Annotate means
    ax.text(1, np.mean(males) + 0.01, f'mean={np.mean(males):.3f}', ha='center', fontsize=9, color=MALE_COLOR)
    ax.text(2, np.mean(females) + 0.01, f'mean={np.mean(females):.3f}', ha='center', fontsize=9, color=FEMALE_COLOR)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'intra_patient_consistency.png'))
    plt.close()


def plot_age_vs_prediction(patient_df, save_dir):
    """Scatter plot of age vs mean P(Female), colored by true sex."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for _, row in patient_df.iterrows():
        color = MALE_COLOR if row['sex'] == 'M' else FEMALE_COLOR
        pred_correct = (row['mean_p1'] >= 0.5) == row['Y']
        edgecolor = CORRECT_COLOR if pred_correct else WRONG_COLOR
        marker = 'o' if row['sex'] == 'M' else 's'
        ax.scatter(row['age'], row['mean_p1'], color=color, s=120, marker=marker,
                   edgecolors=edgecolor, linewidths=2, zorder=3)
        ax.annotate(row['case_id'].replace('BONE_', ''), (row['age'], row['mean_p1']),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color='gray')

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.4)
    ax.set_xlabel('Patient Age')
    ax.set_ylabel('Mean P(Female)')
    ax.set_title('Age vs Model Prediction')

    # Separate trend lines per sex
    for sex, color in [('M', MALE_COLOR), ('F', FEMALE_COLOR)]:
        sub = patient_df[patient_df['sex'] == sex]
        if len(sub) >= 3:
            z = np.polyfit(sub['age'], sub['mean_p1'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(sub['age'].min() - 2, sub['age'].max() + 2, 50)
            ax.plot(x_line, p(x_line), color=color, linestyle=':', alpha=0.5, linewidth=1.5)
            slope = z[0]
            r, pval = stats.pearsonr(sub['age'], sub['mean_p1'])
            ax.text(sub['age'].max() + 1, p(sub['age'].max() + 1),
                    f'r={r:.2f}, p={pval:.2f}', fontsize=8, color=color, alpha=0.8)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MALE_COLOR, markersize=10, label='Male'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=FEMALE_COLOR, markersize=10, label='Female'),
        plt.Line2D([0], [0], marker='o', color='w', markeredgecolor=CORRECT_COLOR,
                   markerfacecolor='lightgray', markersize=10, markeredgewidth=2, label='Correct'),
        plt.Line2D([0], [0], marker='o', color='w', markeredgecolor=WRONG_COLOR,
                   markerfacecolor='lightgray', markersize=10, markeredgewidth=2, label='Incorrect'),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'age_vs_prediction.png'))
    plt.close()


def plot_per_patient_summary(patient_df, save_dir):
    """Horizontal bar chart showing accuracy per patient with slide counts."""
    patient_df_sorted = patient_df.sort_values('slide_acc', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = [MALE_COLOR if s == 'M' else FEMALE_COLOR for s in patient_df_sorted['sex']]
    edge_colors = [CORRECT_COLOR if c else WRONG_COLOR for c in patient_df_sorted['patient_correct']]

    bars = ax.barh(range(len(patient_df_sorted)), patient_df_sorted['slide_acc'] * 100,
                   color=colors, edgecolor=edge_colors, linewidth=1.5, height=0.65)

    ax.axvline(x=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)

    labels = []
    for _, row in patient_df_sorted.iterrows():
        correct = int(row['slide_acc'] * row['n_slides'])
        labels.append(f"{row['case_id']} ({row['sex']}{row['age']}) — {correct}/{row['n_slides']}")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Slide-level accuracy (%)')
    ax.set_title('Per-Patient Classification Accuracy')
    ax.set_xlim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_patient_accuracy.png'))
    plt.close()


def main(eval_dir, save_dir, n_folds=15):
    os.makedirs(save_dir, exist_ok=True)

    # ── Load data ──
    slide_df = load_all_folds(eval_dir, n_folds)

    # ── Patient-level aggregation ──
    patient_df = slide_df.groupby('case_id').agg(
        Y=('Y', 'first'),
        sex=('sex', 'first'),
        age=('age', 'first'),
        patient_num=('patient_num', 'first'),
        n_slides=('slide_id', 'count'),
        mean_p1=('p_1', 'mean'),
        std_p1=('p_1', 'std'),
        median_p1=('p_1', 'median'),
        min_p1=('p_1', 'min'),
        max_p1=('p_1', 'max'),
        slide_acc=('Y_hat', lambda x: (x == x.iloc[0]).mean() if len(x) > 0 else 0),
    ).reset_index()

    # Fill std NaN for patients with 1 slide
    patient_df['std_p1'] = patient_df['std_p1'].fillna(0)

    # Patient-level predictions
    patient_df['Y_hat'] = (patient_df['mean_p1'] >= 0.5).astype(int)
    patient_df['patient_correct'] = patient_df['Y'] == patient_df['Y_hat']

    # Fix slide_acc: compute directly from fold results
    for idx, row in patient_df.iterrows():
        mask = slide_df['case_id'] == row['case_id']
        correct = (slide_df.loc[mask, 'Y'] == slide_df.loc[mask, 'Y_hat']).sum()
        total = mask.sum()
        patient_df.at[idx, 'slide_acc'] = correct / total if total > 0 else 0

    print(f"\n{'='*60}")
    print("LOPO EVALUATION RESULTS")
    print(f"{'='*60}")

    # ── 1. Slide-level pooled metrics ──
    slide_metrics = compute_metrics(
        slide_df['Y'].values, slide_df['p_1'].values, slide_df['Y_hat'].values,
        'Slide-level (secondary)'
    )

    # ── 2. Patient-level pooled metrics ──
    patient_metrics = compute_metrics(
        patient_df['Y'].values, patient_df['mean_p1'].values, patient_df['Y_hat'].values,
        'Patient-level (primary)'
    )

    # Print results
    for m in [patient_metrics, slide_metrics]:
        print(f"\n── {m['level']} (n={m['n']}) ──")
        if not np.isnan(m['auc']):
            print(f"  AUC:         {m['auc']:.3f} (95% CI: {m['auc_ci_lo']:.3f}–{m['auc_ci_hi']:.3f})")
        else:
            print(f"  AUC:         undefined (single class)")
        print(f"  Accuracy:    {m['accuracy']:.3f}")
        print(f"  Precision:   {m['precision_F']:.3f}")
        print(f"  Recall (F):  {m['recall_F']:.3f}")
        print(f"  Specificity: {m['specificity_M']:.3f}")
        print(f"  F1:          {m['f1']:.3f}")
        print(f"  Confusion:   TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")

    # ── 3. Per-patient detail table ──
    print(f"\n── Per-Patient Results ──")
    detail_cols = ['case_id', 'sex', 'age', 'n_slides', 'mean_p1', 'std_p1', 'slide_acc', 'patient_correct']
    patient_detail = patient_df[detail_cols].copy()
    patient_detail['mean_p1'] = patient_detail['mean_p1'].round(3)
    patient_detail['std_p1'] = patient_detail['std_p1'].round(3)
    patient_detail['slide_acc'] = (patient_detail['slide_acc'] * 100).round(1)
    patient_detail = patient_detail.rename(columns={
        'slide_acc': 'slide_acc_%', 'patient_correct': 'correct'
    })
    patient_detail = patient_detail.sort_values('case_id')
    print(patient_detail.to_string(index=False))

    # ── 4. Intra-patient consistency ──
    print(f"\n── Intra-Patient Consistency (std of P(Female)) ──")
    male_std = patient_df[patient_df['sex'] == 'M']['std_p1']
    female_std = patient_df[patient_df['sex'] == 'F']['std_p1']
    print(f"  Males  (n={len(male_std)}): mean std = {male_std.mean():.3f} ± {male_std.std():.3f}")
    print(f"  Females (n={len(female_std)}): mean std = {female_std.mean():.3f} ± {female_std.std():.3f}")
    print(f"  Overall (n={len(patient_df)}): mean std = {patient_df['std_p1'].mean():.3f} ± {patient_df['std_p1'].std():.3f}")

    # ── 5. Age correlation ──
    print(f"\n── Age vs Prediction Analysis ──")
    for sex_label, sex_code in [('Males', 'M'), ('Females', 'F')]:
        sub = patient_df[patient_df['sex'] == sex_code]
        if len(sub) >= 3:
            r, p = stats.pearsonr(sub['age'], sub['mean_p1'])
            print(f"  {sex_label}: Pearson r = {r:.3f}, p = {p:.3f}")
        else:
            print(f"  {sex_label}: too few patients for correlation")

    # ── 6. Male-detection vs Female-detection breakdown ──
    print(f"\n── Sex-Specific Performance ──")
    males_correct = patient_df[(patient_df['sex'] == 'M') & (patient_df['patient_correct'])].shape[0]
    males_total = patient_df[patient_df['sex'] == 'M'].shape[0]
    females_correct = patient_df[(patient_df['sex'] == 'F') & (patient_df['patient_correct'])].shape[0]
    females_total = patient_df[patient_df['sex'] == 'F'].shape[0]
    print(f"  Male patients correctly identified:   {males_correct}/{males_total} ({100*males_correct/males_total:.0f}%)")
    print(f"  Female patients correctly identified:  {females_correct}/{females_total} ({100*females_correct/females_total:.0f}%)")

    # ── Generate all plots ──
    print(f"\n── Generating plots ──")
    plot_roc_curves(slide_df, patient_df, save_dir)
    print("  Saved roc_curves.png")
    plot_confusion_matrices(slide_df, patient_df, save_dir)
    print("  Saved confusion_matrices.png")
    plot_probability_distribution(slide_df, patient_df, save_dir)
    print("  Saved probability_distribution.png")
    plot_intra_patient_consistency(patient_df, save_dir)
    print("  Saved intra_patient_consistency.png")
    plot_age_vs_prediction(patient_df, save_dir)
    print("  Saved age_vs_prediction.png")
    plot_per_patient_summary(patient_df, save_dir)
    print("  Saved per_patient_accuracy.png")

    # ── Save summary CSVs ──
    metrics_df = pd.DataFrame([patient_metrics, slide_metrics])
    metrics_df.to_csv(os.path.join(save_dir, 'metrics_summary.csv'), index=False)
    print("  Saved metrics_summary.csv")

    patient_export = patient_df[[
        'case_id', 'sex', 'age', 'n_slides', 'Y', 'Y_hat',
        'mean_p1', 'median_p1', 'std_p1', 'min_p1', 'max_p1',
        'slide_acc', 'patient_correct'
    ]].sort_values('case_id')
    patient_export.to_csv(os.path.join(save_dir, 'patient_level_results.csv'), index=False)
    print("  Saved patient_level_results.csv")

    slide_export = slide_df[[
        'slide_id', 'case_id', 'sex', 'age', 'fold', 'Y', 'Y_hat', 'p_0', 'p_1'
    ]].sort_values(['case_id', 'slide_id'])
    slide_export.to_csv(os.path.join(save_dir, 'slide_level_results.csv'), index=False)
    print("  Saved slide_level_results.csv")

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {save_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LOPO evaluation for bone sex classification')
    parser.add_argument('--eval_dir', type=str,
                        default='eval_results/EVAL_bone_sex_eval_5',
                        help='Directory containing fold_X.csv files')
    parser.add_argument('--save_dir', type=str,
                        default='/data/pathology/projects/lohith/lopo_evaluation',
                        help='Output directory for plots and CSVs')
    parser.add_argument('--n_folds', type=int, default=15)
    args = parser.parse_args()
    main(args.eval_dir, args.save_dir, args.n_folds)
