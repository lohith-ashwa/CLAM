"""
5-Fold Cross-Validation Evaluation for Bone Sex Classification (CLAM-SB + Virchow2)
====================================================================================
Key design principle: AUC is computed WITHIN each fold (same model, comparable
probabilities), then averaged across folds. Probabilities are NEVER pooled across
folds since each fold has a different model with different calibration scale.

Outputs:
  - roc_curves_5fold.png        : 5 individual + mean ROC curve on same plot
  - confusion_matrices_5fold.png: per-fold confusion matrices
  - per_fold_metrics.png        : bar chart of per-fold AUC and accuracy
  - prob_distributions_5fold.png: per-fold probability strip plots
  - metrics_summary_5fold.csv   : per-fold + mean ± std metrics
  - patient_level_results_5fold.csv

Run from ~/CLAM/:
    python evaluate_5fold.py --eval_dir eval_results/EVAL_bone_sex_eval \
                             --save_dir /data/pathology/projects/lohith/evaluation/5fold_best
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

MALE_COLOR    = '#2E86C1'
FEMALE_COLOR  = '#C0392B'
CORRECT_COLOR = '#27AE60'
WRONG_COLOR   = '#E74C3C'
FOLD_COLORS   = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_slide_id(slide_id):
    m = re.match(r'BONE_(\d+)_([MF])(\d+)_(\d+)_(\d+)', slide_id)
    if not m:
        return {}
    return {
        'patient_num': int(m.group(1)),
        'case_id':     f"BONE_{m.group(1)}",
        'sex':         m.group(2),
        'age':         int(m.group(3)),
    }


def load_fold(eval_dir, fold):
    path = os.path.join(eval_dir, f'fold_{fold}.csv')
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping fold {fold}")
        return None
    df = pd.read_csv(path)
    df['fold'] = fold
    parsed = df['slide_id'].apply(parse_slide_id).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)
    return df


def aggregate_patients(fold_df):
    """Aggregate slide-level predictions to patient level within a single fold."""
    pat = fold_df.groupby('case_id').agg(
        Y        =('Y',        'first'),
        sex      =('sex',      'first'),
        age      =('age',      'first'),
        fold     =('fold',     'first'),
        n_slides =('slide_id', 'count'),
        mean_p1  =('p_1',      'mean'),
        std_p1   =('p_1',      'std'),
        median_p1=('p_1',      'median'),
    ).reset_index()

    pat['std_p1']        = pat['std_p1'].fillna(0)
    pat['Y_hat']         = (pat['mean_p1'] >= 0.5).astype(int)
    pat['patient_correct'] = pat['Y'] == pat['Y_hat']

    # Slide-level accuracy per patient
    for idx, row in pat.iterrows():
        mask   = fold_df['case_id'] == row['case_id']
        correct = (fold_df.loc[mask, 'Y'] == fold_df.loc[mask, 'Y_hat']).sum()
        pat.at[idx, 'slide_acc'] = correct / mask.sum() if mask.sum() > 0 else 0

    return pat


def compute_fold_metrics(fold_df, patient_df, fold_idx):
    """Compute slide-level and patient-level metrics for a single fold."""
    fm = {'fold': fold_idx,
          'n_slides': len(fold_df),
          'n_patients': len(patient_df)}

    for level, df, y_col, p_col, yhat_col in [
        ('slide',   fold_df,    'Y', 'p_1',    'Y_hat'),
        ('patient', patient_df, 'Y', 'mean_p1','Y_hat'),
    ]:
        y    = df[y_col].values
        p1   = df[p_col].values
        yhat = df[yhat_col].values

        if len(np.unique(y)) >= 2:
            fm[f'{level}_auc'] = roc_auc_score(y, p1)
            fpr, tpr, _        = roc_curve(y, p1)
            fm[f'{level}_fpr'] = fpr
            fm[f'{level}_tpr'] = tpr
        else:
            fm[f'{level}_auc'] = np.nan
            fm[f'{level}_fpr'] = None
            fm[f'{level}_tpr'] = None

        fm[f'{level}_acc']       = accuracy_score(y, yhat)
        fm[f'{level}_precision'] = precision_score(y, yhat, pos_label=1, zero_division=0)
        fm[f'{level}_recall_F']  = recall_score(y, yhat, pos_label=1, zero_division=0)
        fm[f'{level}_spec_M']    = recall_score(y, yhat, pos_label=0, zero_division=0)
        fm[f'{level}_f1']        = f1_score(y, yhat, pos_label=1, zero_division=0)
        cm                       = confusion_matrix(y, yhat, labels=[0, 1])
        fm[f'{level}_tn']        = cm[0, 0]
        fm[f'{level}_fp']        = cm[0, 1]
        fm[f'{level}_fn']        = cm[1, 0]
        fm[f'{level}_tp']        = cm[1, 1]

    return fm


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_roc_curves(fold_metrics_list, save_dir):
    """
    5 individual fold ROC curves + interpolated mean ± std band.
    One subplot for slide-level, one for patient-level.
    This is the canonical figure for a CV paper result.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    mean_fpr  = np.linspace(0, 1, 200)

    for level, ax, title in [
        ('slide',   axes[0], 'Slide-level ROC (5-fold CV)'),
        ('patient', axes[1], 'Patient-level ROC (5-fold CV)'),
    ]:
        tprs_interp, aucs = [], []

        for i, fm in enumerate(fold_metrics_list):
            fpr     = fm[f'{level}_fpr']
            tpr     = fm[f'{level}_tpr']
            auc_val = fm[f'{level}_auc']
            if fpr is None or np.isnan(auc_val):
                continue

            # Individual fold curve (thin, transparent)
            ax.plot(fpr, tpr,
                    color=FOLD_COLORS[i % len(FOLD_COLORS)],
                    alpha=0.45, linewidth=1.5,
                    label=f'Fold {i}  (AUC = {auc_val:.3f})')

            # Interpolate onto common FPR grid for mean curve
            interp_tpr    = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs_interp.append(interp_tpr)
            aucs.append(auc_val)

        if tprs_interp:
            mean_tpr       = np.mean(tprs_interp, axis=0)
            mean_tpr[-1]   = 1.0
            std_tpr        = np.std(tprs_interp, axis=0)
            mean_auc       = np.mean(aucs)
            std_auc        = np.std(aucs)

            ax.plot(mean_fpr, mean_tpr,
                    color='black', linewidth=2.5, zorder=5,
                    label=f'Mean  (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
            ax.fill_between(mean_fpr,
                            np.clip(mean_tpr - std_tpr, 0, 1),
                            np.clip(mean_tpr + std_tpr, 0, 1),
                            color='gray', alpha=0.18, label='± 1 std')

        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.4)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc='lower right', fontsize=8.5)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves_5fold.png'))
    plt.close()
    print('  Saved roc_curves_5fold.png')


def plot_confusion_matrices(all_fold_dfs, all_patient_dfs, save_dir):
    """2 rows × 5 cols: slide-level on top, patient-level on bottom."""
    n = len(all_fold_dfs)
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 9))

    for i, (fold_df, patient_df) in enumerate(zip(all_fold_dfs, all_patient_dfs)):
        for row, df, y_col, yhat_col, cmap, label in [
            (0, fold_df,    'Y', 'Y_hat', 'Blues', f'Fold {i} slides  (n={len(fold_df)})'),
            (1, patient_df, 'Y', 'Y_hat', 'Reds',  f'Fold {i} patients (n={len(patient_df)})'),
        ]:
            cm   = confusion_matrix(df[y_col], df[yhat_col], labels=[0, 1])
            disp = ConfusionMatrixDisplay(cm, display_labels=['Male', 'Female'])
            disp.plot(ax=axes[row, i], cmap=cmap, colorbar=False, values_format='d')
            axes[row, i].set_title(label, fontsize=10)

    plt.suptitle('Per-Fold Confusion Matrices\n(top: slide-level  |  bottom: patient-level)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices_5fold.png'))
    plt.close()
    print('  Saved confusion_matrices_5fold.png')


def plot_per_fold_metrics(fold_metrics_list, save_dir):
    """Grouped bar chart: AUC and accuracy per fold."""
    folds       = [fm['fold'] for fm in fold_metrics_list]
    slide_aucs  = [fm['slide_auc']  for fm in fold_metrics_list]
    pat_aucs    = [fm['patient_auc'] for fm in fold_metrics_list]
    slide_accs  = [fm['slide_acc']  for fm in fold_metrics_list]
    pat_accs    = [fm['patient_acc'] for fm in fold_metrics_list]

    x     = np.arange(len(folds))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric_pairs, ylabel, title in [
        (axes[0],
         [(slide_aucs, 'Slide-level', MALE_COLOR),
          (pat_aucs,   'Patient-level', FEMALE_COLOR)],
         'AUC', 'Per-Fold AUC'),
        (axes[1],
         [(slide_accs, 'Slide-level', MALE_COLOR),
          (pat_accs,   'Patient-level', FEMALE_COLOR)],
         'Accuracy', 'Per-Fold Accuracy'),
    ]:
        for j, (vals, label, color) in enumerate(metric_pairs):
            clean = [v if not np.isnan(v) else 0 for v in vals]
            ax.bar(x + (j - 0.5) * width, clean, width,
                   label=label, color=color, alpha=0.82, edgecolor='white')

            # Mean line
            valid = [v for v in vals if not np.isnan(v)]
            if valid:
                ax.axhline(np.mean(valid), color=color, linestyle='--',
                           linewidth=1.2, alpha=0.7,
                           label=f'Mean {label} = {np.mean(valid):.3f}')

        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {i}' for i in folds])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 1.08)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_fold_metrics.png'))
    plt.close()
    print('  Saved per_fold_metrics.png')


def plot_prob_distributions(all_fold_dfs, all_patient_dfs, save_dir):
    """
    One strip plot per fold.
    Small dots = individual slides, large markers = patient mean.
    Green/red edge = correct/incorrect patient prediction.
    """
    n   = len(all_fold_dfs)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 8), sharey=False)
    if n == 1:
        axes = [axes]

    for i, (fold_df, patient_df) in enumerate(zip(all_fold_dfs, all_patient_dfs)):
        ax      = axes[i]
        pat_sorted = patient_df.sort_values(['Y', 'mean_p1']).reset_index(drop=True)
        y_pos   = {row['case_id']: j for j, (_, row) in enumerate(pat_sorted.iterrows())}

        # Individual slides
        for _, srow in fold_df.iterrows():
            color = MALE_COLOR if srow['sex'] == 'M' else FEMALE_COLOR
            ax.scatter(srow['p_1'], y_pos[srow['case_id']],
                       color=color, alpha=0.15, s=14, edgecolors='none', zorder=1)

        # Patient means
        for _, row in pat_sorted.iterrows():
            color    = MALE_COLOR   if row['sex'] == 'M'                         else FEMALE_COLOR
            ec       = CORRECT_COLOR if bool(row['patient_correct'])              else WRONG_COLOR
            marker   = 'o'          if row['sex'] == 'M'                         else 's'
            ax.scatter(row['mean_p1'], y_pos[row['case_id']],
                       color=color, s=110, zorder=3,
                       edgecolors=ec, linewidths=2.0, marker=marker)

        ax.axvline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        # Separator between M and F groups
        n_m = (pat_sorted['Y'] == 0).sum()
        ax.axhline(n_m - 0.5, color='gray', linewidth=0.5, alpha=0.3)

        labels = [f"{r['case_id']} ({r['sex']}{r['age']})"
                  for _, r in pat_sorted.iterrows()]
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel('P(Female)')
        ax.set_xlim(-0.02, 1.02)

        slide_auc = fold_df['Y'].pipe(
            lambda y: roc_auc_score(y, fold_df['p_1'])
            if len(np.unique(y)) >= 2 else float('nan'))
        ax.set_title(f'Fold {i}  (slide AUC={slide_auc:.3f})', fontsize=10)

    legend_elements = [
        mpatches.Patch(facecolor=MALE_COLOR,   label='Male'),
        mpatches.Patch(facecolor=FEMALE_COLOR, label='Female'),
        plt.Line2D([0],[0], marker='o', color='w',
                   markeredgecolor=CORRECT_COLOR, markerfacecolor='gray',
                   markersize=8, markeredgewidth=2, label='Correct'),
        plt.Line2D([0],[0], marker='o', color='w',
                   markeredgecolor=WRONG_COLOR, markerfacecolor='gray',
                   markersize=8, markeredgewidth=2, label='Incorrect'),
        plt.Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='gray', markersize=8, label='Male (circle)'),
        plt.Line2D([0],[0], marker='s', color='w',
                   markerfacecolor='gray', markersize=8, label='Female (square)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.04))

    plt.suptitle('Prediction Probabilities per Fold\n'
                 '(large markers = patient mean, small dots = individual slides)',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(os.path.join(save_dir, 'prob_distributions_5fold.png'))
    plt.close()
    print('  Saved prob_distributions_5fold.png')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(eval_dir, save_dir, n_folds=5):
    os.makedirs(save_dir, exist_ok=True)

    all_fold_dfs, all_patient_dfs, all_fm = [], [], []
    all_patient_rows = []

    print(f"\n{'='*60}")
    print("5-FOLD CROSS-VALIDATION EVALUATION")
    print(f"{'='*60}")
    print(f"Note: AUC computed within each fold (same model, comparable")
    print(f"probabilities). Probabilities are NOT pooled across folds.")

    for fold in range(n_folds):
        fold_df = load_fold(eval_dir, fold)
        if fold_df is None:
            continue
        patient_df = aggregate_patients(fold_df)
        fm         = compute_fold_metrics(fold_df, patient_df, fold)

        all_fold_dfs.append(fold_df)
        all_patient_dfs.append(patient_df)
        all_fm.append(fm)
        all_patient_rows.append(patient_df)

        print(f"\n── Fold {fold}  "
              f"({fm['n_patients']} patients, {fm['n_slides']} slides) ──")
        print(f"  Slide-level   AUC: {fm['slide_auc']:.3f}   "
              f"Acc: {fm['slide_acc']:.3f}   "
              f"F1: {fm['slide_f1']:.3f}")
        pat_auc_str = (f"{fm['patient_auc']:.3f}"
                       if not np.isnan(fm['patient_auc']) else 'undefined')
        print(f"  Patient-level AUC: {pat_auc_str}   "
              f"Acc: {fm['patient_acc']:.3f}   "
              f"F1: {fm['patient_f1']:.3f}")
        print(f"  Confusion (patient) TN={fm['patient_tn']} "
              f"FP={fm['patient_fp']} "
              f"FN={fm['patient_fn']} "
              f"TP={fm['patient_tp']}")

    # ── Summary ──────────────────────────────────────────────────────────────
    slide_aucs  = [fm['slide_auc']   for fm in all_fm if not np.isnan(fm['slide_auc'])]
    pat_aucs    = [fm['patient_auc'] for fm in all_fm if not np.isnan(fm['patient_auc'])]
    slide_accs  = [fm['slide_acc']   for fm in all_fm]
    pat_accs    = [fm['patient_acc'] for fm in all_fm]

    print(f"\n{'='*60}")
    print("SUMMARY (mean ± std across folds)")
    print(f"{'='*60}")
    print(f"\n── Slide-level (conventional reporting metric) ──")
    print(f"  AUC:      {np.mean(slide_aucs):.3f} ± {np.std(slide_aucs):.3f}")
    print(f"  Accuracy: {np.mean(slide_accs):.3f} ± {np.std(slide_accs):.3f}")
    print(f"  Per-fold: {[round(a, 3) for a in slide_aucs]}")

    print(f"\n── Patient-level ──")
    if pat_aucs:
        print(f"  AUC:      {np.mean(pat_aucs):.3f} ± {np.std(pat_aucs):.3f}")
        print(f"  Accuracy: {np.mean(pat_accs):.3f} ± {np.std(pat_accs):.3f}")
        print(f"  Per-fold: {[round(a, 3) for a in pat_aucs]}")
    else:
        print("  AUC: undefined (single class in every fold's test set)")

    # ── Per-patient table ────────────────────────────────────────────────────
    all_patients = pd.concat(all_patient_rows, ignore_index=True)
    print(f"\n── Per-Patient Results ──")
    cols = ['case_id','sex','age','fold','n_slides','mean_p1','std_p1','slide_acc','patient_correct']
    detail = all_patients[cols].copy()
    detail['mean_p1']  = detail['mean_p1'].round(3)
    detail['std_p1']   = detail['std_p1'].round(3)
    detail['slide_acc']= (detail['slide_acc'] * 100).round(1)
    print(detail.sort_values('case_id').to_string(index=False))

    # ── Plots ────────────────────────────────────────────────────────────────
    print(f"\n── Generating plots ──")
    plot_roc_curves(all_fm, save_dir)
    plot_confusion_matrices(all_fold_dfs, all_patient_dfs, save_dir)
    plot_per_fold_metrics(all_fm, save_dir)
    plot_prob_distributions(all_fold_dfs, all_patient_dfs, save_dir)

    # ── Save CSVs ────────────────────────────────────────────────────────────
    rows = []
    for fm in all_fm:
        rows.append({
            'fold':            fm['fold'],
            'n_patients':      fm['n_patients'],
            'n_slides':        fm['n_slides'],
            'slide_auc':       round(fm['slide_auc'],   3),
            'slide_acc':       round(fm['slide_acc'],   3),
            'slide_precision': round(fm['slide_precision'], 3),
            'slide_recall_F':  round(fm['slide_recall_F'],  3),
            'slide_spec_M':    round(fm['slide_spec_M'],    3),
            'slide_f1':        round(fm['slide_f1'],    3),
            'patient_auc':     round(fm['patient_auc'], 3) if not np.isnan(fm['patient_auc']) else 'nan',
            'patient_acc':     round(fm['patient_acc'], 3),
            'patient_f1':      round(fm['patient_f1'],  3),
        })

    # Add mean and std rows
    metric_cols = ['slide_auc','slide_acc','slide_f1','patient_acc','patient_f1']
    mean_row = {'fold': 'mean'}
    std_row  = {'fold': 'std'}
    for col in metric_cols:
        vals = [r[col] for r in rows if r[col] != 'nan']
        mean_row[col] = round(float(np.mean(vals)), 3)
        std_row[col]  = round(float(np.std(vals)),  3)
    pat_auc_vals = [r['patient_auc'] for r in rows if r['patient_auc'] != 'nan']
    if pat_auc_vals:
        mean_row['patient_auc'] = round(float(np.mean(pat_auc_vals)), 3)
        std_row['patient_auc']  = round(float(np.std(pat_auc_vals)),  3)

    metrics_df = pd.DataFrame(rows + [mean_row, std_row])
    metrics_df.to_csv(os.path.join(save_dir, 'metrics_summary_5fold.csv'), index=False)
    print('  Saved metrics_summary_5fold.csv')

    all_patients.to_csv(os.path.join(save_dir, 'patient_level_results_5fold.csv'), index=False)
    print('  Saved patient_level_results_5fold.csv')

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {save_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='5-fold CV evaluation for bone sex classification')
    parser.add_argument('--eval_dir', type=str,
                        default='eval_results/EVAL_bone_sex_eval',
                        help='Directory containing fold_0.csv .. fold_4.csv')
    parser.add_argument('--save_dir', type=str,
                        default='/data/pathology/projects/lohith/evaluation/5fold_best',
                        help='Output directory for plots and CSVs')
    parser.add_argument('--n_folds', type=int, default=5)
    args = parser.parse_args()
    main(args.eval_dir, args.save_dir, args.n_folds)
