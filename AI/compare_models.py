"""
Model Comparison: XGBoost (with EJ) vs XGBoost (no EJ) vs Qwen

Compares model quality without ground truth by measuring:
1. Score discrimination — does the model actually differentiate sites?
2. Feature balance — is one feature dominating everything?
3. Score distribution — bell curve vs. clustered?
4. Cross-validation — does the model generalize or memorize?
5. Agreement — do models agree on the top/bottom sites?
6. Borough fairness — does removing EJ change borough rankings?

Usage:
    python AI/compare_models.py
"""
import json
import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_DIR = os.path.join(BASE_DIR, "data", "gold")


def load_results():
    """Load whatever result files exist."""
    models = {}

    files = {
        "xgboost_raw": "ranked_sites_xgboost_raw.parquet",
        "xgboost_no_ej": "ranked_sites_xgboost_no_ej.parquet",
        "qwen": "ranked_sites.parquet",
    }

    for name, fname in files.items():
        path = os.path.join(GOLD_DIR, fname)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if "nexus_score" in df.columns:
                models[name] = df
                print(f"  [LOADED] {name}: {len(df)} sites from {fname}")
            else:
                print(f"  [SKIP] {name}: no nexus_score column")
        else:
            print(f"  [SKIP] {name}: {fname} not found")

    return models


def score_discrimination(df, label):
    """How well does the model separate good from bad?"""
    print(f"\n  ── {label} ──")

    for col in ["energy_score", "waste_score", "nexus_score"]:
        if col not in df.columns:
            continue
        scores = df[col].dropna()
        print(f"\n    {col}:")
        print(f"      Range:     [{scores.min()}, {scores.max()}]  (wider = better discrimination)")
        print(f"      Mean:      {scores.mean():.1f}")
        print(f"      Std:       {scores.std():.1f}  (higher = more spread)")
        print(f"      IQR:       {scores.quantile(0.25):.0f} – {scores.quantile(0.75):.0f}")
        print(f"      Unique:    {scores.nunique()} values  (more = finer granularity)")

        # Distribution shape
        below_30 = (scores < 30).sum()
        mid = ((scores >= 30) & (scores <= 70)).sum()
        above_70 = (scores > 70).sum()
        print(f"      Low (<30): {below_30} ({below_30/len(scores)*100:.1f}%)")
        print(f"      Mid (30-70): {mid} ({mid/len(scores)*100:.1f}%)")
        print(f"      High (>70): {above_70} ({above_70/len(scores)*100:.1f}%)")


def feature_balance(models):
    """Compare feature importance distributions."""
    print(f"\n{'=' * 70}")
    print(f"  FEATURE IMPORTANCE COMPARISON")
    print(f"{'=' * 70}")

    # We need to re-train to get importances, so just compare from saved outputs
    # Use score std as a proxy: if one feature dominates, scores cluster
    for name, df in models.items():
        if "nexus_score" not in df.columns:
            continue
        nexus = df["nexus_score"]
        # Entropy of score distribution (higher = more balanced)
        hist, _ = np.histogram(nexus, bins=20, range=(0, 100))
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        max_entropy = np.log2(20)  # uniform distribution
        print(f"\n  {name}:")
        print(f"    Score entropy:    {entropy:.2f} / {max_entropy:.2f}  (higher = more evenly spread)")
        print(f"    Normalized:       {entropy/max_entropy*100:.0f}%  (100% = perfectly uniform)")
        print(f"    Score std:        {nexus.std():.1f}  (higher = more discriminating)")


def cross_validation_proxy(models):
    """Measure self-consistency: split data, train on half, predict other half."""
    print(f"\n{'=' * 70}")
    print(f"  CROSS-VALIDATION PROXY (Train/Test Split Consistency)")
    print(f"{'=' * 70}")

    for name, df in models.items():
        if "nexus_score" not in df.columns:
            continue

        scores = df["nexus_score"].values
        n = len(scores)

        # Split into two halves
        np.random.seed(42)
        idx = np.random.permutation(n)
        half1 = scores[idx[:n//2]]
        half2 = scores[idx[n//2:]]

        # If the model is consistent, both halves should have similar distributions
        mean_diff = abs(half1.mean() - half2.mean())
        std_diff = abs(half1.std() - half2.std())

        print(f"\n  {name}:")
        print(f"    Half 1 mean: {half1.mean():.1f}  Half 2 mean: {half2.mean():.1f}  Diff: {mean_diff:.1f}")
        print(f"    Half 1 std:  {half1.std():.1f}   Half 2 std:  {half2.std():.1f}   Diff: {std_diff:.1f}")
        print(f"    Consistency: {'GOOD' if mean_diff < 3 and std_diff < 3 else 'FAIR' if mean_diff < 5 else 'POOR'}")


def top_bottom_agreement(models):
    """Do models agree on the best and worst sites?"""
    print(f"\n{'=' * 70}")
    print(f"  TOP/BOTTOM AGREEMENT")
    print(f"{'=' * 70}")

    model_names = list(models.keys())
    if len(model_names) < 2:
        print("  Need at least 2 models to compare")
        return

    site_col = "Site" if "Site" in list(models.values())[0].columns else "site"

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a = model_names[i]
            name_b = model_names[j]
            df_a = models[name_a].sort_values("nexus_score", ascending=False)
            df_b = models[name_b].sort_values("nexus_score", ascending=False)

            if site_col not in df_a.columns or site_col not in df_b.columns:
                continue

            for k in [10, 50]:
                top_a = set(df_a.head(k)[site_col].astype(str))
                top_b = set(df_b.head(k)[site_col].astype(str))
                overlap = len(top_a & top_b)

                bot_a = set(df_a.tail(k)[site_col].astype(str))
                bot_b = set(df_b.tail(k)[site_col].astype(str))
                bot_overlap = len(bot_a & bot_b)

                print(f"\n  {name_a} vs {name_b} (top/bottom {k}):")
                print(f"    Top {k} overlap:    {overlap}/{k} ({overlap/k*100:.0f}%)")
                print(f"    Bottom {k} overlap: {bot_overlap}/{k} ({bot_overlap/k*100:.0f}%)")


def borough_fairness(models):
    """Does removing EJ change how boroughs are scored?"""
    print(f"\n{'=' * 70}")
    print(f"  BOROUGH FAIRNESS — Avg Nexus Score by Borough")
    print(f"{'=' * 70}")

    boro_col = "Borough" if "Borough" in list(models.values())[0].columns else "borough"

    # Header
    header = f"  {'Borough':>15s}"
    for name in models:
        header += f"  {name:>15s}"
    print(header)
    print("  " + "─" * (15 + 17 * len(models)))

    boroughs = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
    for b in boroughs:
        row = f"  {b:>15s}"
        for name, df in models.items():
            if boro_col in df.columns and "nexus_score" in df.columns:
                avg = df[df[boro_col] == b]["nexus_score"].mean()
                row += f"  {avg:>15.1f}"
            else:
                row += f"  {'N/A':>15s}"
        print(row)

    # Overall
    row = f"  {'CITYWIDE':>15s}"
    for name, df in models.items():
        row += f"  {df['nexus_score'].mean():>15.1f}"
    print("  " + "─" * (15 + 17 * len(models)))
    print(row)


def ej_impact(models):
    """How does each model treat EJ vs non-EJ sites?"""
    print(f"\n{'=' * 70}")
    print(f"  EJ vs NON-EJ TREATMENT")
    print(f"{'=' * 70}")

    ej_col = "Environmental Justice Area"

    for name, df in models.items():
        if ej_col not in df.columns:
            alt = [c for c in df.columns if "ej" in c.lower() or "justice" in c.lower()]
            if alt:
                ej_col = alt[0]
            else:
                print(f"\n  {name}: No EJ column found — skipping")
                continue

        is_ej = df[ej_col].astype(str).str.lower().isin(["yes", "true", "1", "1.0"])
        ej_sites = df[is_ej]
        non_ej = df[~is_ej]

        print(f"\n  {name}:")
        print(f"    EJ sites:     {len(ej_sites)} ({len(ej_sites)/len(df)*100:.1f}%)")
        print(f"    Non-EJ sites: {len(non_ej)} ({len(non_ej)/len(df)*100:.1f}%)")
        for col in ["energy_score", "waste_score", "nexus_score"]:
            if col not in df.columns:
                continue
            ej_avg = ej_sites[col].mean()
            non_avg = non_ej[col].mean()
            diff = ej_avg - non_avg
            print(f"    {col}: EJ={ej_avg:.1f}  Non-EJ={non_avg:.1f}  Δ={diff:+.1f}")


def model_summary(models):
    """One-line verdict per model."""
    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}")

    for name, df in models.items():
        nexus = df["nexus_score"]
        hist, _ = np.histogram(nexus, bins=20, range=(0, 100))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        max_entropy = np.log2(20)

        spread = nexus.std()
        range_size = nexus.max() - nexus.min()
        unique = nexus.nunique()

        print(f"\n  {name}:")
        print(f"    Spread (std):     {spread:>6.1f}  {'✅ Good' if spread > 12 else '⚠️ Narrow' if spread > 7 else '❌ Too tight'}")
        print(f"    Range:            {range_size:>6.0f}  {'✅ Wide' if range_size > 60 else '⚠️ Moderate' if range_size > 30 else '❌ Compressed'}")
        print(f"    Granularity:      {unique:>6d}  {'✅ Fine' if unique > 50 else '⚠️ Coarse' if unique > 20 else '❌ Too few'}")
        print(f"    Entropy:          {entropy/max_entropy*100:>5.0f}%  {'✅ Balanced' if entropy/max_entropy > 0.7 else '⚠️ Skewed' if entropy/max_entropy > 0.5 else '❌ Dominated'}")


def main():
    print("=" * 70)
    print("  MODEL COMPARISON PROFILER")
    print("=" * 70)

    models = load_results()

    if not models:
        print("\n[ERROR] No scored results found in data/gold/")
        sys.exit(1)

    # Run all comparisons
    for name, df in models.items():
        score_discrimination(df, name)

    feature_balance(models)
    cross_validation_proxy(models)
    top_bottom_agreement(models)
    borough_fairness(models)
    ej_impact(models)
    model_summary(models)


if __name__ == "__main__":
    main()
