"""
Linear Regression Site Scoring — Baseline comparison against XGBoost.

Same pipeline: 12 raw CSVs → 30 features → 70/15/15 split → train/val/test.
Uses sklearn LinearRegression (no trees, no interactions, pure linear weights).

Purpose: shows what a simple linear model can do vs XGBoost's non-linear trees.
If linear matches XGBoost → XGBoost is overkill.
If XGBoost wins → non-linear interactions matter.

Usage:
    python AI/score_raw_linear_regression.py
    RAW_DIR=/home/acergn100_6/smart-city-management/data/raw python3 AI/score_raw_linear_regression.py
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.getenv("RAW_DIR", os.path.join(BASE_DIR, "data", "raw"))
OUT_DIR = os.getenv("OUT_DIR", os.path.join(BASE_DIR, "data", "gold"))
os.makedirs(OUT_DIR, exist_ok=True)

# Reuse data loading from score_raw_xgboost_no_ej.py
sys.path.insert(0, os.path.join(BASE_DIR, "AI"))


# ---------------------------------------------------------------------------
# Load data (reuse XGBoost pipeline)
# ---------------------------------------------------------------------------
def load_data():
    from score_raw_xgboost_no_ej import (
        load_e5, load_e3, load_e4, load_e7, load_e10,
        load_w1, load_w2, load_w7, load_w8,
        build_feature_matrix
    )
    print("── PHASE 1: LOAD & BUILD FEATURES ──")
    e5 = load_e5()
    e3 = load_e3()
    e4 = load_e4()
    e7 = load_e7()
    e10 = load_e10()
    w1 = load_w1()
    w2 = load_w2()
    w7 = load_w7()
    w8 = load_w8()
    sites_df, features_df, feature_cols = build_feature_matrix(e5, e3, e4, e7, e10, w1, w2, w7, w8)
    return sites_df, features_df.values.astype(np.float32), feature_cols


# ---------------------------------------------------------------------------
# Generate labels (same as XGBoost for fair comparison)
# ---------------------------------------------------------------------------
def generate_labels(X, feature_cols):
    n = len(X)
    np.random.seed(42)
    col = {name: i for i, name in enumerate(feature_cols)}

    def rp(arr):
        return rankdata(arr, method="average") / n

    solar = X[:, col["solar_kwh"]]
    sqft = X[:, col["sqft"]]
    roof_g = X[:, col["roof_good"]]
    ev_1km = X[:, col["ev_1km"]]
    e7_total = X[:, col["e7_avg_total"]]
    w1_refuse = X[:, col["w1_refuse"]]
    w1_div = X[:, col["w1_diversion"]]
    compost = X[:, col["compost_1km"]]
    transfer = X[:, col["nearest_transfer_m"]]
    ghg = X[:, col["ghg"]]
    site_eui = X[:, col["site_eui"]]
    e_star = X[:, col["energy_star"]]
    w1_organics = X[:, col["w1_organics"]]

    energy_raw = (
        rp(solar) * 20 + rp(e7_total) * 20 + rp(ghg) * 15 + rp(site_eui) * 10 +
        roof_g * 10 + rp(sqft) * 10 + rp(ev_1km) * 10 +
        (1 - rp(np.clip(e_star, 0, 100))) * 5
    )
    energy = energy_raw / energy_raw.max() * 100 + np.random.normal(0, 3, n)

    waste_raw = (
        rp(w1_refuse) * 25 + (1 - np.clip(w1_div, 0, 1)) * 20 + rp(w1_organics) * 15 +
        (1 - np.clip(rp(compost), 0, 1)) * 15 + rp(transfer) * 15 + rp(sqft) * 10
    )
    waste = waste_raw / waste_raw.max() * 100 + np.random.normal(0, 3, n)

    nexus_raw = (
        energy_raw * 0.35 + waste_raw * 0.35 +
        (rp(solar) * rp(w1_refuse)) * 20 +
        (rp(ev_1km) * rp(solar)) * 10
    )
    nexus = nexus_raw / nexus_raw.max() * 100 + np.random.normal(0, 3, n)

    return np.clip(energy, 0, 100), np.clip(waste, 0, 100), np.clip(nexus, 0, 100)


# ---------------------------------------------------------------------------
# Train & evaluate with 3 linear methods
# ---------------------------------------------------------------------------
def train_and_score(X, feature_cols, energy_y, waste_y, nexus_y, sites_df):
    """Train Linear Regression, Ridge, Lasso with 70/15/15 split."""
    n = len(X)

    # Same split as XGBoost (seed=42)
    np.random.seed(42)
    indices = np.random.permutation(n)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    print(f"\n  Split: Train={len(train_idx)} (70%) | Val={len(val_idx)} (15%) | Test={len(test_idx)} (15%)")
    print(f"  Same seed=42 as XGBoost — identical sites in each split.\n")

    # Standardize features (important for linear models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X[train_idx])
    X_val_scaled = scaler.transform(X[val_idx])
    X_test_scaled = scaler.transform(X[test_idx])
    X_all_scaled = scaler.transform(X)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge (α=1.0)": Ridge(alpha=1.0),
        "Lasso (α=0.1)": Lasso(alpha=0.1, max_iter=5000),
    }

    all_results = {}

    for score_name, y_all in [("energy", energy_y), ("waste", waste_y), ("nexus", nexus_y)]:
        y_train = y_all[train_idx]
        y_val = y_all[val_idx]
        y_test = y_all[test_idx]

        print(f"\n  ── {score_name}_score ──")
        print(f"  {'Model':25s}  {'Train RMSE':>10s}  {'Val RMSE':>10s}  {'Test RMSE':>10s}  "
              f"{'Test R²':>8s}  {'Test MAE':>8s}  {'Overfit':>8s}")
        print(f"  {'─'*85}")

        best_model = None
        best_val_rmse = float('inf')

        for model_name, model in models.items():
            t0 = time.time()

            model.fit(X_train_scaled, y_train)

            train_preds = model.predict(X_train_scaled)
            val_preds = model.predict(X_val_scaled)
            test_preds = model.predict(X_test_scaled)

            train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            test_r2 = r2_score(y_test, test_preds)
            test_mae = mean_absolute_error(y_test, test_preds)

            overfit_gap = val_rmse - train_rmse
            overfit = "YES" if overfit_gap > 2 else "mild" if overfit_gap > 1 else "no"

            elapsed = time.time() - t0

            print(f"  {model_name:25s}  {train_rmse:>10.2f}  {val_rmse:>10.2f}  {test_rmse:>10.2f}  "
                  f"{test_r2:>8.3f}  {test_mae:>8.2f}  {overfit:>8s}")

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model = model
                best_model_name = model_name

        # Use best model for final predictions
        all_preds = np.clip(best_model.predict(X_all_scaled), 0, 100).astype(int)
        val_preds_best = np.clip(best_model.predict(X_val_scaled), 0, 100).astype(int)
        test_preds_best = np.clip(best_model.predict(X_test_scaled), 0, 100).astype(int)

        test_r2_best = r2_score(y_test, best_model.predict(X_test_scaled))
        val_r2_best = r2_score(y_val, best_model.predict(X_val_scaled))

        print(f"  → Best: {best_model_name} (val RMSE={best_val_rmse:.2f})")

        # Print coefficients for the best model
        if hasattr(best_model, 'coef_'):
            print(f"\n  Coefficients ({best_model_name}):")
            coefs = list(zip(feature_cols, best_model.coef_))
            coefs.sort(key=lambda x: abs(x[1]), reverse=True)
            for fname, coef in coefs[:10]:
                direction = "+" if coef > 0 else "−"
                print(f"    {fname:25s}  {direction}{abs(coef):>8.3f}  "
                      f"{'↑ higher = higher score' if coef > 0 else '↑ higher = lower score'}")

        all_results[score_name] = {
            "model": best_model,
            "model_name": best_model_name,
            "all_preds": all_preds,
            "val_preds": val_preds_best,
            "test_preds": test_preds_best,
            "val_r2": val_r2_best,
            "test_r2": test_r2_best,
        }

    # Store indices
    all_results["_train_idx"] = train_idx
    all_results["_val_idx"] = val_idx
    all_results["_test_idx"] = test_idx

    return all_results


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
def save_results(sites_df, results, feature_cols):
    val_idx = results["_val_idx"]
    test_idx = results["_test_idx"]

    def build_ranked(df_indices, preds_key, split_label):
        df = sites_df.iloc[df_indices].copy()
        df["energy_score"] = results["energy"][preds_key]
        df["waste_score"] = results["waste"][preds_key]
        df["nexus_score"] = results["nexus"][preds_key]
        df["split"] = split_label

        ranked = df[["Site", "Address", "Borough", "Agency",
                      "Environmental Justice Area", "lat", "lon", "bbl",
                      "energy_score", "waste_score", "nexus_score", "split"]].copy()

        solar = df["solar_kwh"].fillna(0).values if "solar_kwh" in df.columns else np.zeros(len(df))
        ranked["recommended_bess_kwh"] = np.where(solar > 100000, 750,
                                         np.where(solar > 10000, 500,
                                         np.where(ranked["energy_score"] > 60, 250, 100)))
        ranked["estimated_annual_savings_usd"] = np.where(solar > 100000, 50000 + (solar * 0.05).astype(int),
                                                 np.where(solar > 10000, 20000, 5000))
        recs = []
        for _, r in ranked.iterrows():
            if r["nexus_score"] >= 75:
                recs.append(f"Deploy {r['recommended_bess_kwh']} kWh BESS + partner with AD facility")
            elif r["energy_score"] >= 70:
                recs.append(f"Install {r['recommended_bess_kwh']} kWh BESS paired with solar")
            elif r["waste_score"] >= 70:
                recs.append("Prioritize organic waste diversion")
            else:
                recs.append("Monitor for future assessment")
        ranked["top_recommendation"] = recs
        ranked = ranked.sort_values("nexus_score", ascending=False).reset_index(drop=True)
        ranked["rank"] = range(1, len(ranked) + 1)
        return ranked

    val_ranked = build_ranked(val_idx, "val_preds", "validation")
    test_ranked = build_ranked(test_idx, "test_preds", "test")

    # Save
    val_path = os.path.join(OUT_DIR, "ranked_sites_linear_val.parquet")
    val_ranked.to_parquet(val_path, index=False)
    print(f"\n[SAVED] {val_path} — {len(val_ranked)} val sites")

    val_json = os.path.join(OUT_DIR, "top50_linear_val.json")
    with open(val_json, "w") as f:
        json.dump(val_ranked.head(50).to_dict(orient="records"), f, indent=2, default=str)
    print(f"[SAVED] {val_json}")

    test_path = os.path.join(OUT_DIR, "ranked_sites_linear_test.parquet")
    test_ranked.to_parquet(test_path, index=False)
    print(f"[SAVED] {test_path} — {len(test_ranked)} test sites")

    test_json = os.path.join(OUT_DIR, "top50_linear_test.json")
    with open(test_json, "w") as f:
        json.dump(test_ranked.head(50).to_dict(orient="records"), f, indent=2, default=str)
    print(f"[SAVED] {test_json}")

    # Print tops
    print(f"\n  VAL TOP 10 (Linear Regression):")
    print("  " + "─" * 75)
    for _, r in val_ranked.head(10).iterrows():
        print(f"  #{int(r['rank']):>3d}  E:{int(r['energy_score']):>3d}  W:{int(r['waste_score']):>3d}  "
              f"N:{int(r['nexus_score']):>3d}  {str(r['Borough']):>12s}  {str(r['Site'])[:35]}")

    print(f"\n  TEST TOP 10 (Linear Regression):")
    print("  " + "─" * 75)
    for _, r in test_ranked.head(10).iterrows():
        print(f"  #{int(r['rank']):>3d}  E:{int(r['energy_score']):>3d}  W:{int(r['waste_score']):>3d}  "
              f"N:{int(r['nexus_score']):>3d}  {str(r['Borough']):>12s}  {str(r['Site'])[:35]}")
    print("  " + "─" * 75)

    # Compare with XGBoost
    print(f"\n  MODEL COMPARISON (same test set, seed=42):")
    print(f"  {'':15s}  {'Linear R²':>10s}  {'XGBoost R²':>10s}  {'Winner':>8s}")
    print(f"  {'─'*50}")
    xgb_r2 = {"energy": 0.945, "waste": 0.884, "nexus": 0.866}  # from previous run
    for name in ["energy", "waste", "nexus"]:
        lr = results[name]["test_r2"]
        xg = xgb_r2[name]
        winner = "XGBoost" if xg > lr else "Linear" if lr > xg else "Tie"
        print(f"  {name+'_score':15s}  {lr:>10.3f}  {xg:>10.3f}  {winner:>8s}")

    avg_lr = np.mean([results[n]["test_r2"] for n in ["energy", "waste", "nexus"]])
    avg_xg = np.mean(list(xgb_r2.values()))
    print(f"  {'AVERAGE':15s}  {avg_lr:>10.3f}  {avg_xg:>10.3f}  {'XGBoost' if avg_xg > avg_lr else 'Linear'}")
    print(f"\n  R² difference: {avg_xg - avg_lr:.3f} ({'XGBoost wins by ' + f'{(avg_xg-avg_lr)*100:.1f}%' if avg_xg > avg_lr else 'Linear wins'})")

    return val_ranked, test_ranked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  LINEAR REGRESSION: Baseline Comparison")
    print(f"  Raw data: {RAW_DIR}")
    print("=" * 65)

    t_total = time.time()

    sites_df, X, feature_cols = load_data()
    print(f"\n  {X.shape[0]} sites × {X.shape[1]} features")

    print("\n── PHASE 2: GENERATE LABELS ──")
    energy_y, waste_y, nexus_y = generate_labels(X, feature_cols)

    print("\n── PHASE 3: TRAIN & EVALUATE ──")
    results = train_and_score(X, feature_cols, energy_y, waste_y, nexus_y, sites_df)

    print("\n── PHASE 4: SAVE RESULTS ──")
    save_results(sites_df, results, feature_cols)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 65}")
    print(f"  COMPLETE — {elapsed:.1f}s total")
    print(f"  Val R²:  {np.mean([results[n]['val_r2'] for n in ['energy','waste','nexus']]):.3f}")
    print(f"  Test R²: {np.mean([results[n]['test_r2'] for n in ['energy','waste','nexus']]):.3f}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
