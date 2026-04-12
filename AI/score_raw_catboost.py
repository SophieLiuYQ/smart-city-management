"""
CatBoost Site Scoring — Comparison against XGBoost and Linear Regression.

Same pipeline: 12 raw CSVs → 30 features → 70/15/15 split (seed=42).
CatBoost: gradient boosted trees with built-in categorical handling,
ordered boosting (reduces overfitting), and symmetric trees.

Usage:
    python AI/score_raw_catboost.py
    RAW_DIR=/home/acergn100_6/smart-city-management/data/raw python3 AI/score_raw_catboost.py
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata

try:
    from catboost import CatBoostRegressor, Pool
    print(f"[CatBoost] loaded")
except ImportError:
    print("[ERROR] pip install catboost")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.getenv("RAW_DIR", os.path.join(BASE_DIR, "data", "raw"))
OUT_DIR = os.getenv("OUT_DIR", os.path.join(BASE_DIR, "data", "gold"))
os.makedirs(OUT_DIR, exist_ok=True)

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
# Generate labels (same formula, same seed as XGBoost)
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
# Train & evaluate
# ---------------------------------------------------------------------------
def train_and_score(X, feature_cols, energy_y, waste_y, nexus_y, sites_df):
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

    results = {}

    print(f"  {'Score':15s}  {'Train RMSE':>10s}  {'Val RMSE':>10s}  {'Val R²':>8s}  "
          f"{'Test RMSE':>10s}  {'Test R²':>8s}  {'Rounds':>6s}  {'Overfit':>8s}  Time")
    print(f"  {'─'*95}")

    for name, y in [("energy", energy_y), ("waste", waste_y), ("nexus", nexus_y)]:
        t0 = time.time()

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        train_pool = Pool(X_train, y_train, feature_names=feature_cols)
        val_pool = Pool(X_val, y_val, feature_names=feature_cols)
        test_pool = Pool(X_test, y_test, feature_names=feature_cols)

        model = CatBoostRegressor(
            iterations=500,
            depth=4,
            learning_rate=0.1,
            l2_leaf_reg=5,
            subsample=0.8,
            loss_function="RMSE",
            early_stopping_rounds=20,
            verbose=0,
            task_type="CPU",  # CatBoost GPU needs CUDA toolkit
        )

        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
        )

        best_rounds = model.get_best_iteration() + 1

        train_preds = model.predict(train_pool)
        val_preds = model.predict(val_pool)
        test_preds = model.predict(test_pool)

        train_rmse = np.sqrt(np.mean((train_preds - y_train) ** 2))
        val_rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))
        test_rmse = np.sqrt(np.mean((test_preds - y_test) ** 2))

        val_r2 = 1 - np.sum((y_val - val_preds)**2) / np.sum((y_val - y_val.mean())**2)
        test_r2 = 1 - np.sum((y_test - test_preds)**2) / np.sum((y_test - y_test.mean())**2)

        overfit_gap = val_rmse - train_rmse
        overfit = "YES" if overfit_gap > 2 else "mild" if overfit_gap > 1 else "no"

        elapsed = time.time() - t0

        print(f"  {name+'_score':15s}  {train_rmse:>10.2f}  {val_rmse:>10.2f}  {val_r2:>8.3f}  "
              f"{test_rmse:>10.2f}  {test_r2:>8.3f}  {best_rounds:>6d}  {overfit:>8s}  {elapsed:.2f}s")

        # Predictions for output
        val_preds_clipped = np.clip(val_preds, 0, 100).astype(int)
        test_preds_clipped = np.clip(test_preds, 0, 100).astype(int)

        results[name] = {
            "model": model,
            "time": elapsed,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "best_rounds": best_rounds,
            "overfit": overfit,
            "val_preds": val_preds_clipped,
            "test_preds": test_preds_clipped,
        }

    # Summary
    avg_val_r2 = np.mean([results[n]["val_r2"] for n in ["energy", "waste", "nexus"]])
    avg_test_r2 = np.mean([results[n]["test_r2"] for n in ["energy", "waste", "nexus"]])

    print(f"\n  RESULTS:")
    print(f"    Val R²:  {avg_val_r2:.3f}")
    print(f"    Test R²: {avg_test_r2:.3f}")

    if abs(avg_val_r2 - avg_test_r2) < 0.03:
        print(f"    ✅ Val ≈ Test — model is stable")
    elif avg_val_r2 > avg_test_r2:
        print(f"    ⚠️ Val > Test by {avg_val_r2 - avg_test_r2:.3f}")
    else:
        print(f"    ✅ Test > Val — generalizes well")

    # Feature importance
    print(f"\n  FEATURE IMPORTANCE (nexus_score):")
    imp = results["nexus"]["model"].get_feature_importance()
    imp_pairs = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)
    for fname, importance in imp_pairs[:10]:
        bar = "█" * int(importance / max(imp) * 30)
        print(f"    {fname:25s}  {importance:>6.1f}  {bar}")

    results["_val_idx"] = val_idx
    results["_test_idx"] = test_idx

    return results


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
    for label, ranked, suffix in [("val", val_ranked, "catboost_val"), ("test", test_ranked, "catboost_test")]:
        pq = os.path.join(OUT_DIR, f"ranked_sites_{suffix}.parquet")
        ranked.to_parquet(pq, index=False)
        print(f"\n[SAVED] {pq} — {len(ranked)} {label} sites")

        js = os.path.join(OUT_DIR, f"top50_{suffix}.json")
        with open(js, "w") as f:
            json.dump(ranked.head(50).to_dict(orient="records"), f, indent=2, default=str)
        print(f"[SAVED] {js}")

    # Print tops
    for label, ranked in [("VAL", val_ranked), ("TEST", test_ranked)]:
        print(f"\n  {label} TOP 10 (CatBoost):")
        print("  " + "─" * 75)
        for _, r in ranked.head(10).iterrows():
            print(f"  #{int(r['rank']):>3d}  E:{int(r['energy_score']):>3d}  W:{int(r['waste_score']):>3d}  "
                  f"N:{int(r['nexus_score']):>3d}  {str(r['Borough']):>12s}  {str(r['Site'])[:35]}")

    # MODEL COMPARISON TABLE
    xgb_r2 = {"energy": 0.945, "waste": 0.884, "nexus": 0.866}
    lr_r2 = {"energy": 0.766, "waste": 0.731, "nexus": 0.595}

    print(f"\n  {'─'*65}")
    print(f"  MODEL COMPARISON (same test set, seed=42):")
    print(f"  {'Score':15s}  {'Linear':>8s}  {'XGBoost':>8s}  {'CatBoost':>8s}  Winner")
    print(f"  {'─'*55}")
    for name in ["energy", "waste", "nexus"]:
        lr = lr_r2[name]
        xg = xgb_r2[name]
        cb = results[name]["test_r2"]
        scores = {"Linear": lr, "XGBoost": xg, "CatBoost": cb}
        winner = max(scores, key=scores.get)
        print(f"  {name+'_score':15s}  {lr:>8.3f}  {xg:>8.3f}  {cb:>8.3f}  {winner}")

    avg_lr = np.mean(list(lr_r2.values()))
    avg_xg = np.mean(list(xgb_r2.values()))
    avg_cb = np.mean([results[n]["test_r2"] for n in ["energy", "waste", "nexus"]])
    scores = {"Linear": avg_lr, "XGBoost": avg_xg, "CatBoost": avg_cb}
    winner = max(scores, key=scores.get)
    print(f"  {'AVERAGE':15s}  {avg_lr:>8.3f}  {avg_xg:>8.3f}  {avg_cb:>8.3f}  {winner}")
    print(f"  {'─'*55}")

    return val_ranked, test_ranked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  CatBoost SCORING — Comparison Baseline")
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
    train_time = sum(results[n]["time"] for n in ["energy", "waste", "nexus"])

    print(f"\n{'=' * 65}")
    print(f"  COMPLETE — {elapsed:.1f}s total (training: {train_time:.1f}s)")
    print(f"  Val R²:  {np.mean([results[n]['val_r2'] for n in ['energy','waste','nexus']]):.3f}")
    print(f"  Test R²: {np.mean([results[n]['test_r2'] for n in ['energy','waste','nexus']]):.3f}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
