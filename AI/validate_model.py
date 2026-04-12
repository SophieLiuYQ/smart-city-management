"""
Model Validation & Improvement for XGBoost scoring.

Implements:
1. K-Fold Cross-Validation (5-fold, 10-fold)
2. Leave-One-Out CV (LOO) — feasible because XGBoost trains in <1s
3. Hyperparameter tuning via CV
4. Learning curves — is more data helping?
5. Residual analysis — where does the model fail?

Usage:
    python AI/validate_model.py
    python AI/validate_model.py --loo        # run full LOO (slower, ~20 min)
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from collections import defaultdict

try:
    import xgboost as xgb
    print(f"[XGBoost] v{xgb.__version__}")
except ImportError:
    print("[ERROR] pip install xgboost")
    sys.exit(1)

from scipy.stats import rankdata

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_DIR = os.path.join(BASE_DIR, "data", "gold")
RAW_DIR = os.getenv("RAW_DIR", os.path.join(BASE_DIR, "data", "raw"))

# Reuse data loading from score_raw_xgboost_no_ej
sys.path.insert(0, os.path.join(BASE_DIR, "AI"))


# ---------------------------------------------------------------------------
# Load features (reuse from no_ej script)
# ---------------------------------------------------------------------------
def load_data():
    """Load and build feature matrix — same as score_raw_xgboost_no_ej.py"""
    from score_raw_xgboost_no_ej import (
        load_e5, load_e3, load_e4, load_e7, load_e10,
        load_w1, load_w2, load_w7, load_w8,
        build_feature_matrix
    )

    print("── Loading data ──")
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
    X = features_df.values.astype(np.float32)

    return sites_df, X, feature_cols


def generate_labels(X, feature_cols):
    """Same label generation as no_ej script."""
    n = len(X)
    np.random.seed(42)
    col = {name: i for i, name in enumerate(feature_cols)}

    def rp(arr):
        return rankdata(arr, method="average") / len(arr)

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

    nexus_raw = (
        (rp(solar) * 20 + rp(e7_total) * 20 + rp(ghg) * 15 + rp(site_eui) * 10 +
         roof_g * 10 + rp(sqft) * 10 + rp(ev_1km) * 10 +
         (1 - rp(np.clip(e_star, 0, 100))) * 5) * 0.35 +
        (rp(w1_refuse) * 25 + (1 - np.clip(w1_div, 0, 1)) * 20 + rp(w1_organics) * 15 +
         (1 - np.clip(rp(compost), 0, 1)) * 15 + rp(transfer) * 15 + rp(sqft) * 10) * 0.35 +
        (rp(solar) * rp(w1_refuse)) * 20 +
        (rp(ev_1km) * rp(solar)) * 10
    )
    nexus = nexus_raw / nexus_raw.max() * 100 + np.random.normal(0, 3, n)
    return np.clip(nexus, 0, 100).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. K-Fold Cross-Validation
# ---------------------------------------------------------------------------
def kfold_cv(X, y, feature_cols, params, n_folds=5, n_rounds=200):
    """Standard K-Fold CV. Returns per-fold RMSE and R²."""
    print(f"\n{'=' * 65}")
    print(f"  K-FOLD CROSS-VALIDATION (k={n_folds})")
    print(f"{'=' * 65}")

    n = len(X)
    indices = np.random.permutation(n)
    fold_size = n // n_folds

    fold_rmse = []
    fold_r2 = []
    fold_mae = []
    all_preds = np.zeros(n)

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else n

        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

        model = xgb.train(params, dtrain, num_boost_round=n_rounds, verbose_eval=False)
        preds = model.predict(dtest)
        all_preds[test_idx] = preds

        rmse = np.sqrt(np.mean((preds - y_test) ** 2))
        mae = np.mean(np.abs(preds - y_test))
        ss_res = np.sum((y_test - preds) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        fold_rmse.append(rmse)
        fold_r2.append(r2)
        fold_mae.append(mae)

        print(f"  Fold {fold+1}/{n_folds}: RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}  "
              f"(train={len(train_idx)}, test={len(test_idx)})")

    print(f"\n  SUMMARY:")
    print(f"    RMSE:  {np.mean(fold_rmse):.2f} ± {np.std(fold_rmse):.2f}")
    print(f"    MAE:   {np.mean(fold_mae):.2f} ± {np.std(fold_mae):.2f}")
    print(f"    R²:    {np.mean(fold_r2):.3f} ± {np.std(fold_r2):.3f}")
    print(f"    (R² of 1.0 = perfect, 0.0 = random, <0 = worse than mean)")

    # Interpretation
    r2_mean = np.mean(fold_r2)
    if r2_mean > 0.95:
        print(f"\n    ⚠️  R² > 0.95 — model is likely memorizing the label formula.")
        print(f"        This is expected with self-supervised labels.")
    elif r2_mean > 0.7:
        print(f"\n    ✅ R² = {r2_mean:.3f} — model generalizes well.")
    else:
        print(f"\n    ❌ R² = {r2_mean:.3f} — model struggles to generalize.")

    return all_preds, fold_rmse, fold_r2


# ---------------------------------------------------------------------------
# 2. Leave-One-Out Cross-Validation
# ---------------------------------------------------------------------------
def loo_cv(X, y, feature_cols, params, n_rounds=200, max_samples=None):
    """LOO-CV: train on N-1, predict 1, repeat N times."""
    n = len(X)
    if max_samples:
        n = min(n, max_samples)

    print(f"\n{'=' * 65}")
    print(f"  LEAVE-ONE-OUT CROSS-VALIDATION (n={n})")
    print(f"{'=' * 65}")

    preds = np.zeros(n)
    t0 = time.time()

    for i in range(n):
        train_mask = np.ones(len(X), dtype=bool)
        train_mask[i] = False

        dtrain = xgb.DMatrix(X[train_mask], label=y[train_mask], feature_names=feature_cols)
        dtest = xgb.DMatrix(X[i:i+1], feature_names=feature_cols)

        model = xgb.train(params, dtrain, num_boost_round=n_rounds, verbose_eval=False)
        preds[i] = model.predict(dtest)[0]

        if (i + 1) % 500 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate / 60
            print(f"    {i+1}/{n} ({elapsed:.0f}s, ~{eta:.1f}min remaining)")

    y_subset = y[:n]
    rmse = np.sqrt(np.mean((preds - y_subset) ** 2))
    mae = np.mean(np.abs(preds - y_subset))
    ss_res = np.sum((y_subset - preds) ** 2)
    ss_tot = np.sum((y_subset - y_subset.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    elapsed = time.time() - t0
    print(f"\n  LOO RESULTS ({elapsed:.1f}s total):")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    MAE:  {mae:.2f}")
    print(f"    R²:   {r2:.3f}")

    # Residual analysis
    residuals = preds - y_subset
    print(f"\n  RESIDUAL ANALYSIS:")
    print(f"    Mean residual:  {residuals.mean():.2f} (0 = unbiased)")
    print(f"    Std residual:   {residuals.std():.2f}")
    print(f"    Max overpredict: {residuals.max():.1f} points")
    print(f"    Max underpredict: {residuals.min():.1f} points")

    return preds, rmse, r2


# ---------------------------------------------------------------------------
# 3. Hyperparameter Tuning via CV
# ---------------------------------------------------------------------------
def tune_hyperparams(X, y, feature_cols):
    """Grid search over key XGBoost params using 5-fold CV."""
    print(f"\n{'=' * 65}")
    print(f"  HYPERPARAMETER TUNING (5-fold CV)")
    print(f"{'=' * 65}")

    base_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "verbosity": 0,
    }

    grid = {
        "max_depth": [3, 4, 6, 8],
        "eta": [0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [3, 5, 10],
        "num_boost_round": [100, 200, 300],
    }

    # Instead of full grid (2916 combos), do sequential tuning
    best_params = {
        "max_depth": 6, "eta": 0.1, "subsample": 0.8,
        "colsample_bytree": 0.8, "min_child_weight": 5,
    }
    best_rounds = 200
    best_rmse = float("inf")

    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)

    # Tune each param independently
    for param_name, values in grid.items():
        if param_name == "num_boost_round":
            continue
        print(f"\n  Tuning {param_name}: {values}")
        param_results = []

        for val in values:
            test_params = {**base_params, **best_params, param_name: val}
            cv_results = xgb.cv(
                test_params, dtrain,
                num_boost_round=best_rounds,
                nfold=5, seed=42,
                verbose_eval=False,
            )
            rmse = cv_results["test-rmse-mean"].iloc[-1]
            param_results.append((val, rmse))
            print(f"    {param_name}={val}: RMSE={rmse:.3f}")

        best_val, best_val_rmse = min(param_results, key=lambda x: x[1])
        best_params[param_name] = best_val
        if best_val_rmse < best_rmse:
            best_rmse = best_val_rmse
        print(f"    → Best: {param_name}={best_val} (RMSE={best_val_rmse:.3f})")

    # Tune num_boost_round
    print(f"\n  Tuning num_boost_round: {grid['num_boost_round']}")
    for rounds in grid["num_boost_round"]:
        test_params = {**base_params, **best_params}
        cv_results = xgb.cv(
            test_params, dtrain,
            num_boost_round=rounds,
            nfold=5, seed=42,
            verbose_eval=False,
        )
        rmse = cv_results["test-rmse-mean"].iloc[-1]
        print(f"    rounds={rounds}: RMSE={rmse:.3f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_rounds = rounds

    print(f"\n  BEST CONFIGURATION:")
    print(f"    {json.dumps(best_params, indent=4)}")
    print(f"    num_boost_round: {best_rounds}")
    print(f"    Best CV RMSE: {best_rmse:.3f}")

    return best_params, best_rounds, best_rmse


# ---------------------------------------------------------------------------
# 4. Learning Curve
# ---------------------------------------------------------------------------
def learning_curve(X, y, feature_cols, params, n_rounds=200):
    """Train on increasing data sizes. Does more data help?"""
    print(f"\n{'=' * 65}")
    print(f"  LEARNING CURVE — Does more data improve the model?")
    print(f"{'=' * 65}")

    n = len(X)
    fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

    np.random.seed(42)
    idx = np.random.permutation(n)
    test_size = n // 5
    test_idx = idx[:test_size]
    pool_idx = idx[test_size:]

    X_test, y_test = X[test_idx], y[test_idx]
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    print(f"\n  Test set: {len(test_idx)} sites (fixed)")
    print(f"  Training pool: {len(pool_idx)} sites\n")
    print(f"  {'Fraction':>10s}  {'Train size':>10s}  {'Train RMSE':>10s}  {'Test RMSE':>10s}  {'Test R²':>8s}  {'Overfit?':>8s}")
    print(f"  {'─'*62}")

    for frac in fractions:
        train_n = int(len(pool_idx) * frac)
        train_idx = pool_idx[:train_n]
        X_train, y_train = X[train_idx], y[train_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        model = xgb.train(params, dtrain, num_boost_round=n_rounds, verbose_eval=False)

        train_preds = model.predict(dtrain)
        test_preds = model.predict(dtest)

        train_rmse = np.sqrt(np.mean((train_preds - y_train) ** 2))
        test_rmse = np.sqrt(np.mean((test_preds - y_test) ** 2))

        ss_res = np.sum((y_test - test_preds) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        gap = test_rmse - train_rmse
        overfit = "YES" if gap > 2 else "mild" if gap > 1 else "no"

        print(f"  {frac:>10.0%}  {train_n:>10d}  {train_rmse:>10.2f}  {test_rmse:>10.2f}  {r2:>8.3f}  {overfit:>8s}")


# ---------------------------------------------------------------------------
# 5. Feature Ablation — Which features actually matter?
# ---------------------------------------------------------------------------
def feature_ablation(X, y, feature_cols, params, n_rounds=200):
    """Remove one feature at a time. Which removal hurts most?"""
    print(f"\n{'=' * 65}")
    print(f"  FEATURE ABLATION — Drop one feature, measure impact")
    print(f"{'=' * 65}")

    dtrain_full = xgb.DMatrix(X, label=y, feature_names=feature_cols)
    cv_full = xgb.cv(params, dtrain_full, num_boost_round=n_rounds, nfold=5, seed=42, verbose_eval=False)
    baseline_rmse = cv_full["test-rmse-mean"].iloc[-1]

    print(f"\n  Baseline RMSE (all {len(feature_cols)} features): {baseline_rmse:.3f}\n")
    print(f"  {'Removed feature':>25s}  {'RMSE':>8s}  {'Δ RMSE':>8s}  Impact")
    print(f"  {'─'*58}")

    results = []
    for i, col in enumerate(feature_cols):
        # Remove column i
        X_reduced = np.delete(X, i, axis=1)
        cols_reduced = [c for j, c in enumerate(feature_cols) if j != i]

        dtrain = xgb.DMatrix(X_reduced, label=y, feature_names=cols_reduced)
        cv_res = xgb.cv(params, dtrain, num_boost_round=n_rounds, nfold=5, seed=42, verbose_eval=False)
        rmse = cv_res["test-rmse-mean"].iloc[-1]
        delta = rmse - baseline_rmse

        impact = "CRITICAL" if delta > 0.5 else "important" if delta > 0.1 else "helpful" if delta > 0.01 else "negligible"
        results.append((col, rmse, delta, impact))
        print(f"  {col:>25s}  {rmse:>8.3f}  {delta:>+8.3f}  {impact}")

    results.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  RANKED BY IMPACT (most → least):")
    for col, rmse, delta, impact in results:
        bar = "█" * max(0, int(delta * 20))
        print(f"    {col:>25s}  Δ={delta:>+.3f}  {bar}  {impact}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    run_loo = "--loo" in sys.argv

    print("=" * 65)
    print("  MODEL VALIDATION & IMPROVEMENT")
    print("=" * 65)

    t0 = time.time()

    sites_df, X, feature_cols = load_data()
    y = generate_labels(X, feature_cols)

    print(f"\n  Data: {X.shape[0]} sites × {X.shape[1]} features")
    print(f"  Label: nexus_score (self-supervised)")

    params = {
        "objective": "reg:squarederror",
        "max_depth": 6, "eta": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 5, "eval_metric": "rmse",
    }

    # 1. K-Fold CV
    kfold_preds, _, _ = kfold_cv(X, y, feature_cols, params, n_folds=5)
    kfold_cv(X, y, feature_cols, params, n_folds=10)

    # 2. LOO-CV (optional — slow)
    if run_loo:
        loo_cv(X, y, feature_cols, params, n_rounds=200, max_samples=500)
    else:
        print(f"\n  [SKIP] LOO-CV — run with --loo flag (takes ~20 min)")

    # 3. Hyperparameter tuning
    best_params, best_rounds, best_rmse = tune_hyperparams(X, y, feature_cols)

    # 4. Re-run K-Fold with best params
    print(f"\n  Re-validating with tuned params:")
    tuned_params = {**params, **best_params}
    kfold_cv(X, y, feature_cols, tuned_params, n_folds=5)

    # 5. Learning curve
    learning_curve(X, y, feature_cols, tuned_params, n_rounds=best_rounds)

    # 6. Feature ablation
    feature_ablation(X, y, feature_cols, tuned_params, n_rounds=best_rounds)

    elapsed = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f"  VALIDATION COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
