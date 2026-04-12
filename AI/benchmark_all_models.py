"""
Model Benchmark — Linear vs XGBoost vs CatBoost

Single script that runs all three models on the same data, same split,
and outputs one comparison table with accuracy + device performance.

Usage:
    python AI/benchmark_all_models.py
    RAW_DIR=/home/acergn100_6/smart-city-management/data/raw python3 AI/benchmark_all_models.py
"""
import json
import os
import sys
import time
import platform
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.getenv("RAW_DIR", os.path.join(BASE_DIR, "data", "raw"))
OUT_DIR = os.getenv("OUT_DIR", os.path.join(BASE_DIR, "data", "gold"))
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, "AI"))


# ---------------------------------------------------------------------------
# Device info
# ---------------------------------------------------------------------------
def get_device_info():
    """Collect hardware information."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_cores": os.cpu_count(),
    }

    # GPU info
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            info["gpu_name"] = parts[0]
            info["gpu_memory_mb"] = int(parts[1])
            info["gpu_driver"] = parts[2]
            info["gpu_temp_c"] = int(parts[3])
            info["has_gpu"] = True
        else:
            info["has_gpu"] = False
    except Exception:
        info["has_gpu"] = False

    # XGBoost version
    try:
        import xgboost
        info["xgboost_version"] = xgboost.__version__
    except ImportError:
        info["xgboost_version"] = "not installed"

    # CatBoost version
    try:
        import catboost
        info["catboost_version"] = catboost.__version__
    except ImportError:
        info["catboost_version"] = "not installed"

    return info


# ---------------------------------------------------------------------------
# Load data (reuse XGBoost pipeline)
# ---------------------------------------------------------------------------
def load_data():
    from score_raw_xgboost_no_ej import (
        load_e5, load_e3, load_e4, load_e7, load_e10,
        load_w1, load_w2, load_w7, load_w8,
        build_feature_matrix
    )
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
# Generate labels (same for all models)
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

    return {
        "energy": np.clip(energy, 0, 100),
        "waste": np.clip(waste, 0, 100),
        "nexus": np.clip(nexus, 0, 100),
    }


# ---------------------------------------------------------------------------
# Split data (same for all models)
# ---------------------------------------------------------------------------
def split_data(X, labels):
    n = len(X)
    np.random.seed(42)
    indices = np.random.permutation(n)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    split = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }
    return split


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def run_linear(X, labels, split, feature_cols):
    """Linear Regression — CPU only."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[split["train"]])
    X_val = scaler.transform(X[split["val"]])
    X_test = scaler.transform(X[split["test"]])

    results = {}
    for name in ["energy", "waste", "nexus"]:
        y_train = labels[name][split["train"]]
        y_val = labels[name][split["val"]]
        y_test = labels[name][split["test"]]

        t0 = time.time()
        model = LinearRegression()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        test_preds = model.predict(X_test)
        predict_time = time.time() - t0

        results[name] = {
            "train_rmse": np.sqrt(mean_squared_error(y_train, model.predict(X_train))),
            "val_rmse": np.sqrt(mean_squared_error(y_val, model.predict(X_val))),
            "test_rmse": np.sqrt(mean_squared_error(y_test, test_preds)),
            "test_r2": r2_score(y_test, test_preds),
            "test_mae": mean_absolute_error(y_test, test_preds),
            "train_time": train_time,
            "predict_time": predict_time,
            "rounds": "N/A",
            "device": "CPU",
        }

    return results


def run_xgboost(X, labels, split, feature_cols):
    """XGBoost — auto GPU/CPU."""
    import xgboost as xgb

    # Detect GPU
    try:
        gpu_params = {"device": "cuda", "tree_method": "hist"}
        xgb.DMatrix(X[:5], label=labels["energy"][:5])
        device = "GPU"
    except Exception:
        gpu_params = {}
        device = "CPU"

    results = {}
    for name in ["energy", "waste", "nexus"]:
        y_train = labels[name][split["train"]]
        y_val = labels[name][split["val"]]
        y_test = labels[name][split["test"]]

        dtrain = xgb.DMatrix(X[split["train"]], label=y_train, feature_names=feature_cols)
        dval = xgb.DMatrix(X[split["val"]], label=y_val, feature_names=feature_cols)
        dtest = xgb.DMatrix(X[split["test"]], label=y_test, feature_names=feature_cols)

        params = {
            "objective": "reg:squarederror",
            "max_depth": 3, "eta": 0.1,
            "subsample": 0.7, "colsample_bytree": 0.8,
            "min_child_weight": 10, "eval_metric": "rmse",
            **gpu_params,
        }

        t0 = time.time()
        model = xgb.train(
            params, dtrain, num_boost_round=500,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=20, verbose_eval=False,
        )
        train_time = time.time() - t0

        t0 = time.time()
        test_preds = model.predict(dtest)
        predict_time = time.time() - t0

        results[name] = {
            "train_rmse": np.sqrt(mean_squared_error(y_train, model.predict(dtrain))),
            "val_rmse": np.sqrt(mean_squared_error(y_val, model.predict(dval))),
            "test_rmse": np.sqrt(mean_squared_error(y_test, test_preds)),
            "test_r2": r2_score(y_test, test_preds),
            "test_mae": mean_absolute_error(y_test, test_preds),
            "train_time": train_time,
            "predict_time": predict_time,
            "rounds": model.best_iteration + 1,
            "device": device,
            "importance": model.get_score(importance_type="gain") if name == "nexus" else None,
        }

    return results


def run_catboost(X, labels, split, feature_cols):
    """CatBoost — auto GPU/CPU."""
    from catboost import CatBoostRegressor, Pool

    # Detect GPU
    try:
        test_model = CatBoostRegressor(iterations=1, task_type="GPU", verbose=0)
        test_model.fit(Pool(X[split["train"][:10]], labels["energy"][split["train"][:10]]))
        device = "GPU"
        depth = 6
    except Exception:
        device = "CPU"
        depth = 4

    results = {}
    for name in ["energy", "waste", "nexus"]:
        y_train = labels[name][split["train"]]
        y_val = labels[name][split["val"]]
        y_test = labels[name][split["test"]]

        train_pool = Pool(X[split["train"]], y_train, feature_names=feature_cols)
        val_pool = Pool(X[split["val"]], y_val, feature_names=feature_cols)
        test_pool = Pool(X[split["test"]], y_test, feature_names=feature_cols)

        model = CatBoostRegressor(
            iterations=500, depth=depth, learning_rate=0.1,
            l2_leaf_reg=5, subsample=0.8, loss_function="RMSE",
            early_stopping_rounds=20, verbose=0, task_type=device,
        )

        t0 = time.time()
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        train_time = time.time() - t0

        t0 = time.time()
        test_preds = model.predict(test_pool)
        predict_time = time.time() - t0

        imp = None
        if name == "nexus":
            raw_imp = model.get_feature_importance()
            imp = dict(zip(feature_cols, raw_imp))

        results[name] = {
            "train_rmse": np.sqrt(mean_squared_error(y_train, model.predict(train_pool))),
            "val_rmse": np.sqrt(mean_squared_error(y_val, model.predict(val_pool))),
            "test_rmse": np.sqrt(mean_squared_error(y_test, test_preds)),
            "test_r2": r2_score(y_test, test_preds),
            "test_mae": mean_absolute_error(y_test, test_preds),
            "train_time": train_time,
            "predict_time": predict_time,
            "rounds": model.get_best_iteration() + 1,
            "device": device,
            "importance": imp,
        }

    return results


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
def print_results(all_results, device_info, split, feature_cols, labels):
    print("\n" + "=" * 90)
    print("  BENCHMARK RESULTS")
    print("=" * 90)

    # Device info
    print(f"\n  DEVICE:")
    print(f"    Platform:   {device_info['platform']}")
    print(f"    Processor:  {device_info['processor']}")
    print(f"    CPU Cores:  {device_info['cpu_cores']}")
    if device_info.get("has_gpu"):
        print(f"    GPU:        {device_info['gpu_name']}")
        print(f"    GPU Memory: {device_info['gpu_memory_mb']} MB")
        print(f"    GPU Driver: {device_info['gpu_driver']}")
        print(f"    GPU Temp:   {device_info['gpu_temp_c']}°C")
    else:
        print(f"    GPU:        None detected")
    print(f"    XGBoost:    {device_info['xgboost_version']}")
    print(f"    CatBoost:   {device_info['catboost_version']}")

    # Data info
    print(f"\n  DATA:")
    print(f"    Total sites:  {sum(len(v) for v in split.values())}")
    print(f"    Train:        {len(split['train'])} (70%)")
    print(f"    Validation:   {len(split['val'])} (15%)")
    print(f"    Test:         {len(split['test'])} (15%)")
    print(f"    Features:     {len(feature_cols)}")
    print(f"    Random seed:  42")

    model_names = list(all_results.keys())

    # Per-score comparison
    for score in ["energy", "waste", "nexus"]:
        print(f"\n  ── {score}_score ──")
        print(f"  {'Model':15s}  {'Device':>6s}  {'Train':>8s}  {'Val':>8s}  {'Test':>8s}  "
              f"{'R²':>7s}  {'MAE':>6s}  {'Rounds':>6s}  {'Train(s)':>8s}  {'Pred(ms)':>8s}")
        print(f"  {'─'*90}")

        for model_name in model_names:
            r = all_results[model_name][score]
            rounds = str(r["rounds"])
            pred_ms = r["predict_time"] * 1000

            print(f"  {model_name:15s}  {r['device']:>6s}  {r['train_rmse']:>8.2f}  {r['val_rmse']:>8.2f}  "
                  f"{r['test_rmse']:>8.2f}  {r['test_r2']:>7.3f}  {r['test_mae']:>6.2f}  "
                  f"{rounds:>6s}  {r['train_time']:>8.3f}  {pred_ms:>8.1f}")

    # Summary table
    print(f"\n  {'═'*90}")
    print(f"  SUMMARY — Average across energy + waste + nexus")
    print(f"  {'═'*90}")
    print(f"  {'Model':15s}  {'Device':>6s}  {'Avg R²':>7s}  {'Avg RMSE':>8s}  {'Avg MAE':>7s}  "
          f"{'Total Train':>11s}  {'Total Pred':>10s}  {'Overfit':>7s}")
    print(f"  {'─'*80}")

    best_r2 = -1
    best_model = ""

    for model_name in model_names:
        scores = ["energy", "waste", "nexus"]
        avg_r2 = np.mean([all_results[model_name][s]["test_r2"] for s in scores])
        avg_rmse = np.mean([all_results[model_name][s]["test_rmse"] for s in scores])
        avg_mae = np.mean([all_results[model_name][s]["test_mae"] for s in scores])
        total_train = sum(all_results[model_name][s]["train_time"] for s in scores)
        total_pred = sum(all_results[model_name][s]["predict_time"] for s in scores) * 1000
        device = all_results[model_name]["energy"]["device"]

        avg_train_rmse = np.mean([all_results[model_name][s]["train_rmse"] for s in scores])
        avg_test_rmse = np.mean([all_results[model_name][s]["test_rmse"] for s in scores])
        gap = avg_test_rmse - avg_train_rmse
        overfit = "YES" if gap > 2 else "mild" if gap > 1 else "no"

        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_model = model_name

        print(f"  {model_name:15s}  {device:>6s}  {avg_r2:>7.3f}  {avg_rmse:>8.2f}  {avg_mae:>7.2f}  "
              f"{total_train:>10.3f}s  {total_pred:>9.1f}ms  {overfit:>7s}")

    print(f"  {'─'*80}")
    print(f"  🏆 Winner: {best_model} (R²={best_r2:.3f})")

    # Feature importance comparison (nexus only)
    print(f"\n  FEATURE IMPORTANCE — nexus_score (top 10)")
    print(f"  {'Feature':25s}", end="")
    for model_name in model_names:
        print(f"  {model_name:>12s}", end="")
    print()
    print(f"  {'─'*(25 + 14 * len(model_names))}")

    # Collect all importances
    all_features = set()
    for model_name in model_names:
        imp = all_results[model_name]["nexus"].get("importance")
        if imp:
            all_features.update(imp.keys())

    # Rank by first model that has importance
    ranked_features = []
    for model_name in model_names:
        imp = all_results[model_name]["nexus"].get("importance")
        if imp:
            ranked_features = sorted(imp.keys(), key=lambda x: imp.get(x, 0), reverse=True)[:10]
            break

    for feat in ranked_features:
        print(f"  {feat:25s}", end="")
        for model_name in model_names:
            imp = all_results[model_name]["nexus"].get("importance")
            if imp and feat in imp:
                print(f"  {imp[feat]:>12.1f}", end="")
            else:
                print(f"  {'N/A':>12s}", end="")
        print()

    # Speed comparison
    print(f"\n  SPEED COMPARISON")
    print(f"  {'Model':15s}  {'Train 3 models':>14s}  {'Predict 641':>11s}  {'Speedup vs Linear':>18s}")
    print(f"  {'─'*65}")

    linear_train = sum(all_results["Linear"][s]["train_time"] for s in ["energy", "waste", "nexus"])

    for model_name in model_names:
        total_train = sum(all_results[model_name][s]["train_time"] for s in ["energy", "waste", "nexus"])
        total_pred = sum(all_results[model_name][s]["predict_time"] for s in ["energy", "waste", "nexus"]) * 1000
        speedup = linear_train / total_train if total_train > 0 else 0
        print(f"  {model_name:15s}  {total_train:>13.3f}s  {total_pred:>10.1f}ms  {speedup:>17.1f}x")

    return best_model, best_r2


# ---------------------------------------------------------------------------
# Save benchmark report
# ---------------------------------------------------------------------------
def save_report(all_results, device_info, best_model, best_r2):
    report = {
        "device": device_info,
        "best_model": best_model,
        "best_r2": round(best_r2, 4),
        "models": {},
    }

    for model_name, model_results in all_results.items():
        report["models"][model_name] = {}
        for score in ["energy", "waste", "nexus"]:
            r = model_results[score]
            report["models"][model_name][score] = {
                "test_r2": round(r["test_r2"], 4),
                "test_rmse": round(r["test_rmse"], 3),
                "test_mae": round(r["test_mae"], 3),
                "train_rmse": round(r["train_rmse"], 3),
                "val_rmse": round(r["val_rmse"], 3),
                "train_time_sec": round(r["train_time"], 4),
                "predict_time_ms": round(r["predict_time"] * 1000, 2),
                "rounds": r["rounds"],
                "device": r["device"],
            }

    out_path = os.path.join(OUT_DIR, "benchmark_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SAVED] {out_path}")

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 90)
    print("  MODEL BENCHMARK — Linear vs XGBoost vs CatBoost")
    print(f"  Raw data: {RAW_DIR}")
    print("=" * 90)

    total_start = time.time()

    # Device info
    print("\n── DEVICE INFO ──")
    device_info = get_device_info()

    # Load data
    print("\n── LOAD DATA ──")
    sites_df, X, feature_cols = load_data()
    print(f"\n  {X.shape[0]} sites × {X.shape[1]} features")

    # Labels
    print("\n── GENERATE LABELS ──")
    labels = generate_labels(X, feature_cols)

    # Split
    split = split_data(X, labels)
    print(f"\n  Train={len(split['train'])} | Val={len(split['val'])} | Test={len(split['test'])}")

    # Run all models
    all_results = {}

    print("\n── LINEAR REGRESSION ──")
    t0 = time.time()
    all_results["Linear"] = run_linear(X, labels, split, feature_cols)
    print(f"  Done ({time.time()-t0:.2f}s)")

    print("\n── XGBOOST ──")
    t0 = time.time()
    all_results["XGBoost"] = run_xgboost(X, labels, split, feature_cols)
    print(f"  Done ({time.time()-t0:.2f}s)")

    print("\n── CATBOOST ──")
    t0 = time.time()
    all_results["CatBoost"] = run_catboost(X, labels, split, feature_cols)
    print(f"  Done ({time.time()-t0:.2f}s)")

    # Print results
    best_model, best_r2 = print_results(all_results, device_info, split, feature_cols, labels)

    # Save report
    save_report(all_results, device_info, best_model, best_r2)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 90}")
    print(f"  TOTAL BENCHMARK TIME: {total_elapsed:.1f}s")
    print(f"  WINNER: {best_model} (R²={best_r2:.3f})")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
