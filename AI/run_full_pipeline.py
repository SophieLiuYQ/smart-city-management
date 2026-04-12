"""
Full Pipeline: Data → 3 Models → BESS Simulation

Runs everything in sequence:
  Step 1: Load 12 raw CSVs, build 30 features
  Step 2: Generate labels (same for all models)
  Step 3: Split 70/15/15 (seed=42, same for all)
  Step 4: Train & score — Linear, XGBoost, CatBoost
  Step 5: BESS simulation on best model's top sites
  Step 6: Save all results + comparison report

Usage:
    python AI/run_full_pipeline.py
    python AI/run_full_pipeline.py --top 50

    # On GB10:
    RAW_DIR=/home/acergn100_6/smart-city-management/data/raw python3 AI/run_full_pipeline.py
"""
import json
import os
import sys
import time
import platform
import subprocess
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

# Parse args
TOP_N = 50
for i, arg in enumerate(sys.argv[1:]):
    if arg == "--top" and i + 1 < len(sys.argv) - 1:
        TOP_N = int(sys.argv[i + 2])


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Device Info
# ═══════════════════════════════════════════════════════════════════════════

def get_device_info():
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_cores": os.cpu_count(),
        "has_gpu": False,
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            info["gpu_name"] = parts[0]
            info["gpu_memory_mb"] = int(parts[1])
            info["has_gpu"] = True
    except Exception:
        pass
    return info


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Load Data
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Labels + Split
# ═══════════════════════════════════════════════════════════════════════════

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


def make_split(n):
    np.random.seed(42)
    indices = np.random.permutation(n)
    t_end = int(n * 0.70)
    v_end = int(n * 0.85)
    return {
        "train": indices[:t_end],
        "val": indices[t_end:v_end],
        "test": indices[v_end:],
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Three Models
# ═══════════════════════════════════════════════════════════════════════════

def train_linear(X, labels, split, feature_cols):
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X[split["train"]])
    Xv = scaler.transform(X[split["val"]])
    Xs = scaler.transform(X[split["test"]])
    Xa = scaler.transform(X)

    results = {}
    for name in ["energy", "waste", "nexus"]:
        t0 = time.time()
        model = LinearRegression()
        model.fit(Xt, labels[name][split["train"]])
        train_time = time.time() - t0

        results[name] = {
            "test_r2": r2_score(labels[name][split["test"]], model.predict(Xs)),
            "test_rmse": np.sqrt(mean_squared_error(labels[name][split["test"]], model.predict(Xs))),
            "train_time": train_time,
            "all_preds": np.clip(model.predict(Xa), 0, 100).astype(int),
            "device": "CPU",
        }
    return results


def train_xgboost(X, labels, split, feature_cols):
    import xgboost as xgb

    try:
        gpu_p = {"device": "cuda", "tree_method": "hist"}
        xgb.DMatrix(X[:5], label=labels["energy"][:5])
        device = "GPU"
    except:
        gpu_p = {}
        device = "CPU"

    results = {}
    for name in ["energy", "waste", "nexus"]:
        dtrain = xgb.DMatrix(X[split["train"]], label=labels[name][split["train"]], feature_names=feature_cols)
        dval = xgb.DMatrix(X[split["val"]], label=labels[name][split["val"]], feature_names=feature_cols)
        dtest = xgb.DMatrix(X[split["test"]], label=labels[name][split["test"]], feature_names=feature_cols)
        dall = xgb.DMatrix(X, feature_names=feature_cols)

        params = {"objective": "reg:squarederror", "max_depth": 3, "eta": 0.1,
                  "subsample": 0.7, "colsample_bytree": 0.8, "min_child_weight": 10,
                  "eval_metric": "rmse", **gpu_p}

        t0 = time.time()
        model = xgb.train(params, dtrain, num_boost_round=500,
                          evals=[(dtrain, "train"), (dval, "val")],
                          early_stopping_rounds=20, verbose_eval=False)
        train_time = time.time() - t0

        test_preds = model.predict(dtest)
        results[name] = {
            "test_r2": r2_score(labels[name][split["test"]], test_preds),
            "test_rmse": np.sqrt(mean_squared_error(labels[name][split["test"]], test_preds)),
            "train_time": train_time,
            "rounds": model.best_iteration + 1,
            "all_preds": np.clip(model.predict(dall), 0, 100).astype(int),
            "device": device,
            "importance": model.get_score(importance_type="gain") if name == "nexus" else None,
        }
    return results


def train_catboost(X, labels, split, feature_cols):
    from catboost import CatBoostRegressor, Pool

    try:
        t = CatBoostRegressor(iterations=1, task_type="GPU", verbose=0)
        t.fit(Pool(X[split["train"][:10]], labels["energy"][split["train"][:10]]))
        device, depth = "GPU", 6
    except:
        device, depth = "CPU", 4

    results = {}
    for name in ["energy", "waste", "nexus"]:
        tp = Pool(X[split["train"]], labels[name][split["train"]], feature_names=feature_cols)
        vp = Pool(X[split["val"]], labels[name][split["val"]], feature_names=feature_cols)
        sp = Pool(X[split["test"]], labels[name][split["test"]], feature_names=feature_cols)
        ap = Pool(X, feature_names=feature_cols)

        model = CatBoostRegressor(iterations=500, depth=depth, learning_rate=0.1,
                                  l2_leaf_reg=5, subsample=0.8, loss_function="RMSE",
                                  early_stopping_rounds=20, verbose=0, task_type=device)

        t0 = time.time()
        model.fit(tp, eval_set=vp, use_best_model=True)
        train_time = time.time() - t0

        test_preds = model.predict(sp)
        imp = dict(zip(feature_cols, model.get_feature_importance())) if name == "nexus" else None

        results[name] = {
            "test_r2": r2_score(labels[name][split["test"]], test_preds),
            "test_rmse": np.sqrt(mean_squared_error(labels[name][split["test"]], test_preds)),
            "train_time": train_time,
            "rounds": model.get_best_iteration() + 1,
            "all_preds": np.clip(model.predict(ap), 0, 100).astype(int),
            "device": device,
            "importance": imp,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: BESS Simulation (imported)
# ═══════════════════════════════════════════════════════════════════════════

def run_bess_on_best(sites_df, best_preds, best_name, feature_cols):
    """Attach best model's scores to sites, save, then run BESS simulation."""
    site_col = "Site" if "Site" in sites_df.columns else "site"

    ranked = sites_df.copy()
    ranked["energy_score"] = best_preds["energy"]
    ranked["waste_score"] = best_preds["waste"]
    ranked["nexus_score"] = best_preds["nexus"]

    # Dedup
    ranked = ranked.sort_values("nexus_score", ascending=False)
    ranked = ranked.drop_duplicates(subset=[site_col], keep="first")
    ranked = ranked.reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)

    # Save for BESS simulation to read
    temp_path = os.path.join(OUT_DIR, "ranked_sites_pipeline.parquet")
    ranked.to_parquet(temp_path, index=False)

    # Run BESS simulation
    from bess_simulation import run_simulation
    results = run_simulation(input_file="ranked_sites_pipeline.parquet", top_n=TOP_N)

    return ranked, results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    total_start = time.time()

    print("╔" + "═" * 78 + "╗")
    print("║" + "  FULL PIPELINE: Data → Models → BESS Simulation".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    # ── Step 1: Device ──
    print("\n┌── STEP 1: DEVICE INFO ──")
    device = get_device_info()
    print(f"│  Platform:  {device['platform']}")
    print(f"│  Processor: {device['processor']}")
    print(f"│  CPU Cores: {device['cpu_cores']}")
    if device["has_gpu"]:
        print(f"│  GPU:       {device['gpu_name']} ({device['gpu_memory_mb']} MB)")
    else:
        print(f"│  GPU:       None")
    print(f"└{'─'*40}")

    # ── Step 2: Load data ──
    print("\n┌── STEP 2: LOAD 12 RAW DATASETS ──")
    t0 = time.time()
    sites_df, X, feature_cols = load_data()
    data_time = time.time() - t0
    print(f"│  {X.shape[0]} sites × {X.shape[1]} features in {data_time:.1f}s")
    print(f"└{'─'*40}")

    # ── Step 3: Labels + Split ──
    print("\n┌── STEP 3: LABELS + SPLIT ──")
    labels = generate_labels(X, feature_cols)
    split = make_split(len(X))
    print(f"│  Train: {len(split['train'])} | Val: {len(split['val'])} | Test: {len(split['test'])}")
    print(f"│  Seed: 42 (identical across all models)")
    print(f"└{'─'*40}")

    # ── Step 4: Train 3 models ──
    print("\n┌── STEP 4: TRAIN 3 MODELS ──")

    print("│")
    print("│  [1/3] Linear Regression...")
    t0 = time.time()
    lr = train_linear(X, labels, split, feature_cols)
    lr_time = time.time() - t0
    lr_r2 = np.mean([lr[s]["test_r2"] for s in ["energy", "waste", "nexus"]])
    print(f"│        R²={lr_r2:.3f}  Time={lr_time:.3f}s  Device={lr['energy']['device']}")

    print("│")
    print("│  [2/3] XGBoost...")
    t0 = time.time()
    xgb_res = train_xgboost(X, labels, split, feature_cols)
    xgb_time = time.time() - t0
    xgb_r2 = np.mean([xgb_res[s]["test_r2"] for s in ["energy", "waste", "nexus"]])
    print(f"│        R²={xgb_r2:.3f}  Time={xgb_time:.3f}s  Device={xgb_res['energy']['device']}")

    print("│")
    print("│  [3/3] CatBoost...")
    t0 = time.time()
    cb = train_catboost(X, labels, split, feature_cols)
    cb_time = time.time() - t0
    cb_r2 = np.mean([cb[s]["test_r2"] for s in ["energy", "waste", "nexus"]])
    print(f"│        R²={cb_r2:.3f}  Time={cb_time:.3f}s  Device={cb['energy']['device']}")

    # Pick best model
    model_scores = {"Linear": (lr_r2, lr), "XGBoost": (xgb_r2, xgb_res), "CatBoost": (cb_r2, cb)}
    best_name = max(model_scores, key=lambda k: model_scores[k][0])
    best_r2, best_results = model_scores[best_name]

    print("│")
    print(f"│  ┌─────────────────────────────────────────────┐")
    print(f"│  │  {'Model':12s}  {'R²':>7s}  {'Time':>8s}  {'Device':>6s}  │")
    print(f"│  ├─────────────────────────────────────────────┤")
    print(f"│  │  {'Linear':12s}  {lr_r2:>7.3f}  {lr_time:>7.3f}s  {'CPU':>6s}  │")
    print(f"│  │  {'XGBoost':12s}  {xgb_r2:>7.3f}  {xgb_time:>7.3f}s  {xgb_res['energy']['device']:>6s}  │")
    print(f"│  │  {'CatBoost':12s}  {cb_r2:>7.3f}  {cb_time:>7.3f}s  {cb['energy']['device']:>6s}  │")
    print(f"│  ├─────────────────────────────────────────────┤")
    print(f"│  │  🏆 Winner: {best_name} (R²={best_r2:.3f})" + " " * (45 - 20 - len(best_name)) + "│")
    print(f"│  └─────────────────────────────────────────────┘")
    print(f"└{'─'*40}")

    # ── Step 5: BESS Simulation ──
    print(f"\n┌── STEP 5: BESS SIMULATION (top {TOP_N} sites from {best_name}) ──")
    best_preds = {s: best_results[s]["all_preds"] for s in ["energy", "waste", "nexus"]}
    t0 = time.time()
    ranked, bess_results = run_bess_on_best(sites_df, best_preds, best_name, feature_cols)
    bess_time = time.time() - t0
    print(f"└{'─'*40}")

    # ── Step 6: Save everything ──
    print(f"\n┌── STEP 6: SAVE RESULTS ──")

    # Save ranked sites from best model
    ranked_path = os.path.join(OUT_DIR, "ranked_sites_pipeline.parquet")
    print(f"│  {ranked_path}")

    # Save top 50
    top50 = ranked.head(50)
    top50_json = os.path.join(OUT_DIR, "top50_pipeline.json")
    top50.to_json(top50_json, orient="records", indent=2)
    print(f"│  {top50_json}")

    # Save BESS results (already saved by bess_simulation.py)
    bess_path = os.path.join(OUT_DIR, "bess_simulation_results.json")
    print(f"│  {bess_path}")

    # Save pipeline report
    total_time = time.time() - total_start
    report = {
        "pipeline_version": "1.0",
        "device": device,
        "data": {"sites": len(sites_df), "features": len(feature_cols),
                 "train": len(split["train"]), "val": len(split["val"]), "test": len(split["test"])},
        "models": {
            "Linear": {"r2": round(lr_r2, 4), "time_sec": round(lr_time, 3), "device": "CPU"},
            "XGBoost": {"r2": round(xgb_r2, 4), "time_sec": round(xgb_time, 3),
                        "device": xgb_res["energy"]["device"],
                        "rounds": {s: xgb_res[s]["rounds"] for s in ["energy", "waste", "nexus"]}},
            "CatBoost": {"r2": round(cb_r2, 4), "time_sec": round(cb_time, 3),
                         "device": cb["energy"]["device"],
                         "rounds": {s: cb[s]["rounds"] for s in ["energy", "waste", "nexus"]}},
        },
        "best_model": best_name,
        "best_r2": round(best_r2, 4),
        "bess_simulation": {
            "sites_simulated": len(bess_results),
            "total_bess_kwh": sum(r["bess"]["capacity_kwh"] for r in bess_results),
            "annual_savings_usd": sum(r["annual_results"]["savings_usd"] for r in bess_results),
            "annual_co2_tons": round(sum(r["annual_results"]["co2_offset_tons"] for r in bess_results), 1),
            "avg_payback_years": round(
                sum(r["bess"]["install_cost_usd"] for r in bess_results) /
                max(sum(r["annual_results"]["savings_usd"] for r in bess_results), 1), 1
            ),
        },
        "timing": {
            "data_loading_sec": round(data_time, 1),
            "linear_sec": round(lr_time, 3),
            "xgboost_sec": round(xgb_time, 3),
            "catboost_sec": round(cb_time, 3),
            "bess_sim_sec": round(bess_time, 1),
            "total_sec": round(total_time, 1),
        },
    }

    report_path = os.path.join(OUT_DIR, "pipeline_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"│  {report_path}")
    print(f"└{'─'*40}")

    # ── Final Summary ──
    bess = report["bess_simulation"]

    print(f"\n╔{'═'*78}╗")
    print(f"║{'  PIPELINE COMPLETE'.center(78)}║")
    print(f"╠{'═'*78}╣")
    print(f"║  Total time:        {total_time:>8.1f}s" + " " * (78 - 31) + "║")
    print(f"║  Data loading:      {data_time:>8.1f}s" + " " * (78 - 31) + "║")
    print(f"║  3 models trained:  {lr_time+xgb_time+cb_time:>8.1f}s" + " " * (78 - 31) + "║")
    print(f"║  BESS simulation:   {bess_time:>8.1f}s" + " " * (78 - 31) + "║")
    print(f"╠{'═'*78}╣")
    print(f"║  Best model:     {best_name} (R²={best_r2:.3f})" + " " * (78 - 34 - len(best_name)) + "║")
    print(f"║  Sites scored:   {len(ranked):,}" + " " * (78 - 19 - len(f'{len(ranked):,}')) + "║")
    print(f"║  BESS simulated: {bess['sites_simulated']} sites, {bess['total_bess_kwh']:,} kWh" + " " * max(0, 78 - 35 - len(f"{bess['total_bess_kwh']:,}")) + "║")
    print(f"║  Annual savings: ${bess['annual_savings_usd']:,}" + " " * max(0, 78 - 20 - len(f"${bess['annual_savings_usd']:,}")) + "║")
    print(f"║  CO₂ offset:     {bess['annual_co2_tons']:,} tons/yr" + " " * max(0, 78 - 27 - len(f"{bess['annual_co2_tons']:,}")) + "║")
    print(f"║  Payback:        {bess['avg_payback_years']} years" + " " * max(0, 78 - 24 - len(f"{bess['avg_payback_years']}")) + "║")
    print(f"╚{'═'*78}╝")


if __name__ == "__main__":
    main()
