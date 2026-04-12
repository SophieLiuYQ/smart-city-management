"""
Qwen 3 80B Site Scoring — with Train/Val/Test Split

Same pipeline as score_raw_xgboost_no_ej.py but uses Qwen LLM instead of XGBoost.
Proper 70/15/15 split to measure generalization.

Flow:
  1. Load & clean 12 raw CSVs → 4,268 sites × 30 features
  2. Generate proxy labels (same formula as XGBoost for fair comparison)
  3. Split 70/15/15
  4. Score TRAIN sites with Qwen → measure train RMSE
  5. Score TEST sites with Qwen → measure test RMSE + R²
  6. Score ALL sites for output

Usage (on GB10):
    QWEN_URL=http://localhost:3000/v1 python3 AI/score_raw_qwen_split.py

    # Adjust batch size:
    QWEN_URL=http://localhost:3000/v1 BATCH_SIZE=20 python3 AI/score_raw_qwen_split.py
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import rankdata

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.getenv("RAW_DIR", os.path.join(BASE_DIR, "data", "raw"))
OUT_DIR = os.getenv("OUT_DIR", os.path.join(BASE_DIR, "data", "gold"))
os.makedirs(OUT_DIR, exist_ok=True)

QWEN_URL = os.getenv("QWEN_URL", "http://localhost:3000/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3-80b")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
WORKERS = int(os.getenv("WORKERS", "2"))


# ---------------------------------------------------------------------------
# File resolver (same as score_raw_xgboost_no_ej.py)
# ---------------------------------------------------------------------------
FILE_PATTERNS = {
    "E1": ["E1_energy_cost_savings.csv", "Value_of_Energy_Cost_Savings"],
    "E3": ["E3_electric_consumption.csv", "Electric_Consumption_And_Cost"],
    "E4": ["E4_ev_fleet_stations.csv", "NYC_EV_Fleet_Station_Network"],
    "E5": ["E5_solar_readiness.csv", "City_of_New_York_Municipal_Solar", "Municipal_Solar"],
    "E7": ["E7_ll84_monthly.csv", "Local_Law_84_Monthly_Data"],
    "E10": ["E10_ll84_benchmarking.csv", "NYC_Building_Energy_and_Water_Data"],
    "W1": ["W1_dsny_monthly_tonnage.csv", "DSNY_Monthly_Tonnage"],
    "W2": ["W2_311_dsny.csv", "311_Service_Requests"],
    "W3": ["W3_litter_baskets.csv", "DSNY_Litter_Basket_Inventory"],
    "W7": ["W7_food_scrap_dropoffs.csv", "Food_Scrap_Drop-Off", "Food_Scrap_Drop_Off"],
    "W8": ["W8_disposal_facilities.csv", "Location_of_Disposal_Facilities"],
    "W12": ["W12_waste_characterization.csv", "DSNY_Waste_Characterization"],
}

def find_raw_file(key):
    patterns = FILE_PATTERNS.get(key, [])
    files = os.listdir(RAW_DIR)
    for p in patterns:
        if p in files:
            return os.path.join(RAW_DIR, p)
    for p in patterns:
        for f in files:
            if p.lower() in f.lower() and f.endswith(".csv"):
                return os.path.join(RAW_DIR, f)
    return None

def sf(s):
    return pd.to_numeric(s, errors="coerce")


# ---------------------------------------------------------------------------
# Reuse data loading from score_raw_xgboost_no_ej.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(BASE_DIR, "AI"))

def load_and_build_features():
    """Load raw data and build feature matrix — reuses XGBoost pipeline."""
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
# Generate proxy labels (same formula as XGBoost for fair comparison)
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
# Qwen scoring
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Score NYC municipal sites. Return ONLY a JSON array.

Input: site features (solar_kwh, sqft, roof, consumption, EV nearby, waste, composting, etc.)

For each site return:
{"site_index":N,"energy_score":0-100,"waste_score":0-100,"nexus_score":0-100}

Scoring:
- energy_score: high solar + high consumption + good roof + EV nearby = high
- waste_score: high refuse + low diversion + few compost sites + far from transfer = high
- nexus_score: high when BOTH energy and waste opportunity overlap

Return ONLY JSON array. No markdown. No explanation."""


def build_compact_features(sites_df, X, feature_cols, indices):
    """Build compact feature dicts for Qwen — minimal tokens."""
    features = []
    for idx in indices:
        row = sites_df.iloc[idx]
        f = {"site_index": int(idx)}
        for i, col in enumerate(feature_cols):
            val = float(X[idx, i])
            if val != 0:  # skip zeros to save tokens
                f[col] = round(val, 1)
        features.append(f)
    return features


def score_batch_qwen(client, batch_features, batch_id):
    """Send one batch to Qwen. Returns (batch_id, results, method)."""
    user_msg = f"Score these {len(batch_features)} NYC sites.\n{json.dumps(batch_features, default=str)}"

    try:
        response = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=4096,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        text = response.choices[0].message.content.strip()
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        results = json.loads(text)
        if not isinstance(results, list):
            raise ValueError(f"Expected list, got {type(results)}")
        return batch_id, results, "LLM"

    except Exception as e:
        # Fallback: return None scores (will be filled later)
        return batch_id, [{"site_index": f["site_index"],
                           "energy_score": -1, "waste_score": -1, "nexus_score": -1}
                          for f in batch_features], f"FAIL({e})"


def score_subset_with_qwen(client, sites_df, X, feature_cols, indices, label=""):
    """Score a subset of sites using Qwen in batches."""
    all_features = build_compact_features(sites_df, X, feature_cols, indices)

    batches = []
    for i in range(0, len(all_features), BATCH_SIZE):
        batches.append(all_features[i:i + BATCH_SIZE])

    all_results = {}
    total_batches = len(batches)
    success = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(score_batch_qwen, client, batch, idx): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_id, results, method = future.result()
            for r in results:
                all_results[r["site_index"]] = r
            if "FAIL" in method:
                fail += len(results)
            else:
                success += len(results)
            done = len(all_results)
            print(f"    [{label}] {done}/{len(indices)} ({method})")

    # Build score arrays aligned to indices
    energy_preds = np.zeros(len(indices))
    waste_preds = np.zeros(len(indices))
    nexus_preds = np.zeros(len(indices))

    for i, idx in enumerate(indices):
        r = all_results.get(int(idx), {})
        energy_preds[i] = r.get("energy_score", 50)
        waste_preds[i] = r.get("waste_score", 50)
        nexus_preds[i] = r.get("nexus_score", 50)

    # Replace -1 (failed) with 50
    energy_preds[energy_preds < 0] = 50
    waste_preds[waste_preds < 0] = 50
    nexus_preds[nexus_preds < 0] = 50

    return energy_preds, waste_preds, nexus_preds, success, fail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  QWEN SCORING — with Train/Val/Test Split")
    print(f"  Endpoint:   {QWEN_URL}")
    print(f"  Model:      {QWEN_MODEL}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Workers:    {WORKERS}")
    print("=" * 70)

    t_total = time.time()

    # Phase 1: Load data (reuse XGBoost pipeline)
    print("\n── PHASE 1: LOAD & BUILD FEATURES ──")
    sites_df, X, feature_cols = load_and_build_features()
    n = len(X)
    print(f"\n  {n} sites × {len(feature_cols)} features")

    # Phase 2: Generate labels
    print("\n── PHASE 2: GENERATE PROXY LABELS ──")
    energy_y, waste_y, nexus_y = generate_labels(X, feature_cols)
    print(f"  Labels generated (same formula as XGBoost for fair comparison)")

    # Phase 3: Split
    print("\n── PHASE 3: TRAIN/VAL/TEST SPLIT ──")
    np.random.seed(42)
    indices = np.random.permutation(n)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    print(f"  Train: {len(train_idx)} ({len(train_idx)/n*100:.0f}%)")
    print(f"  Val:   {len(val_idx)} ({len(val_idx)/n*100:.0f}%)")
    print(f"  Test:  {len(test_idx)} ({len(test_idx)/n*100:.0f}%)")

    # Phase 4: Connect to Qwen
    print("\n── PHASE 4: CONNECT TO QWEN ──")
    from openai import OpenAI
    client = OpenAI(base_url=QWEN_URL, api_key="not-needed")

    try:
        models = client.models.list()
        print(f"  Connected. Models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"  Connection failed: {e}")
        print(f"  Cannot run Qwen scoring without the endpoint.")
        sys.exit(1)

    # Phase 5: Score ALL sites with Qwen (LLM doesn't train — every site is "unseen")
    print("\n── PHASE 5: SCORE ALL SITES WITH QWEN ──")
    print(f"  Note: Qwen is pre-trained — no learning from our data.")
    print(f"  Every site is scored independently. We measure R² on the")
    print(f"  same test split as XGBoost for fair comparison.\n")

    t0 = time.time()
    all_e, all_w, all_n, all_ok, all_fail = score_subset_with_qwen(
        client, sites_df, X, feature_cols, np.arange(n), "ALL"
    )
    scoring_time = time.time() - t0

    # Extract train and test predictions from the full run
    train_e = all_e[train_idx]
    train_w = all_w[train_idx]
    train_n = all_n[train_idx]

    test_e = all_e[test_idx]
    test_w = all_w[test_idx]
    test_n = all_n[test_idx]

    # Train metrics
    train_rmse_e = np.sqrt(np.mean((train_e - energy_y[train_idx]) ** 2))
    train_rmse_w = np.sqrt(np.mean((train_w - waste_y[train_idx]) ** 2))
    train_rmse_n = np.sqrt(np.mean((train_n - nexus_y[train_idx]) ** 2))

    train_sample_size = len(train_idx)
    train_time = scoring_time  # all scored together

    # Phase 6: Measure on test split (same 641 sites as XGBoost)
    print("\n── PHASE 6: MEASURE ON TEST SPLIT ──")
    test_time = 0  # already scored above

    # Test metrics
    test_rmse_e = np.sqrt(np.mean((test_e - energy_y[test_idx]) ** 2))
    test_rmse_w = np.sqrt(np.mean((test_w - waste_y[test_idx]) ** 2))
    test_rmse_n = np.sqrt(np.mean((test_n - nexus_y[test_idx]) ** 2))

    ss_res_e = np.sum((energy_y[test_idx] - test_e) ** 2)
    ss_tot_e = np.sum((energy_y[test_idx] - energy_y[test_idx].mean()) ** 2)
    r2_e = 1 - ss_res_e / ss_tot_e if ss_tot_e > 0 else 0

    ss_res_w = np.sum((waste_y[test_idx] - test_w) ** 2)
    ss_tot_w = np.sum((waste_y[test_idx] - waste_y[test_idx].mean()) ** 2)
    r2_w = 1 - ss_res_w / ss_tot_w if ss_tot_w > 0 else 0

    ss_res_n = np.sum((nexus_y[test_idx] - test_n) ** 2)
    ss_tot_n = np.sum((nexus_y[test_idx] - nexus_y[test_idx].mean()) ** 2)
    r2_n = 1 - ss_res_n / ss_tot_n if ss_tot_n > 0 else 0

    print(f"\n  Test results ({len(test_idx)} sites, {test_time:.0f}s):")
    print(f"  {'Score':15s}  {'Train RMSE':>10s}  {'Test RMSE':>10s}  {'Test R²':>8s}  {'Overfit':>8s}")
    print(f"  {'─'*55}")

    for name, tr, te, r2 in [("energy_score", train_rmse_e, test_rmse_e, r2_e),
                               ("waste_score", train_rmse_w, test_rmse_w, r2_w),
                               ("nexus_score", train_rmse_n, test_rmse_n, r2_n)]:
        gap = te - tr
        overfit = "YES" if gap > 5 else "mild" if gap > 2 else "no"
        print(f"  {name:15s}  {tr:>10.2f}  {te:>10.2f}  {r2:>8.3f}  {overfit:>8s}")

    avg_r2 = (r2_e + r2_w + r2_n) / 3
    print(f"\n  Avg test R²: {avg_r2:.3f}")
    print(f"  LLM success: {test_ok}, failed: {test_fail}")

    # Phase 7: Save results (already scored all sites in Phase 5)
    print("\n── PHASE 7: SAVE RESULTS ──")
    all_time = scoring_time
    ranked = sites_df[["Site", "Address", "Borough", "Agency",
                        "Environmental Justice Area", "lat", "lon", "bbl"]].copy()
    ranked["energy_score"] = np.clip(all_e, 0, 100).astype(int)
    ranked["waste_score"] = np.clip(all_w, 0, 100).astype(int)
    ranked["nexus_score"] = np.clip(all_n, 0, 100).astype(int)

    # BESS recommendation (same logic as XGBoost script)
    solar = sites_df.get("solar_kwh", pd.Series(0, index=sites_df.index)).fillna(0).values
    ranked["recommended_bess_kwh"] = np.where(solar > 100000, 750,
                                     np.where(solar > 10000, 500,
                                     np.where(ranked["energy_score"] > 60, 250, 100)))
    ranked["estimated_annual_savings_usd"] = np.where(solar > 100000, 50000 + (solar * 0.05).astype(int),
                                             np.where(solar > 10000, 20000, 5000))

    # Recommendations
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

    # Save parquet
    pq_path = os.path.join(OUT_DIR, "ranked_sites_qwen_split.parquet")
    ranked.to_parquet(pq_path, index=False)
    print(f"  [SAVED] {pq_path}")

    # Save top 50 JSON
    top50 = ranked.head(50).to_dict(orient="records")
    json_path = os.path.join(OUT_DIR, "top50_qwen_split.json")
    with open(json_path, "w") as f:
        json.dump(top50, f, indent=2, default=str)
    print(f"  [SAVED] {json_path}")

    # Save metrics
    metrics = {
        "model": "qwen3-80b",
        "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "train_sample_size": train_sample_size,
        "test_metrics": {
            "energy": {"rmse": round(test_rmse_e, 2), "r2": round(r2_e, 3)},
            "waste": {"rmse": round(test_rmse_w, 2), "r2": round(r2_w, 3)},
            "nexus": {"rmse": round(test_rmse_n, 2), "r2": round(r2_n, 3)},
            "avg_r2": round(avg_r2, 3),
        },
        "llm_stats": {
            "total_scored": all_ok + all_fail,
            "llm_success": all_ok,
            "llm_failed": all_fail,
        },
        "timing": {
            "train_scoring_sec": round(train_time, 1),
            "test_scoring_sec": round(test_time, 1),
            "all_scoring_sec": round(all_time, 1),
            "total_sec": round(time.time() - t_total, 1),
        },
    }
    metrics_path = os.path.join(OUT_DIR, "qwen_split_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [SAVED] {metrics_path}")

    # Top 10
    print(f"\n  TOP 10 (by Nexus Score):")
    print(f"  {'─'*65}")
    for _, r in ranked.head(10).iterrows():
        print(f"  #{int(r['rank']):>3d}  E:{int(r['energy_score']):>3d}  W:{int(r['waste_score']):>3d}  "
              f"N:{int(r['nexus_score']):>3d}  {str(r['Borough']):>12s}  {str(r['Site'])[:35]}")
    print(f"  {'─'*65}")

    # Final summary
    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"  QWEN SCORING COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Model: {QWEN_MODEL}")
    print(f"  Split: {len(train_idx)}/{len(val_idx)}/{len(test_idx)} train/val/test")
    print(f"  Test R²: energy={r2_e:.3f}  waste={r2_w:.3f}  nexus={r2_n:.3f}  avg={avg_r2:.3f}")
    print(f"  LLM: {all_ok} success, {all_fail} failed")
    print(f"\n  vs XGBoost (for comparison):")
    print(f"    XGBoost test R²: energy=0.945  waste=0.884  nexus=0.866  avg=0.898")
    print(f"    XGBoost time: 5.9s")
    print(f"    Qwen time: {elapsed:.0f}s")
    print(f"{'=' * 70}")

    # Save comparison summary
    print(f"\n  Outputs:")
    print(f"    {pq_path}")
    print(f"    {json_path}")
    print(f"    {metrics_path}")


if __name__ == "__main__":
    main()
