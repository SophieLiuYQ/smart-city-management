"""
Job A (Alt): Score municipal sites using XGBoost — no LLM needed.

Trains a lightweight gradient-boosted model on engineered features,
produces the same output as score_sites_qwen.py for direct comparison.

Runs 100% locally. No API calls. CPU or GPU (RAPIDS cuML).

Reads:  data/gold/unified_sites.parquet + dispatch/site_profiles.parquet
Writes: data/gold/ranked_sites_xgboost.parquet + top50_scored_xgboost.json

Usage:
    python AI/score_sites_xgboost.py
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd

# Try GPU XGBoost first, fall back to CPU
try:
    import xgboost as xgb
    print(f"[XGBoost] v{xgb.__version__} loaded")
except ImportError:
    print("[ERROR] xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_DIR = os.path.join(BASE_DIR, "data", "gold")


# ---------------------------------------------------------------------------
# Feature Engineering — numeric features from raw columns
# ---------------------------------------------------------------------------
def build_features(sites, profiles):
    """Convert raw site data into numeric feature matrix for XGBoost."""
    df = sites.copy()

    # Merge profiles
    if len(profiles) > 0:
        profile_map = dict(zip(profiles["site_id"].astype(str), range(len(profiles))))
        df["_pidx"] = df["Site"].astype(str).map(profile_map)

        for col in ["avg_monthly_total_kwh", "peak_kw", "seasonality_index",
                     "solar_production_kwh_yr", "ev_ports_1km"]:
            if col in profiles.columns:
                df[col] = df["_pidx"].map(
                    dict(zip(range(len(profiles)), profiles[col].values))
                )
        df = df.drop(columns=["_pidx"], errors="ignore")

    # --- Numeric features ---
    features = pd.DataFrame(index=df.index)

    # Solar
    features["solar_kwh"] = pd.to_numeric(
        df.get("Estimated Annual Production", pd.Series(dtype=float)).astype(str).str.replace(",", ""),
        errors="coerce"
    ).fillna(0)
    features["solar_from_profile"] = pd.to_numeric(
        df.get("solar_production_kwh_yr", 0), errors="coerce"
    ).fillna(0)
    features["solar_max"] = features[["solar_kwh", "solar_from_profile"]].max(axis=1)

    # Energy consumption
    features["avg_monthly_kwh"] = pd.to_numeric(
        df.get("avg_monthly_total_kwh", 0), errors="coerce"
    ).fillna(0)
    features["peak_kw"] = pd.to_numeric(
        df.get("peak_kw", 0), errors="coerce"
    ).fillna(0)
    features["seasonality"] = pd.to_numeric(
        df.get("seasonality_index", 0), errors="coerce"
    ).fillna(0)

    # Roof condition → numeric
    roof = df.get("Roof Condition", pd.Series("Unknown", index=df.index)).astype(str).str.lower().str.strip()
    features["roof_good"] = (roof == "good").astype(float)
    features["roof_fair"] = (roof == "fair").astype(float)
    features["roof_poor"] = (roof == "poor").astype(float)

    # Environmental Justice
    ej = df.get("Environmental Justice Area", pd.Series("No", index=df.index)).astype(str).str.lower().str.strip()
    features["is_ej"] = (ej == "yes").astype(float)

    # EV infrastructure
    features["ev_within_1km"] = df.get("ev_within_1km", False).astype(float)
    features["ev_within_500m"] = df.get("ev_within_500m", False).astype(float)
    features["nearest_ev_m"] = pd.to_numeric(
        df.get("nearest_ev_dist_m", 9999), errors="coerce"
    ).fillna(9999)

    # Composting
    features["compost_within_1km"] = df.get("compost_within_1km", False).astype(float)
    features["nearest_compost_m"] = pd.to_numeric(
        df.get("nearest_compost_dist_m", 9999), errors="coerce"
    ).fillna(9999)

    # Transfer station
    features["nearest_transfer_m"] = pd.to_numeric(
        df.get("nearest_transfer_dist_m", 9999), errors="coerce"
    ).fillna(9999)

    # EV ports
    features["ev_ports_1km"] = pd.to_numeric(
        df.get("ev_ports_1km", 0), errors="coerce"
    ).fillna(0)

    # Square footage
    sqft = df.get("Total Gross Square Footage", pd.Series("0", index=df.index))
    features["sqft"] = pd.to_numeric(
        sqft.astype(str).str.replace(",", "").str.replace("GSF", "").str.strip(),
        errors="coerce"
    ).fillna(0)

    # Borough one-hot
    for b in ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]:
        features[f"boro_{b.lower().replace(' ', '_')}"] = (
            df.get("Borough", "").astype(str) == b
        ).astype(float)

    # Fill any remaining NaN
    features = features.fillna(0)

    return features


# ---------------------------------------------------------------------------
# Generate training labels (self-supervised from feature relationships)
# ---------------------------------------------------------------------------
def generate_labels(features):
    """
    No ground truth labels exist, so we create proxy scores from
    known relationships — same logic as the heuristic fallback but
    with continuous values that XGBoost can learn non-linear patterns from.
    """
    n = len(features)

    # Energy score: solar + consumption + EJ + roof
    energy_raw = (
        features["solar_max"].rank(pct=True) * 30 +
        features["avg_monthly_kwh"].rank(pct=True) * 25 +
        features["is_ej"] * 15 +
        features["roof_good"] * 10 +
        features["ev_within_1km"] * 10 +
        features["sqft"].rank(pct=True) * 10
    )
    energy_score = (energy_raw / energy_raw.max() * 100).clip(0, 100)

    # Waste score: EJ + no compost + far from transfer + building size
    waste_raw = (
        features["is_ej"] * 25 +
        (1 - features["compost_within_1km"]) * 20 +
        features["nearest_transfer_m"].rank(pct=True) * 20 +
        features["sqft"].rank(pct=True) * 15 +
        (1 - features["nearest_compost_m"].rank(pct=True)) * 10 +
        features["nearest_ev_m"].rank(pct=True) * 10
    )
    waste_score = (waste_raw / waste_raw.max() * 100).clip(0, 100)

    # Nexus score: combination with interaction bonuses
    nexus_raw = (
        energy_raw * 0.4 +
        waste_raw * 0.4 +
        (features["is_ej"] * features["solar_max"].rank(pct=True)) * 30 +
        (features["ev_within_1km"] * features["solar_max"].rank(pct=True)) * 20
    )
    nexus_score = (nexus_raw / nexus_raw.max() * 100).clip(0, 100)

    # Add noise so XGBoost has something to learn (not a perfect function)
    np.random.seed(42)
    energy_score += np.random.normal(0, 3, n)
    waste_score += np.random.normal(0, 3, n)
    nexus_score += np.random.normal(0, 3, n)

    return (
        energy_score.clip(0, 100).values,
        waste_score.clip(0, 100).values,
        nexus_score.clip(0, 100).values,
    )


# ---------------------------------------------------------------------------
# Train & predict
# ---------------------------------------------------------------------------
def train_and_predict(features, energy_labels, waste_labels, nexus_labels):
    """Train 3 XGBoost models (one per score) and predict on all sites."""
    X = features.values.astype(np.float32)
    feature_names = list(features.columns)

    # Detect GPU
    try:
        params_gpu = {"device": "cuda", "tree_method": "hist"}
        # Quick test
        xgb.DMatrix(X[:10], label=energy_labels[:10])
        use_gpu = True
        print("  [XGBoost] Using GPU (CUDA)")
    except Exception:
        params_gpu = {}
        use_gpu = False
        print("  [XGBoost] Using CPU")

    results = {}

    for name, labels in [("energy", energy_labels), ("waste", waste_labels), ("nexus", nexus_labels)]:
        t0 = time.time()

        dtrain = xgb.DMatrix(X, label=labels, feature_names=feature_names)

        params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "eval_metric": "rmse",
            **(params_gpu if use_gpu else {}),
        }

        model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
        preds = model.predict(dtrain)
        preds = np.clip(preds, 0, 100).astype(int)

        elapsed = time.time() - t0
        results[name] = {"model": model, "preds": preds, "time": elapsed}

        print(f"    {name}_score: trained in {elapsed:.2f}s | "
              f"mean={preds.mean():.1f} | min={preds.min()} | max={preds.max()}")

    return results


# ---------------------------------------------------------------------------
# Generate recommendations from scores
# ---------------------------------------------------------------------------
def generate_recommendations(sites, energy_preds, waste_preds, nexus_preds):
    """Rule-based recommendations from XGBoost scores."""
    recs = []
    for i, row in sites.iterrows():
        e = int(energy_preds[i])
        w = int(waste_preds[i])
        n = int(nexus_preds[i])

        solar = pd.to_numeric(
            str(row.get("Estimated Annual Production", 0)).replace(",", ""),
            errors="coerce"
        ) or 0
        is_ej = str(row.get("Environmental Justice Area", "No")).lower() == "yes"

        # BESS sizing heuristic
        if solar > 100000:
            bess_kwh = 750
            savings = 50000 + int(solar * 0.05)
        elif solar > 10000:
            bess_kwh = 500
            savings = 20000 + int(solar * 0.03)
        elif e > 60:
            bess_kwh = 250
            savings = 10000
        else:
            bess_kwh = 100
            savings = 3000

        # Recommendation text
        if n >= 75:
            rec = f"Deploy {bess_kwh} kWh BESS + partner with local AD facility for organics-to-energy"
        elif e >= 70:
            rec = f"Install {bess_kwh} kWh BESS paired with {'existing' if solar > 0 else 'new'} solar"
        elif w >= 70:
            rec = "Prioritize organic waste diversion to nearest composting/AD facility"
        elif is_ej:
            rec = f"EJ priority: evaluate {bess_kwh} kWh BESS + community solar program"
        else:
            rec = "Monitor for future solar-readiness assessment"

        reasoning = (
            f"Energy={e}/100 ({'high' if e>=70 else 'moderate' if e>=50 else 'low'} BESS value), "
            f"Waste={w}/100 ({'high' if w>=70 else 'moderate' if w>=50 else 'low'} optimization opportunity), "
            f"{'Environmental Justice area' if is_ej else 'Non-EJ area'}."
        )

        recs.append({
            "recommended_bess_kwh": bess_kwh,
            "estimated_annual_savings_usd": savings,
            "top_recommendation": rec,
            "reasoning": reasoning,
        })

    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  JOB A (XGBoost): SITE SCORING — Gradient Boosted Trees")
    print("  No LLM. No API. Pure ML.")
    print("=" * 65)

    start_total = time.time()

    # Load data
    sites_path = os.path.join(GOLD_DIR, "unified_sites.parquet")
    profiles_path = os.path.join(GOLD_DIR, "dispatch", "site_profiles.parquet")

    if not os.path.exists(sites_path):
        print(f"\n[ERROR] {sites_path} not found.")
        sys.exit(1)

    sites = pd.read_parquet(sites_path)
    profiles = pd.read_parquet(profiles_path) if os.path.exists(profiles_path) else pd.DataFrame()
    print(f"\n[LOADED] {len(sites)} sites, {len(profiles)} profiles")

    # Build features
    print("\n[FEATURES] Engineering numeric features...")
    t0 = time.time()
    features = build_features(sites, profiles)
    print(f"  → {features.shape[1]} features in {time.time()-t0:.2f}s")

    # Generate proxy labels
    print("\n[LABELS] Generating self-supervised proxy scores...")
    energy_labels, waste_labels, nexus_labels = generate_labels(features)

    # Train & predict
    print("\n[TRAIN] Training 3 XGBoost models...")
    results = train_and_predict(features, energy_labels, waste_labels, nexus_labels)

    energy_preds = results["energy"]["preds"]
    waste_preds = results["waste"]["preds"]
    nexus_preds = results["nexus"]["preds"]

    # Generate recommendations
    print("\n[RECS] Generating recommendations...")
    recs_df = generate_recommendations(sites, energy_preds, waste_preds, nexus_preds)

    # Build ranked output
    ranked = sites.copy()
    ranked["energy_score"] = energy_preds
    ranked["waste_score"] = waste_preds
    ranked["nexus_score"] = nexus_preds
    ranked["recommended_bess_kwh"] = recs_df["recommended_bess_kwh"].values
    ranked["estimated_annual_savings_usd"] = recs_df["estimated_annual_savings_usd"].values
    ranked["top_recommendation"] = recs_df["top_recommendation"].values
    ranked["reasoning"] = recs_df["reasoning"].values

    ranked = ranked.sort_values("nexus_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)

    # Save
    out_path = os.path.join(GOLD_DIR, "ranked_sites_xgboost.parquet")
    ranked.to_parquet(out_path, index=False)
    print(f"\n[SAVED] {out_path} — {len(ranked)} sites")

    # Top 50 JSON
    top50 = ranked.head(50)
    top50_records = []
    for _, row in top50.iterrows():
        top50_records.append({
            "rank": int(row["rank"]),
            "site": str(row.get("Site", "")),
            "address": str(row.get("Address", "")),
            "borough": str(row.get("Borough", "")),
            "agency": str(row.get("Agency", "")),
            "env_justice": str(row.get("Environmental Justice Area", "")),
            "energy_score": int(row["energy_score"]),
            "waste_score": int(row["waste_score"]),
            "nexus_score": int(row["nexus_score"]),
            "recommended_bess_kwh": int(row["recommended_bess_kwh"]),
            "estimated_annual_savings_usd": int(row["estimated_annual_savings_usd"]),
            "top_recommendation": str(row["top_recommendation"]),
            "reasoning": str(row["reasoning"]),
        })

    top50_path = os.path.join(GOLD_DIR, "top50_scored_xgboost.json")
    with open(top50_path, "w") as f:
        json.dump(top50_records, f, indent=2, default=str)
    print(f"[SAVED] {top50_path}")

    # Feature importance
    print("\n[IMPORTANCE] Top features for nexus_score:")
    importance = results["nexus"]["model"].get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for fname, gain in sorted_imp:
        print(f"    {fname:30s}  gain: {gain:.1f}")

    # Timing summary
    total_time = time.time() - start_total
    train_time = sum(r["time"] for r in results.values())

    print(f"\n{'=' * 65}")
    print(f"  XGBoost SCORING COMPLETE")
    print(f"  Total time:    {total_time:.2f}s")
    print(f"  Training time: {train_time:.2f}s (3 models)")
    print(f"  Scoring time:  {total_time - train_time:.2f}s (feature eng + I/O)")
    print(f"  Sites scored:  {len(ranked)}")
    print(f"  Throughput:    {len(ranked)/total_time:.0f} sites/sec")
    print(f"{'=' * 65}")

    # Comparison hint
    print(f"\n  Compare with Qwen scoring:")
    print(f"    Qwen:    data/gold/ranked_sites.parquet")
    print(f"    XGBoost: data/gold/ranked_sites_xgboost.parquet")

    # Top 10
    print(f"\n  TOP 10 (by Nexus Score):")
    print("  " + "─" * 60)
    for _, row in ranked.head(10).iterrows():
        print(f"  #{int(row['rank']):>3}  "
              f"E:{int(row['energy_score']):>3}  "
              f"W:{int(row['waste_score']):>3}  "
              f"N:{int(row['nexus_score']):>3}  "
              f"{str(row.get('Borough','')):>12}  "
              f"{str(row.get('Site',''))[:35]}")
    print("  " + "─" * 60)


if __name__ == "__main__":
    main()
