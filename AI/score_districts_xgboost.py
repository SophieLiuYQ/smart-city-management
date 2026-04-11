"""
District Scoring via XGBoost — no LLM needed.

Reads:  data/gold/district_analysis.json (from analyze_districts.py)
Writes: data/gold/district_scores_xgboost.json

Same input/output format as score_districts_qwen.py for direct comparison.

Usage:
    python AI/analyze_districts.py                    # Step 1: build profiles
    python AI/score_districts_xgboost.py              # Step 2: score with XGBoost
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    print(f"[XGBoost] v{xgb.__version__}")
except ImportError:
    print("[ERROR] pip install xgboost")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_DIR = os.path.join(BASE_DIR, "data", "gold")


# ---------------------------------------------------------------------------
# Extract numeric features from district_analysis.json
# ---------------------------------------------------------------------------
def extract_features(districts):
    """Flatten nested JSON into a numeric feature matrix."""
    rows = []
    for d in districts:
        bs = d.get("buildings_summary", {})
        w = d.get("waste", {})
        w2e = d.get("waste_to_energy", {})
        ad = w2e.get("if_diverted_to_AD", {})
        comp = d.get("complaints", {})
        infra = d.get("infrastructure", {})

        rows.append({
            "district_code": d["district_code"],
            "borough": d["borough"],

            # Building stock
            "num_buildings": bs.get("total", 0),
            "num_ej": bs.get("environmental_justice", 0),
            "pct_ej": bs.get("pct_ej", 0),
            "solar_ready": bs.get("solar_ready", 0),
            "total_solar_kwh": bs.get("total_solar_potential_kwh_yr", 0),
            "avg_solar_kwh": bs.get("avg_solar_per_building_kwh_yr", 0),
            "pct_roof_good": bs.get("pct_roof_good", 0),
            "total_sqft": bs.get("total_sqft", 0),
            "with_energy_data": bs.get("with_energy_data", 0),
            "total_bess_kwh": bs.get("total_bess_capacity_kwh", 0),
            "total_bess_savings": bs.get("total_bess_savings_usd_yr", 0),

            # Waste
            "refuse_tons_mo": w.get("refuse_tons_per_month", 0),
            "recycling_tons_mo": w.get("paper_recycling_tons_per_month", 0) + w.get("mgp_recycling_tons_per_month", 0),
            "organics_collected_mo": w.get("organics_collected_tons_per_month", 0),
            "total_tons_mo": w.get("total_tons_per_month", 0),
            "diversion_rate": w.get("diversion_rate_pct", 0) / 100,
            "diversion_gap": w.get("diversion_gap_pct", 0) / 100,

            # Waste-to-energy
            "organics_in_refuse": w2e.get("organics_in_refuse_tons_per_month", 0),
            "energy_mwh_mo": ad.get("energy_mwh_per_month", 0),
            "energy_mwh_yr": ad.get("energy_mwh_per_year", 0),
            "homes_powered": ad.get("apartments_powered_per_month", 0),
            "co2_avoided_mo": ad.get("co2_avoided_tons_per_month", 0),
            "nearest_ad_km": w2e.get("nearest_ad_distance_km", 999),

            # Complaints
            "total_complaints": comp.get("total", 0),
            "missed": comp.get("missed_collection", 0),
            "overflow": comp.get("overflow", 0),
            "dumping": comp.get("illegal_dumping", 0),
            "dirty": comp.get("dirty_conditions", 0),

            # Infrastructure
            "ev_ports_2km": infra.get("ev_ports_within_2km", 0),
            "compost_sites_2km": infra.get("composting_sites_within_2km", 0),
            "nearest_transfer_km": infra.get("nearest_transfer_distance_km", 999),
        })

    df = pd.DataFrame(rows)

    # Borough one-hot
    for b in ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]:
        df[f"boro_{b.lower().replace(' ', '_')}"] = (df["borough"] == b).astype(int)

    feature_cols = [c for c in df.columns if c not in ["district_code", "borough"]]
    return df, feature_cols


# ---------------------------------------------------------------------------
# Self-supervised labels
# ---------------------------------------------------------------------------
def generate_labels(df, feature_cols):
    """Create proxy scores from domain knowledge."""
    n = len(df)
    np.random.seed(42)

    from scipy.stats import rankdata
    def rp(col):
        return rankdata(df[col].values, method="average") / n

    # Energy score
    energy_raw = (
        rp("total_solar_kwh") * 25 +
        rp("solar_ready") * 15 +
        df["pct_ej"].values / 100 * 20 +
        rp("total_bess_savings") * 15 +
        rp("ev_ports_2km") * 10 +
        rp("total_sqft") * 10 +
        df["pct_roof_good"].values / 100 * 5
    )
    energy = energy_raw / energy_raw.max() * 100 + np.random.normal(0, 2, n)

    # Waste score
    waste_raw = (
        rp("refuse_tons_mo") * 20 +
        rp("organics_in_refuse") * 20 +
        df["diversion_gap"].values * 50 * 100 +  # bigger gap = more opportunity
        rp("total_complaints") * 15 +
        rp("dumping") * 10 +
        (1 - np.clip(rp("compost_sites_2km"), 0, 1)) * 10
    )
    waste = waste_raw / waste_raw.max() * 100 + np.random.normal(0, 2, n)

    # Nexus score
    nexus_raw = (
        energy_raw * 0.35 +
        waste_raw * 0.35 +
        (df["pct_ej"].values / 100 * rp("energy_mwh_yr")) * 30 +
        (1 / (df["nearest_ad_km"].values + 1)) * 20 +
        (rp("ev_ports_2km") * rp("total_solar_kwh")) * 15
    )
    nexus = nexus_raw / nexus_raw.max() * 100 + np.random.normal(0, 2, n)

    return np.clip(energy, 0, 100), np.clip(waste, 0, 100), np.clip(nexus, 0, 100)


# ---------------------------------------------------------------------------
# Train + predict
# ---------------------------------------------------------------------------
def train_and_predict(df, feature_cols, energy_y, waste_y, nexus_y):
    X = df[feature_cols].fillna(0).values.astype(np.float32)

    try:
        gpu_params = {"device": "cuda", "tree_method": "hist"}
        xgb.DMatrix(X[:5], label=energy_y[:5])
        mode = "GPU"
    except Exception:
        gpu_params = {}
        mode = "CPU"
    print(f"  [XGBoost] {mode} mode")

    results = {}
    for name, y in [("energy", energy_y), ("waste", waste_y), ("nexus", nexus_y)]:
        t0 = time.time()
        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)
        params = {
            "objective": "reg:squarederror",
            "max_depth": 4, "eta": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 3, "eval_metric": "rmse",
            **gpu_params,
        }
        model = xgb.train(params, dtrain, num_boost_round=150, verbose_eval=False)
        preds = np.clip(model.predict(dtrain), 0, 100).astype(int)
        elapsed = time.time() - t0
        results[name] = {"model": model, "preds": preds, "time": elapsed}
        print(f"    {name}_score: {elapsed:.2f}s  mean={preds.mean():.1f}  [{preds.min()}, {preds.max()}]")

    return results


# ---------------------------------------------------------------------------
# Generate text analysis from scores + data (rule-based, no LLM)
# ---------------------------------------------------------------------------
def generate_analysis(districts, df, results):
    """Create structured analysis per district — same format as Qwen output."""
    output = []

    for i, d in enumerate(districts):
        code = d["district_code"]
        bs = d.get("buildings_summary", {})
        w = d.get("waste", {})
        w2e = d.get("waste_to_energy", {})
        ad = w2e.get("if_diverted_to_AD", {})
        infra = d.get("infrastructure", {})

        e_score = int(results["energy"]["preds"][i])
        w_score = int(results["waste"]["preds"][i])
        n_score = int(results["nexus"]["preds"][i])

        # Energy analysis
        solar = bs.get("total_solar_potential_kwh_yr", 0)
        bess_total = bs.get("total_bess_capacity_kwh", 0)
        bess_savings = bs.get("total_bess_savings_usd_yr", 0)
        energy_text = (
            f"District has {bs.get('num_buildings',0)} municipal buildings with "
            f"{bs.get('solar_ready',0)} solar-ready ({solar:,.0f} kWh/yr potential). "
            f"Deploying BESS across the district ({bess_total:,} kWh total capacity) "
            f"could save ${bess_savings:,}/yr in peak shaving."
        )

        # Waste analysis
        refuse = w.get("refuse_tons_per_month", 0)
        div_rate = w.get("diversion_rate_pct", 0)
        div_gap = w.get("diversion_gap_pct", 0)
        waste_text = (
            f"Generates {refuse:,.0f} tons refuse/month with only {div_rate:.1f}% diversion rate "
            f"({div_gap:.1f}% below 30% target). "
            f"{d.get('complaints', {}).get('total', 0)} sanitation complaints, "
            f"including {d.get('complaints', {}).get('illegal_dumping', 0)} illegal dumping reports."
        )

        # Waste-to-energy
        org = w2e.get("organics_in_refuse_tons_per_month", 0)
        mwh = ad.get("energy_mwh_per_month", 0)
        homes = ad.get("apartments_powered_per_month", 0)
        co2 = ad.get("co2_avoided_tons_per_month", 0)
        ad_name = w2e.get("nearest_ad_facility", "Unknown")
        ad_km = w2e.get("nearest_ad_distance_km", 999)
        w2e_text = (
            f"{org:,.0f} tons/month of organic waste in the refuse stream could be diverted "
            f"to {ad_name} ({ad_km:.1f} km away). "
            f"This would generate {mwh:,.0f} MWh/month of biogas energy, "
            f"powering {homes:,.0f} apartments and avoiding {co2:,.0f} tons CO₂/month."
        )

        # BESS recommendation
        top_bldgs = d.get("buildings", [])[:3]
        bess_lines = []
        for b in top_bldgs:
            if b.get("bess_recommendation", {}).get("capacity_kwh", 0) > 0:
                bess_lines.append(
                    f"{b['site']}: {b['bess_recommendation']['capacity_kwh']} kWh "
                    f"(${b['bess_recommendation']['est_annual_savings_usd']:,}/yr)"
                )
        bess_text = "Priority BESS deployments: " + "; ".join(bess_lines) if bess_lines else "No high-priority BESS sites identified."

        # Equity
        pct_ej = bs.get("pct_ej", 0)
        if pct_ej > 30:
            equity_text = f"Environmental Justice district ({pct_ej:.0f}% EJ buildings) — should receive priority investment in clean energy and waste infrastructure."
        elif pct_ej > 0:
            equity_text = f"Partially EJ area ({pct_ej:.0f}% EJ buildings) — equity weighting applies to a subset of sites."
        else:
            equity_text = "Non-EJ district — standard investment prioritization applies."

        # Top 3 actions
        actions = []
        if n_score >= 60 and org > 1000:
            actions.append(f"Divert {org:,.0f}t/mo organics to {ad_name} → {mwh:,.0f} MWh/mo biogas")
        if e_score >= 50 and bess_lines:
            actions.append(f"Deploy BESS at top sites: {bess_lines[0]}")
        if w_score >= 60 and div_gap > 10:
            actions.append(f"Close {div_gap:.0f}% diversion gap — add composting sites (currently {infra.get('composting_sites_within_2km', 0)} within 2km)")
        if infra.get("ev_ports_within_2km", 0) > 20:
            actions.append(f"Buffer EV demand ({infra['ev_ports_within_2km']} ports) with depot-level BESS")
        if not actions:
            actions = ["Monitor for future assessment", "Evaluate solar-readiness for new installations"]

        output.append({
            "district_code": code,
            "energy_score": e_score,
            "waste_score": w_score,
            "nexus_score": n_score,
            "energy_analysis": energy_text,
            "waste_analysis": waste_text,
            "waste_to_energy_analysis": w2e_text,
            "bess_recommendation": bess_text,
            "equity_note": equity_text,
            "top_3_actions": actions[:3],
        })

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  DISTRICT SCORING — XGBoost (No LLM)")
    print("=" * 65)

    t0 = time.time()

    # Load
    input_path = os.path.join(GOLD_DIR, "district_analysis.json")
    if not os.path.exists(input_path):
        print(f"\n[ERROR] {input_path} not found. Run analyze_districts.py first.")
        sys.exit(1)

    with open(input_path) as f:
        districts = json.load(f)
    print(f"\n[LOADED] {len(districts)} districts")

    # Features
    df, feature_cols = extract_features(districts)
    print(f"[FEATURES] {len(feature_cols)} numeric features extracted")

    # Labels
    energy_y, waste_y, nexus_y = generate_labels(df, feature_cols)

    # Train
    print("\n[TRAIN]")
    results = train_and_predict(df, feature_cols, energy_y, waste_y, nexus_y)

    # Feature importance
    print("\n[IMPORTANCE] Top 10 for nexus_score:")
    imp = results["nexus"]["model"].get_score(importance_type="gain")
    for fname, gain in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {fname:25s}  gain: {gain:.1f}")

    # Analysis
    print("\n[ANALYSIS] Generating per-district analysis...")
    output = generate_analysis(districts, df, results)

    # Save
    out_path = os.path.join(GOLD_DIR, "district_scores_xgboost.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n[SAVED] {out_path}")

    print(f"\n{'=' * 65}")
    print(f"  COMPLETE — {elapsed:.1f}s total")
    print(f"  {len(output)} districts scored")
    print(f"{'=' * 65}")

    # Print all
    print(f"\n  {'Code':>6s}  {'E':>3s}  {'W':>3s}  {'N':>3s}  Top Action")
    print("  " + "─" * 70)
    for r in sorted(output, key=lambda x: x["nexus_score"], reverse=True):
        top = r["top_3_actions"][0][:50] if r["top_3_actions"] else ""
        print(f"  {r['district_code']:>6s}  {r['energy_score']:>3d}  {r['waste_score']:>3d}  "
              f"{r['nexus_score']:>3d}  {top}")


if __name__ == "__main__":
    main()
