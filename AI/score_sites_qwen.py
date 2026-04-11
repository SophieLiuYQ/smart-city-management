"""
Job A: Score all 4,268 municipal sites using Qwen 3 80B running locally on GB10.

Reads:  data/gold/unified_sites.parquet
        data/gold/dispatch/site_profiles.parquet
Writes: data/gold/ranked_sites.parquet

Runs 100% locally on NVIDIA DGX Spark / Acer Veriton GN100.
No cloud calls. Qwen 3 80B via local OpenAI-compatible endpoint.

Usage (on GB10):
    python AI/score_sites_qwen.py

    # Optional: adjust batch size and endpoint
    QWEN_URL=http://localhost:8000/v1 BATCH_SIZE=10 python AI/score_sites_qwen.py
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QWEN_URL = os.getenv("QWEN_URL", "http://localhost:8000/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3-80b")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_DIR = os.path.join(BASE_DIR, "data", "gold")

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert urban sustainability analyst for New York City. You specialize in Battery Energy Storage Systems (BESS), waste-to-energy optimization, and environmental equity.

TASK: Score each municipal site on three dimensions (0-100) and provide one actionable recommendation.

SCORING DIMENSIONS:

1. energy_score (0-100): BESS deployment priority
   - Large buildings (high square footage) with solar-ready roofs score higher
   - Sites with solar installations or high estimated production score higher
   - Environmental Justice areas get a bonus (+10)
   - Good roof condition = ready for solar+BESS pairing

2. waste_score (0-100): Waste optimization opportunity
   - Sites near composting drop-offs (compost_within_1km = true) but in high-waste districts score higher
   - Sites far from transfer stations (high nearest_transfer_dist_m) = inefficient collection = higher opportunity
   - Environmental Justice areas often bear disproportionate waste burden = bonus

3. nexus_score (0-100): Energy-waste cross-domain synergy
   - Sites where BESS + waste diversion would have combined impact
   - EV infrastructure nearby (ev_within_1km = true) + solar + BESS = high synergy
   - Environmental Justice + high energy + high waste = maximum nexus score
   - Sites near composting but far from transfer stations = organic diversion + energy savings

RETURN FORMAT: A valid JSON array. Each element must have exactly these fields:
{
  "site": "<site name>",
  "energy_score": <int 0-100>,
  "waste_score": <int 0-100>,
  "nexus_score": <int 0-100>,
  "recommended_bess_kwh": <int>,
  "estimated_annual_savings_usd": <int>,
  "top_recommendation": "<one specific actionable sentence>",
  "reasoning": "<2-3 sentences explaining the scores>"
}

RULES:
- Return ONLY the JSON array. No markdown fencing. No explanation outside JSON.
- Every site in the input MUST appear in the output.
- Be specific in recommendations — name the action, the size, and the expected impact."""


# ---------------------------------------------------------------------------
# Build site features for LLM
# ---------------------------------------------------------------------------
def build_site_features(sites_df, profiles_df):
    """Merge unified_sites with site_profiles for richer context."""
    # Build a clean feature dict per site
    features = []

    for idx, row in sites_df.iterrows():
        site_name = str(row.get("Site", "Unknown"))

        # Match profile by site name
        profile = profiles_df[profiles_df["site_id"] == site_name]
        avg_kwh = float(profile["avg_monthly_total_kwh"].iloc[0]) if len(profile) > 0 else 0
        peak_kw = float(profile["peak_kw"].iloc[0]) if len(profile) > 0 else 0
        solar_prod = profile["solar_production_kwh_yr"].iloc[0] if len(profile) > 0 else row.get("Estimated Annual Production", 0)

        f = {
            "site": site_name,
            "address": str(row.get("Address", "")),
            "borough": str(row.get("Borough", "")),
            "agency": str(row.get("Agency", "")),
            "env_justice_area": str(row.get("Environmental Justice Area", "No")),
            "roof_condition": str(row.get("Roof Condition", "Unknown")),
            "roof_age": str(row.get("Roof Age", "Unknown")),
            "sqft": str(row.get("Total Gross Square Footage", "0")),
            "solar_status": str(row.get("Status", "Unknown")),
            "solar_production_kwh_yr": _safe_num(solar_prod),
            "estimated_annual_savings": str(row.get("Estimated Annual Energy Savings", "$0")),
            "avg_monthly_kwh": round(avg_kwh, 1),
            "peak_kw": round(peak_kw, 1),
            "ev_within_1km": bool(row.get("ev_within_1km", False)),
            "nearest_ev_dist_m": round(float(row.get("nearest_ev_dist_m", 9999) or 9999), 0),
            "compost_within_1km": bool(row.get("compost_within_1km", False)),
            "nearest_transfer_dist_m": round(float(row.get("nearest_transfer_dist_m", 9999) or 9999), 0),
        }
        features.append(f)

    return features


def _safe_num(val):
    """Convert to float, return 0 if not possible."""
    try:
        v = float(str(val).replace(",", "").replace("$", ""))
        return 0 if np.isnan(v) or np.isinf(v) else round(v, 1)
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Score a batch
# ---------------------------------------------------------------------------
def score_batch(client, batch_features):
    """Send a batch of site features to Qwen and parse the response."""
    user_msg = (
        f"Score these {len(batch_features)} NYC municipal sites.\n\n"
        f"Site data:\n{json.dumps(batch_features, default=str)}"
    )

    try:
        response = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=8192,
        )

        text = response.choices[0].message.content.strip()

        # Handle /think tags from Qwen 3
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()

        # Strip markdown fencing
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        results = json.loads(text)

        # Validate: must be a list
        if not isinstance(results, list):
            raise ValueError(f"Expected list, got {type(results)}")

        return results

    except json.JSONDecodeError as e:
        print(f"\n    [WARN] JSON parse failed: {e}")
        print(f"    Response preview: {text[:300]}...")
        return _fallback_scores(batch_features)
    except Exception as e:
        print(f"\n    [ERROR] Qwen call failed: {e}")
        return _fallback_scores(batch_features)


def _fallback_scores(batch_features):
    """Heuristic scoring when LLM fails."""
    results = []
    for f in batch_features:
        is_ej = f.get("env_justice_area", "No").lower() == "yes"
        has_solar = f.get("solar_production_kwh_yr", 0) > 0
        good_roof = f.get("roof_condition", "").lower() == "good"
        has_ev = f.get("ev_within_1km", False)
        has_compost = f.get("compost_within_1km", False)

        e = 40 + (15 if has_solar else 0) + (10 if good_roof else 0) + (10 if is_ej else 0) + (5 if has_ev else 0)
        w = 40 + (10 if not has_compost else 0) + (15 if is_ej else 0) + (5 if f.get("nearest_transfer_dist_m", 0) > 5000 else 0)
        n = int((e + w) / 2) + (10 if is_ej and has_solar else 0)

        results.append({
            "site": f["site"],
            "energy_score": min(100, e),
            "waste_score": min(100, w),
            "nexus_score": min(100, n),
            "recommended_bess_kwh": 250 if has_solar else 100,
            "estimated_annual_savings_usd": 10000 if has_solar else 3000,
            "top_recommendation": "Install BESS paired with existing solar" if has_solar else "Evaluate solar-readiness first",
            "reasoning": "Heuristic fallback — Qwen scoring unavailable for this batch.",
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  JOB A: SITE SCORING — Qwen 3 80B (Local on GB10)")
    print(f"  Endpoint: {QWEN_URL}")
    print(f"  Model: {QWEN_MODEL}")
    print(f"  Batch size: {BATCH_SIZE}")
    print("=" * 65)

    # Load data
    sites_path = os.path.join(GOLD_DIR, "unified_sites.parquet")
    profiles_path = os.path.join(GOLD_DIR, "dispatch", "site_profiles.parquet")

    if not os.path.exists(sites_path):
        print(f"\n[ERROR] {sites_path} not found. Run the gold layer pipeline first.")
        sys.exit(1)

    sites = pd.read_parquet(sites_path)
    profiles = pd.read_parquet(profiles_path) if os.path.exists(profiles_path) else pd.DataFrame()

    print(f"\n[LOADED] {len(sites)} sites from unified_sites.parquet")
    print(f"[LOADED] {len(profiles)} profiles from site_profiles.parquet")

    # Build features
    print("\n[PREP] Building site features for LLM...")
    all_features = build_site_features(sites, profiles)
    print(f"  → {len(all_features)} site feature dicts ready")

    # Connect to local Qwen
    client = OpenAI(base_url=QWEN_URL, api_key="not-needed")

    # Test connection
    print(f"\n[TEST] Pinging {QWEN_URL}...")
    try:
        models = client.models.list()
        print(f"  → Connected. Available models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"  → Connection failed: {e}")
        print(f"  → Running in FALLBACK mode (heuristic scores)")
        # Score everything with fallback
        all_results = _fallback_scores(all_features)
        _save_results(sites, all_results)
        return

    # Score in batches
    all_results = []
    total = len(all_features)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    start_time = time.time()
    success_count = 0
    fallback_count = 0

    for i in range(0, total, BATCH_SIZE):
        batch = all_features[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        elapsed = time.time() - start_time
        rate = (i / elapsed) if elapsed > 0 else 0
        eta = ((total - i) / rate / 60) if rate > 0 else 0

        print(f"\n[BATCH {batch_num}/{total_batches}] "
              f"Sites {i+1}–{min(i+BATCH_SIZE, total)} "
              f"({elapsed:.0f}s elapsed, ~{eta:.1f}min remaining)...", end="", flush=True)

        t0 = time.time()
        results = score_batch(client, batch)
        batch_time = time.time() - t0

        # Check if results came from LLM or fallback
        is_fallback = any("Heuristic fallback" in r.get("reasoning", "") for r in results)
        if is_fallback:
            fallback_count += len(results)
        else:
            success_count += len(results)

        all_results.extend(results)
        print(f" {batch_time:.1f}s ({'LLM' if not is_fallback else 'FALLBACK'})")

    total_time = time.time() - start_time
    _save_results(sites, all_results)

    print(f"\n{'=' * 65}")
    print(f"  SCORING COMPLETE")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  LLM scored: {success_count} sites")
    print(f"  Fallback:   {fallback_count} sites")
    print(f"  Rate:       {total / total_time:.1f} sites/sec")
    print(f"{'=' * 65}")


def _save_results(sites, all_results):
    """Merge scores back onto sites and save."""
    scores_df = pd.DataFrame(all_results)

    # Merge on site name
    site_col = "Site" if "Site" in sites.columns else "site"
    sites_copy = sites.copy()
    sites_copy["_merge_key"] = sites_copy[site_col].astype(str)
    scores_df["_merge_key"] = scores_df["site"].astype(str)

    # Deduplicate scores (keep first per site name)
    scores_dedup = scores_df.drop_duplicates(subset=["_merge_key"], keep="first")

    ranked = sites_copy.merge(
        scores_dedup[["_merge_key", "energy_score", "waste_score", "nexus_score",
                    "recommended_bess_kwh", "estimated_annual_savings_usd",
                    "top_recommendation", "reasoning"]],
        on="_merge_key", how="left"
    ).drop(columns=["_merge_key"])

    # Fill missing scores
    for col in ["energy_score", "waste_score", "nexus_score"]:
        if col in ranked.columns:
            ranked[col] = ranked[col].fillna(50).astype(int)
    ranked["recommended_bess_kwh"] = ranked["recommended_bess_kwh"].fillna(100).astype(int)
    ranked["estimated_annual_savings_usd"] = ranked["estimated_annual_savings_usd"].fillna(0).astype(int)
    ranked["top_recommendation"] = ranked["top_recommendation"].fillna("Review needed")
    ranked["reasoning"] = ranked["reasoning"].fillna("")

    # Sort by nexus score
    ranked = ranked.sort_values("nexus_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)

    # Save
    out_path = os.path.join(GOLD_DIR, "ranked_sites.parquet")
    ranked.to_parquet(out_path, index=False)
    print(f"\n[SAVED] {out_path} — {len(ranked)} sites ranked")

    # Also save top 50 as readable JSON for dashboard
    top50 = ranked.head(50)
    top50_records = []
    for _, row in top50.iterrows():
        top50_records.append({
            "rank": int(row["rank"]),
            "site": str(row.get("Site", row.get("site", ""))),
            "address": str(row.get("Address", "")),
            "borough": str(row.get("Borough", "")),
            "agency": str(row.get("Agency", "")),
            "env_justice": str(row.get("Environmental Justice Area", "")),
            "energy_score": int(row.get("energy_score", 50)),
            "waste_score": int(row.get("waste_score", 50)),
            "nexus_score": int(row.get("nexus_score", 50)),
            "recommended_bess_kwh": int(row.get("recommended_bess_kwh", 0)),
            "estimated_annual_savings_usd": int(row.get("estimated_annual_savings_usd", 0)),
            "top_recommendation": str(row.get("top_recommendation", "")),
            "reasoning": str(row.get("reasoning", "")),
        })

    top50_path = os.path.join(GOLD_DIR, "top50_scored.json")
    with open(top50_path, "w") as f:
        json.dump(top50_records, f, indent=2, default=str)
    print(f"[SAVED] {top50_path} — top 50 sites as JSON")

    # Print top 10
    print(f"\n  TOP 10 SITES (by Nexus Score):")
    print("  " + "─" * 60)
    for _, row in ranked.head(10).iterrows():
        print(f"  #{int(row['rank']):>3}  "
              f"E:{int(row.get('energy_score',0)):>3}  "
              f"W:{int(row.get('waste_score',0)):>3}  "
              f"N:{int(row.get('nexus_score',0)):>3}  "
              f"{str(row.get('Borough','')):>12}  "
              f"{str(row.get('Site', row.get('site','')))[:35]}")
    print("  " + "─" * 60)


if __name__ == "__main__":
    main()
