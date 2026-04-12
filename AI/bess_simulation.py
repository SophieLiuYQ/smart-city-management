"""
BESS Dispatch Simulation — Takes XGBoost/CatBoost scored sites and simulates
real 24-hour battery charge/discharge cycles with NYC energy parameters.

Pipeline: scored sites → BESS sizing → 24h simulation → savings + CO₂ + peak reduction

Input:  data/gold/ranked_sites_xgboost_no_ej_split.parquet (or any ranked_sites file)
Output: data/gold/bess_simulation_results.json

Usage:
    python AI/bess_simulation.py
    python AI/bess_simulation.py --input ranked_sites_catboost_gpu_val.parquet --top 100

    # On GB10:
    RAW_DIR=/home/acergn100_6/smart-city-management/data/raw python3 AI/bess_simulation.py
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_DIR = os.path.join(BASE_DIR, "data", "gold")
RAW_DIR = os.getenv("RAW_DIR", os.path.join(BASE_DIR, "data", "raw"))


# ═══════════════════════════════════════════════════════════════════════════
# NYC Energy Constants
# ═══════════════════════════════════════════════════════════════════════════

# ConEd time-of-use rates ($/kWh)
TOU_RATES = {
    "peak": 0.22,       # 8AM-10PM weekdays
    "off_peak": 0.08,   # 10PM-8AM + weekends
    "demand_charge": 18.0,  # $/kW peak demand charge per month
}

# NYC grid CO₂ (tons per kWh)
CO2_PER_KWH = 0.000288

# Solar generation curve (normalized 0-1, hour 0-23)
SOLAR_CURVE = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.01,
    0.05, 0.15, 0.35, 0.60, 0.80, 0.95,
    1.0, 0.95, 0.85, 0.70, 0.50, 0.25,
    0.08, 0.0, 0.0, 0.0, 0.0, 0.0,
])

# Building demand curves by type (normalized 0-1, hour 0-23)
DEMAND_CURVES = {
    "office": np.array([
        0.30, 0.28, 0.25, 0.25, 0.25, 0.30,
        0.45, 0.65, 0.85, 0.95, 1.00, 0.98,
        0.95, 0.97, 1.00, 0.95, 0.85, 0.70,
        0.55, 0.45, 0.40, 0.38, 0.35, 0.32,
    ]),
    "school": np.array([
        0.15, 0.12, 0.10, 0.10, 0.10, 0.15,
        0.30, 0.65, 0.90, 1.00, 1.00, 0.95,
        0.90, 0.95, 0.85, 0.60, 0.30, 0.20,
        0.15, 0.12, 0.12, 0.12, 0.12, 0.12,
    ]),
    "hospital": np.array([
        0.70, 0.65, 0.62, 0.60, 0.62, 0.65,
        0.75, 0.85, 0.95, 1.00, 1.00, 0.98,
        0.95, 0.98, 1.00, 0.95, 0.90, 0.85,
        0.80, 0.78, 0.75, 0.73, 0.72, 0.70,
    ]),
    "default": np.array([
        0.30, 0.28, 0.25, 0.25, 0.25, 0.30,
        0.45, 0.65, 0.85, 0.95, 1.00, 0.98,
        0.95, 0.97, 1.00, 0.95, 0.85, 0.70,
        0.55, 0.45, 0.40, 0.38, 0.35, 0.32,
    ]),
}

# EV charging pattern (normalized 0-1)
EV_CURVE = np.array([
    0.05, 0.03, 0.02, 0.02, 0.02, 0.05,
    0.15, 0.60, 0.80, 0.40, 0.20, 0.15,
    0.10, 0.10, 0.10, 0.15, 0.30, 0.70,
    0.90, 1.00, 0.80, 0.50, 0.25, 0.10,
])

# Seasonal factors (multiplier on base demand)
SEASONS = {
    "summer": {"demand_mult": 1.3, "solar_mult": 1.2, "label": "Summer (peak AC)"},
    "winter": {"demand_mult": 1.1, "solar_mult": 0.6, "label": "Winter (heating + short days)"},
    "spring": {"demand_mult": 0.9, "solar_mult": 1.0, "label": "Spring (baseline)"},
}


# ═══════════════════════════════════════════════════════════════════════════
# BESS Simulator
# ═══════════════════════════════════════════════════════════════════════════

class BESSSimulator:
    def __init__(self, capacity_kwh, max_power_kw,
                 efficiency=0.90, min_soc=0.10, max_soc=0.95,
                 degradation_rate=0.02):
        self.capacity = capacity_kwh
        self.max_power = max_power_kw
        self.efficiency = efficiency
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.degradation_rate = degradation_rate  # per year

    def simulate_day(self, site_peak_kw, solar_kw, ev_ports,
                     demand_curve, season_mult=1.0, solar_mult=1.0,
                     initial_soc_pct=0.20):
        """Simulate one 24-hour cycle. Returns schedule + metrics."""
        soc = initial_soc_pct * self.capacity
        schedule = []
        total_savings = 0.0
        total_co2_offset = 0.0
        peak_grid_before = 0.0
        peak_grid_after = 0.0
        cycles = 0.0

        ev_load_kw = min(ev_ports * 7.2, 150)  # 7.2 kW per L2 port, cap 150

        for hour in range(24):
            demand_kw = site_peak_kw * demand_curve[hour] * season_mult
            solar_gen = solar_kw * SOLAR_CURVE[hour] * solar_mult
            ev_kw = ev_load_kw * EV_CURVE[hour]
            net_demand = demand_kw + ev_kw - solar_gen

            is_peak = 8 <= hour < 22
            rate = TOU_RATES["peak"] if is_peak else TOU_RATES["off_peak"]

            action = "idle"
            power_kw = 0.0
            soc_pct = soc / self.capacity

            if not is_peak and soc_pct < self.max_soc:
                # Off-peak: charge
                headroom = (self.max_soc * self.capacity - soc) / self.efficiency
                power_kw = min(self.max_power, headroom)
                soc += power_kw * self.efficiency
                action = "charge"
                # Charging costs money at off-peak rate
                charge_cost = power_kw * TOU_RATES["off_peak"]
            elif is_peak and soc_pct > self.min_soc and net_demand > site_peak_kw * 0.5:
                # Peak + high demand: discharge
                available = min(self.max_power, soc - self.min_soc * self.capacity)
                power_kw = min(available, net_demand * 0.4)
                soc -= power_kw
                action = "discharge"
                # Savings: displaced peak-rate energy
                total_savings += power_kw * (rate - TOU_RATES["off_peak"])
                total_co2_offset += power_kw * CO2_PER_KWH
                cycles += power_kw / self.capacity

            grid_before = max(0, net_demand)
            grid_after = max(0, net_demand - (power_kw if action == "discharge" else 0))
            peak_grid_before = max(peak_grid_before, grid_before)
            peak_grid_after = max(peak_grid_after, grid_after)

            schedule.append({
                "hour": hour,
                "action": action,
                "power_kw": round(power_kw, 1),
                "soc_pct": round(soc / self.capacity * 100, 1),
                "demand_kw": round(demand_kw, 1),
                "solar_kw": round(solar_gen, 1),
                "ev_kw": round(ev_kw, 1),
                "net_demand_kw": round(net_demand, 1),
                "grid_kw": round(grid_after, 1),
                "rate": rate,
            })

        peak_reduction = peak_grid_before - peak_grid_after
        demand_charge_savings = peak_reduction * TOU_RATES["demand_charge"]

        return {
            "schedule": schedule,
            "daily_energy_savings_usd": round(total_savings, 2),
            "daily_demand_savings_usd": round(demand_charge_savings / 30, 2),  # monthly charge / 30
            "daily_total_savings_usd": round(total_savings + demand_charge_savings / 30, 2),
            "peak_before_kw": round(peak_grid_before, 1),
            "peak_after_kw": round(peak_grid_after, 1),
            "peak_reduction_kw": round(peak_reduction, 1),
            "peak_reduction_pct": round(peak_reduction / max(peak_grid_before, 1) * 100, 1),
            "daily_co2_offset_tons": round(total_co2_offset, 4),
            "daily_cycles": round(cycles, 3),
            "utilization_pct": round(sum(1 for s in schedule if s["action"] != "idle") / 24 * 100, 1),
        }

    def simulate_year(self, site_peak_kw, solar_kw, ev_ports, demand_curve):
        """Simulate full year across 3 seasons. Accounts for degradation."""
        season_results = {}
        total_savings = 0.0
        total_co2 = 0.0
        total_peak_reduction = 0.0

        for season_name, params in SEASONS.items():
            result = self.simulate_day(
                site_peak_kw, solar_kw, ev_ports, demand_curve,
                season_mult=params["demand_mult"],
                solar_mult=params["solar_mult"],
            )
            season_results[season_name] = result

            # Weight: summer=122 days, winter=121 days, spring=122 days
            days = {"summer": 122, "winter": 121, "spring": 122}[season_name]
            total_savings += result["daily_total_savings_usd"] * days
            total_co2 += result["daily_co2_offset_tons"] * days
            total_peak_reduction += result["peak_reduction_kw"]

        # Apply degradation (year 1 = full, loses 2%/year capacity)
        # Average over 10-year project life
        degradation_factor = 1 - (self.degradation_rate * 5)  # midpoint of 10 years

        annual = {
            "annual_savings_usd": round(total_savings * degradation_factor),
            "annual_co2_offset_tons": round(total_co2 * degradation_factor, 1),
            "avg_peak_reduction_kw": round(total_peak_reduction / 3, 1),
            "degradation_factor": round(degradation_factor, 3),
            "seasons": season_results,
        }

        return annual


# ═══════════════════════════════════════════════════════════════════════════
# BESS Sizing
# ═══════════════════════════════════════════════════════════════════════════

def size_bess(site):
    """Determine BESS capacity based on site characteristics."""
    solar = float(site.get("solar_kwh", 0) or 0)
    energy_score = int(site.get("energy_score", 50) or 50)
    nexus_score = int(site.get("nexus_score", 50) or 50)

    # Size based on solar potential + score
    if solar > 200000 or nexus_score >= 90:
        capacity, power = 1000, 250
    elif solar > 100000 or nexus_score >= 80:
        capacity, power = 750, 188
    elif solar > 50000 or nexus_score >= 70:
        capacity, power = 500, 125
    elif solar > 10000 or energy_score >= 60:
        capacity, power = 250, 63
    else:
        capacity, power = 100, 25

    # Estimated install cost ($400/kWh)
    install_cost = capacity * 400

    return capacity, power, install_cost


def guess_building_type(site_name, agency):
    """Guess building type from name/agency for demand curve selection."""
    name = str(site_name).lower()
    agency = str(agency).lower()

    if any(w in name for w in ["hospital", "medical", "health"]):
        return "hospital"
    elif any(w in name for w in ["school", "academy", "k-12", "ps ", "is ", "hs "]):
        return "school"
    elif agency == "doe":
        return "school"
    elif any(w in name for w in ["k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9",
                                  "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                                  "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
                                  "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9",
                                  "r0", "r1"]):
        return "school"
    else:
        return "default"


# ═══════════════════════════════════════════════════════════════════════════
# Main Simulation Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_simulation(input_file=None, top_n=50):
    print("=" * 80)
    print("  BESS DISPATCH SIMULATION")
    print("  XGBoost/CatBoost scores → BESS sizing → 24h dispatch → annual savings")
    print("=" * 80)

    # Load scored sites
    if input_file is None:
        # Try multiple files in order of preference
        candidates = [
            "ranked_sites_xgboost_no_ej_split.parquet",
            "ranked_sites_xgboost_no_ej.parquet",
            "ranked_sites_catboost_gpu_val.parquet",
            "ranked_sites.parquet",
        ]
        for c in candidates:
            path = os.path.join(GOLD_DIR, c)
            if os.path.exists(path):
                input_file = c
                break
        if input_file is None:
            print("[ERROR] No scored sites found. Run a scoring script first.")
            sys.exit(1)

    input_path = os.path.join(GOLD_DIR, input_file)
    sites = pd.read_parquet(input_path)
    print(f"\n[LOADED] {len(sites)} sites from {input_file}")

    # Deduplicate — keep highest nexus_score per site name
    site_col = "Site" if "Site" in sites.columns else "site"
    if "nexus_score" in sites.columns:
        sites = sites.sort_values("nexus_score", ascending=False)
    sites = sites.drop_duplicates(subset=[site_col], keep="first")
    print(f"[DEDUP] {len(sites)} unique sites")
    if "nexus_score" in sites.columns:
        sites = sites.sort_values("nexus_score", ascending=False).head(top_n)
    else:
        sites = sites.head(top_n)
    sites = sites.reset_index(drop=True)
    print(f"[SIMULATING] Top {len(sites)} sites\n")

    # Load E5 for solar data if not in scored file
    solar_col = None
    for c in ["solar_kwh", "solar_production_kwh_yr", "Estimated Annual Production"]:
        if c in sites.columns:
            solar_col = c
            break

    if solar_col is None:
        # Try to load from E5
        sys.path.insert(0, os.path.join(BASE_DIR, "AI"))
        try:
            from score_raw_xgboost_no_ej import find_raw_file
            e5 = pd.read_csv(find_raw_file("E5"), low_memory=False)
            e5["solar_kwh"] = pd.to_numeric(
                e5["Estimated Annual Production"].astype(str).str.replace(",", ""), errors="coerce"
            ).fillna(0)
            sites = sites.merge(e5[["Site", "solar_kwh"]], on="Site", how="left")
            solar_col = "solar_kwh"
        except Exception:
            sites["solar_kwh"] = 0
            solar_col = "solar_kwh"

    # Run simulations
    results = []
    total_annual_savings = 0
    total_annual_co2 = 0
    total_install_cost = 0
    start = time.time()

    print(f"  {'#':>3s}  {'Site':35s}  {'Type':>8s}  {'BESS':>6s}  {'Annual $':>9s}  "
          f"{'CO₂ t/yr':>8s}  {'Peak -kW':>8s}  {'Payback':>7s}")
    print(f"  {'─'*90}")

    for i, (_, site) in enumerate(sites.iterrows()):
        # BESS sizing
        capacity, power, install_cost = size_bess(site)

        # Building type
        btype = guess_building_type(
            site.get("Site", ""), site.get("Agency", "")
        )
        demand_curve = DEMAND_CURVES[btype]

        # Solar capacity (annual kWh / 1500 peak hours)
        solar_annual = float(site.get(solar_col, 0) or 0)
        solar_kw = min(solar_annual / 1500, 300)  # cap at 300 kW

        # EV ports
        ev_ports = 0
        for c in ["ev_ports_1km", "ev_within_1km", "ev_1km"]:
            if c in site.index:
                val = site[c]
                if isinstance(val, bool):
                    ev_ports = 10 if val else 0
                else:
                    ev_ports = int(float(val or 0))
                break

        # Estimate peak demand
        peak_kw = 200  # default
        for c in ["est_peak_kw", "peak_kw", "peak_monthly_elec_kwh"]:
            if c in site.index:
                val = float(site[c] or 0)
                if val > 0:
                    if "monthly" in c:
                        peak_kw = val / 730 * 1.5
                    else:
                        peak_kw = val
                    break
        peak_kw = max(50, min(peak_kw, 2000))

        # Run simulation
        sim = BESSSimulator(capacity_kwh=capacity, max_power_kw=power)
        annual = sim.simulate_year(peak_kw, solar_kw, ev_ports, demand_curve)

        # Payback
        payback_years = install_cost / max(annual["annual_savings_usd"], 1)

        result = {
            "rank": i + 1,
            "site": str(site.get("Site", "N/A")),
            "address": str(site.get("Address", "N/A")),
            "borough": str(site.get("Borough", "")),
            "agency": str(site.get("Agency", "")),
            "ej": str(site.get("Environmental Justice Area", "")),
            "energy_score": int(site.get("energy_score", 0) or 0),
            "waste_score": int(site.get("waste_score", 0) or 0),
            "nexus_score": int(site.get("nexus_score", 0) or 0),
            "building_type": btype,
            "bess": {
                "capacity_kwh": capacity,
                "power_kw": power,
                "install_cost_usd": install_cost,
            },
            "site_params": {
                "peak_demand_kw": round(peak_kw, 1),
                "solar_capacity_kw": round(solar_kw, 1),
                "solar_annual_kwh": round(solar_annual, 0),
                "ev_ports_nearby": ev_ports,
            },
            "annual_results": {
                "savings_usd": annual["annual_savings_usd"],
                "co2_offset_tons": annual["annual_co2_offset_tons"],
                "peak_reduction_kw": annual["avg_peak_reduction_kw"],
                "payback_years": round(payback_years, 1),
            },
            "seasonal_detail": {
                season: {
                    "daily_savings_usd": data["daily_total_savings_usd"],
                    "peak_reduction_kw": data["peak_reduction_kw"],
                    "peak_reduction_pct": data["peak_reduction_pct"],
                    "utilization_pct": data["utilization_pct"],
                    "daily_cycles": data["daily_cycles"],
                }
                for season, data in annual["seasons"].items()
            },
            "summer_schedule": annual["seasons"]["summer"]["schedule"],
        }

        results.append(result)
        total_annual_savings += annual["annual_savings_usd"]
        total_annual_co2 += annual["annual_co2_offset_tons"]
        total_install_cost += install_cost

        print(f"  {i+1:>3d}  {str(site.get('Site',''))[:35]:35s}  {btype:>8s}  {capacity:>5d}  "
              f"${annual['annual_savings_usd']:>8,}  {annual['annual_co2_offset_tons']:>7.1f}t  "
              f"-{annual['avg_peak_reduction_kw']:>6.1f}  {payback_years:>6.1f}yr")

    elapsed = time.time() - start

    # Save results
    out_path = os.path.join(GOLD_DIR, "bess_simulation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")

    # Summary
    avg_payback = total_install_cost / max(total_annual_savings, 1)

    print(f"\n{'=' * 80}")
    print(f"  SIMULATION COMPLETE — {elapsed:.1f}s for {len(results)} sites")
    print(f"{'=' * 80}")
    print(f"  Sites simulated:     {len(results)}")
    print(f"  Total BESS capacity: {sum(r['bess']['capacity_kwh'] for r in results):,} kWh")
    print(f"  Total install cost:  ${total_install_cost:,}")
    print(f"  Annual savings:      ${total_annual_savings:,}")
    print(f"  Annual CO₂ offset:   {total_annual_co2:,.1f} tons")
    print(f"  Portfolio payback:   {avg_payback:.1f} years")
    print(f"\n  Building type breakdown:")
    type_counts = {}
    for r in results:
        t = r["building_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        type_savings = sum(r["annual_results"]["savings_usd"] for r in results if r["building_type"] == t)
        print(f"    {t:10s}: {c:>3d} sites, ${type_savings:>10,}/yr")

    print(f"\n  Seasonal performance (avg across all sites):")
    for season in ["summer", "winter", "spring"]:
        avg_util = np.mean([r["seasonal_detail"][season]["utilization_pct"] for r in results])
        avg_save = np.mean([r["seasonal_detail"][season]["daily_savings_usd"] for r in results])
        avg_peak = np.mean([r["seasonal_detail"][season]["peak_reduction_pct"] for r in results])
        print(f"    {SEASONS[season]['label']:30s}  util:{avg_util:.0f}%  save:${avg_save:.0f}/day  peak:-{avg_peak:.0f}%")

    print(f"{'=' * 80}")

    return results


if __name__ == "__main__":
    # Parse args
    input_file = None
    top_n = 50

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--input" and i + 1 < len(args):
            input_file = args[i + 1]
        elif arg == "--top" and i + 1 < len(args):
            top_n = int(args[i + 1])

    run_simulation(input_file=input_file, top_n=top_n)
