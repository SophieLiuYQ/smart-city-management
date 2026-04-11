"""
XGBoost Site Scoring — End-to-End from Raw CSVs

Single script: reads 12 raw CSVs → cleans → engineers features →
trains XGBoost → scores all sites → outputs ranked results.

No gold layer needed. No LLM. No API. Raw data in, scores out.

Usage (on GB10):
    python AI/score_raw_xgboost.py

    # Custom raw path:
    RAW_DIR=/home/acergn100_6/smart-city-management/data/raw python AI/score_raw_xgboost.py
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.getenv("RAW_DIR", os.path.join(BASE_DIR, "data", "raw"))
OUT_DIR = os.getenv("OUT_DIR", os.path.join(BASE_DIR, "data", "gold"))
os.makedirs(OUT_DIR, exist_ok=True)


def safe_float(series):
    return pd.to_numeric(series, errors="coerce")


# ---------------------------------------------------------------------------
# Filename resolver — works with both naming conventions
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


def find_raw_file(dataset_key):
    """Find the raw CSV file by trying known name patterns."""
    patterns = FILE_PATTERNS.get(dataset_key, [])
    files_in_dir = os.listdir(RAW_DIR)

    # Try exact match first
    for pattern in patterns:
        if pattern in files_in_dir:
            return os.path.join(RAW_DIR, pattern)

    # Try substring match
    for pattern in patterns:
        for f in files_in_dir:
            if pattern.lower() in f.lower() and f.endswith(".csv"):
                return os.path.join(RAW_DIR, f)

    print(f"  [WARN] No file found for {dataset_key}. Tried: {patterns}")
    print(f"         Files in {RAW_DIR}: {files_in_dir[:5]}...")
    return None


def timer(label):
    class T:
        def __enter__(self):
            self.t = time.time()
            print(f"  [{label}]", end=" ", flush=True)
            return self
        def __exit__(self, *a):
            print(f"({time.time()-self.t:.2f}s)")
    return T()


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: Load & Clean Raw CSVs
# ═══════════════════════════════════════════════════════════════════════════

def load_e5():
    """Solar readiness — BASE TABLE (4,268 buildings)."""
    with timer("E5 Solar Readiness"):
        df = pd.read_csv(find_raw_file("E5"), low_memory=False)
        df["lat"] = safe_float(df["Latitude"])
        df["lon"] = safe_float(df["Longitude"])
        df["solar_kwh"] = safe_float(df["Estimated Annual Production"].astype(str).str.replace(",", ""))
        df["solar_savings"] = safe_float(
            df["Estimated Annual Energy Savings"].astype(str).str.replace("$", "").str.replace(",", "")
        )
        df["is_ej"] = (df["Environmental Justice Area"].astype(str).str.lower() == "yes").astype(int)
        df["roof_condition"] = df["Roof Condition"].astype(str).str.lower().str.strip()
        df["roof_good"] = (df["roof_condition"] == "good").astype(int)
        df["roof_fair"] = (df["roof_condition"] == "fair").astype(int)
        df["roof_poor"] = (df["roof_condition"] == "poor").astype(int)
        df["bbl"] = df["BBL"].astype(str).str.strip().str.split(".").str[0].str.zfill(10)
        df["community_board"] = safe_float(df["Community Board"]).fillna(0).astype(int)
        df["sqft"] = safe_float(
            df["Total Gross Square Footage"].astype(str).str.replace(",", "").str.replace("GSF", "").str.strip()
        ).fillna(0)
        print(f"→ {len(df)} sites")
        return df


def load_e3():
    """NYCHA electric consumption → per-development aggregates."""
    with timer("E3 Electric Consumption"):
        df = pd.read_csv(find_raw_file("E3"), low_memory=False)
        df["kwh"] = safe_float(df["Consumption (KWH)"])
        df["kw"] = safe_float(df["Consumption (KW)"])
        df["cost"] = safe_float(df["Current Charges"])
        df = df.dropna(subset=["kwh"])
        df = df[df["kwh"] > 0]
        agg = df.groupby("Development Name").agg(
            e3_avg_kwh=("kwh", "mean"),
            e3_peak_kw=("kw", "max"),
            e3_avg_cost=("cost", "mean"),
        ).reset_index()
        print(f"→ {len(agg)} developments")
        return agg


def load_e4():
    """EV fleet stations — lat/lon + port counts."""
    with timer("E4 EV Stations"):
        df = pd.read_csv(find_raw_file("E4"), low_memory=False)
        df.columns = df.columns.str.strip().str.upper()
        df["lat"] = safe_float(df["LATITUDE"])
        df["lon"] = safe_float(df["LONGITUDE"])
        df["ports"] = safe_float(df.get("NO. OF PLUGS", df.get("NO_OF_PLUGS", pd.Series(1)))).fillna(1).astype(int)
        df = df.dropna(subset=["lat", "lon"])
        print(f"→ {len(df)} stations")
        return df[["lat", "lon", "ports"]]


def load_e7():
    """LL84 monthly energy → per-property aggregates."""
    with timer("E7 LL84 Monthly (2.2M rows)"):
        df = pd.read_csv(find_raw_file("E7"), low_memory=False)
        KBTU = 0.293071
        elec_col = [c for c in df.columns if "Electricity" in c and "kBtu" in c]
        gas_col = [c for c in df.columns if "Natural Gas" in c and "kBtu" in c]
        df["elec_kwh"] = safe_float(df[elec_col[0]]) * KBTU if elec_col else 0
        df["gas_kwh"] = safe_float(df[gas_col[0]]) * KBTU if gas_col else 0
        df["total_kwh"] = df["elec_kwh"].fillna(0) + df["gas_kwh"].fillna(0)
        df = df[df["total_kwh"] > 0]
        df["Property Id"] = df["Property Id"].astype(str)

        agg = df.groupby("Property Id").agg(
            e7_avg_elec=("elec_kwh", "mean"),
            e7_avg_gas=("gas_kwh", "mean"),
            e7_avg_total=("total_kwh", "mean"),
            e7_peak_elec=("elec_kwh", "max"),
            e7_months=("elec_kwh", "count"),
        ).reset_index()

        # Seasonality
        std = df.groupby("Property Id")["elec_kwh"].std().reset_index(name="e7_std")
        mean = df.groupby("Property Id")["elec_kwh"].mean().reset_index(name="e7_mean")
        seas = std.merge(mean, on="Property Id")
        seas["e7_seasonality"] = (seas["e7_std"] / seas["e7_mean"].replace(0, np.nan)).fillna(0)
        agg = agg.merge(seas[["Property Id", "e7_seasonality"]], on="Property Id", how="left")

        print(f"→ {len(agg)} properties")
        return agg


def load_e10():
    """LL84 benchmarking — ENERGY STAR, GHG, building type."""
    with timer("E10 Benchmarking"):
        usecols = [
            "Property ID", "NYC Borough, Block and Lot (BBL)",
            "ENERGY STAR Score", "Site EUI (kBtu/ft²)",
            "Total (Location-Based) GHG Emissions (Metric Tons CO2e)",
            "Year Built", "Primary Property Type - Self Selected",
        ]
        df = pd.read_csv(find_raw_file("E10"),
                         usecols=usecols, low_memory=False)
        df.columns = ["prop_id", "bbl", "energy_star", "site_eui", "ghg", "year_built", "prop_type"]
        df["bbl"] = df["bbl"].astype(str).str.strip().str.split(".").str[0].str.zfill(10)
        df["energy_star"] = safe_float(df["energy_star"]).fillna(-1).astype(int)
        df["site_eui"] = safe_float(df["site_eui"])
        df["ghg"] = safe_float(df["ghg"])
        df["year_built"] = safe_float(df["year_built"]).fillna(0).astype(int)
        df["prop_id"] = df["prop_id"].astype(str)
        # Keep latest per BBL
        df = df.sort_values("energy_star", ascending=False).drop_duplicates("bbl", keep="first")
        print(f"→ {len(df)} properties")
        return df


def load_w1():
    """DSNY monthly tonnage → per-district aggregates."""
    with timer("W1 Tonnage"):
        df = pd.read_csv(find_raw_file("W1"), low_memory=False)
        for c in ["REFUSETONSCOLLECTED", "PAPERTONSCOLLECTED", "MGPTONSCOLLECTED",
                   "RESORGANICSTONS", "SCHOOLORGANICTONS", "LEAVESORGANICTONS"]:
            if c in df.columns:
                df[c] = safe_float(df[c]).fillna(0)
        df["organics"] = df.get("RESORGANICSTONS", 0) + df.get("SCHOOLORGANICTONS", 0) + df.get("LEAVESORGANICTONS", 0)
        df["recycling"] = df["PAPERTONSCOLLECTED"] + df["MGPTONSCOLLECTED"]
        df["total"] = df["REFUSETONSCOLLECTED"] + df["recycling"] + df["organics"]
        df["diversion"] = (df["recycling"] + df["organics"]) / df["total"].replace(0, np.nan)
        df["cd"] = safe_float(df["COMMUNITYDISTRICT"]).fillna(0).astype(int)
        df["borough"] = df["BOROUGH"]
        agg = df.groupby(["borough", "cd"]).agg(
            w1_refuse=("REFUSETONSCOLLECTED", "mean"),
            w1_organics=("organics", "mean"),
            w1_total=("total", "mean"),
            w1_diversion=("diversion", "mean"),
        ).reset_index()
        print(f"→ {len(agg)} districts")
        return agg


def load_w2():
    """311 DSNY complaints → per-district counts."""
    with timer("W2 311 Complaints"):
        df = pd.read_csv(find_raw_file("W2"), low_memory=False)
        df.columns = df.columns.str.strip().str.lower()
        cb = df.get("community_board", df.get("community board", pd.Series("0")))
        df["cd"] = safe_float(cb.astype(str).str.extract(r"(\d+)", expand=False)).fillna(0).astype(int)
        df["borough"] = df.get("borough", "").astype(str).str.strip().str.title()
        agg = df.groupby(["borough", "cd"]).size().reset_index(name="w2_complaints")
        print(f"→ {len(agg)} districts")
        return agg


def load_w7():
    """Food scrap drop-offs — composting site locations."""
    with timer("W7 Compost"):
        df = pd.read_csv(find_raw_file("W7"), low_memory=False)
        df["lat"] = safe_float(df["Latitude"])
        df["lon"] = safe_float(df["Longitude"])
        df = df.dropna(subset=["lat", "lon"])
        print(f"→ {len(df)} sites")
        return df[["lat", "lon"]]


def load_w8():
    """Disposal facilities — transfer station locations."""
    with timer("W8 Transfer Stations"):
        df = pd.read_csv(find_raw_file("W8"), low_memory=False)
        df["lat"] = safe_float(df["Latitude"])
        df["lon"] = safe_float(df["Longitude"])
        df = df.dropna(subset=["lat", "lon"])
        # NYC area only
        df = df[(df["lat"] > 40.4) & (df["lat"] < 41.0) & (df["lon"] > -74.3) & (df["lon"] < -73.7)]
        print(f"→ {len(df)} NYC stations")
        return df[["Name", "lat", "lon"]]


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: Spatial Features
# ═══════════════════════════════════════════════════════════════════════════

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def spatial_features(sites_lat, sites_lon, points_lat, points_lon, radii=[500, 1000]):
    """Vectorized spatial counting — count points within each radius for all sites."""
    n_sites = len(sites_lat)
    n_pts = len(points_lat)

    counts = {r: np.zeros(n_sites, dtype=int) for r in radii}
    nearest = np.full(n_sites, 999999.0)

    # Process in chunks to avoid memory explosion
    chunk = 500
    for i in range(0, n_sites, chunk):
        end = min(i + chunk, n_sites)
        s_lat = sites_lat[i:end, None]  # (chunk, 1)
        s_lon = sites_lon[i:end, None]
        p_lat = points_lat[None, :]     # (1, n_pts)
        p_lon = points_lon[None, :]

        dists = haversine(s_lat, s_lon, p_lat, p_lon)  # (chunk, n_pts)
        nearest[i:end] = dists.min(axis=1)
        for r in radii:
            counts[r][i:end] = (dists <= r).sum(axis=1)

    return counts, nearest


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: Build Feature Matrix
# ═══════════════════════════════════════════════════════════════════════════

def build_feature_matrix(e5, e3, e4, e7, e10, w1, w2, w7, w8):
    """Merge all data sources into one feature matrix aligned to E5 sites."""
    print("\n── BUILDING FEATURE MATRIX ──")
    df = e5.copy()

    # --- BBL joins ---
    with timer("BBL joins (E10)"):
        df = df.merge(
            e10[["bbl", "energy_star", "site_eui", "ghg", "year_built"]],
            on="bbl", how="left"
        )
        df["energy_star"] = df["energy_star"].fillna(-1).astype(int)
        df["site_eui"] = df["site_eui"].fillna(0)
        df["ghg"] = df["ghg"].fillna(0)
        df["year_built"] = df["year_built"].fillna(0).astype(int)
        print(f"→ {df['energy_star'].gt(0).sum()} matched")

    # --- E7 join via E10 bridge (property_id → bbl) ---
    with timer("E7 energy join via BBL"):
        e10_bridge = e10[["prop_id", "bbl"]].drop_duplicates()
        e7_with_bbl = e7.merge(e10_bridge, left_on="Property Id", right_on="prop_id", how="inner")
        e7_bbl = e7_with_bbl.groupby("bbl").agg(
            e7_avg_elec=("e7_avg_elec", "mean"),
            e7_avg_total=("e7_avg_total", "mean"),
            e7_peak_elec=("e7_peak_elec", "max"),
            e7_seasonality=("e7_seasonality", "mean"),
        ).reset_index()
        df = df.merge(e7_bbl, on="bbl", how="left")
        for c in ["e7_avg_elec", "e7_avg_total", "e7_peak_elec", "e7_seasonality"]:
            df[c] = df[c].fillna(0)
        print(f"→ {df['e7_avg_total'].gt(0).sum()} matched")

    # --- District joins (W1, W2) ---
    with timer("District joins (W1+W2)"):
        # Map borough names
        boro_map = {"Bronx": "Bronx", "Brooklyn": "Brooklyn", "Manhattan": "Manhattan",
                    "Queens": "Queens", "Staten Island": "Staten Island"}
        df["_boro"] = df["Borough"].map(boro_map).fillna(df["Borough"])
        df["_cd"] = df["community_board"].astype(int)

        df = df.merge(w1, left_on=["_boro", "_cd"], right_on=["borough", "cd"], how="left", suffixes=("", "_w1"))
        df = df.merge(w2, left_on=["_boro", "_cd"], right_on=["borough", "cd"], how="left", suffixes=("", "_w2"))
        for c in ["w1_refuse", "w1_organics", "w1_total", "w1_diversion", "w2_complaints"]:
            if c in df.columns:
                df[c] = df[c].fillna(0)
        print(f"→ {df['w1_total'].gt(0).sum()} with waste data")

    # --- Spatial features ---
    valid_geo = df["lat"].notna() & df["lon"].notna()
    s_lat = df.loc[valid_geo, "lat"].values
    s_lon = df.loc[valid_geo, "lon"].values

    with timer("Spatial: EV stations"):
        ev_counts, ev_nearest = spatial_features(s_lat, s_lon, e4["lat"].values, e4["lon"].values, [500, 1000])
        df.loc[valid_geo, "ev_500m"] = ev_counts[500]
        df.loc[valid_geo, "ev_1km"] = ev_counts[1000]
        df.loc[valid_geo, "nearest_ev_m"] = ev_nearest
        print(f"→ done")

    with timer("Spatial: Composting"):
        comp_counts, comp_nearest = spatial_features(s_lat, s_lon, w7["lat"].values, w7["lon"].values, [1000])
        df.loc[valid_geo, "compost_1km"] = comp_counts[1000]
        df.loc[valid_geo, "nearest_compost_m"] = comp_nearest
        print(f"→ done")

    with timer("Spatial: Transfer stations"):
        xfer_counts, xfer_nearest = spatial_features(s_lat, s_lon, w8["lat"].values, w8["lon"].values, [5000])
        df.loc[valid_geo, "nearest_transfer_m"] = xfer_nearest
        print(f"→ done")

    # Fill spatial NaNs
    for c in ["ev_500m", "ev_1km", "nearest_ev_m", "compost_1km",
              "nearest_compost_m", "nearest_transfer_m"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)

    # --- Final feature columns ---
    feature_cols = [
        "solar_kwh", "solar_savings", "is_ej", "roof_good", "roof_fair", "roof_poor",
        "sqft", "energy_star", "site_eui", "ghg", "year_built",
        "e7_avg_elec", "e7_avg_total", "e7_peak_elec", "e7_seasonality",
        "w1_refuse", "w1_organics", "w1_total", "w1_diversion", "w2_complaints",
        "ev_500m", "ev_1km", "nearest_ev_m",
        "compost_1km", "nearest_compost_m", "nearest_transfer_m",
    ]
    # Borough one-hot
    for b in ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]:
        col = f"boro_{b.lower().replace(' ', '_')}"
        df[col] = (df["Borough"] == b).astype(int)
        feature_cols.append(col)

    features = df[feature_cols].fillna(0).astype(np.float32)

    print(f"\n  Feature matrix: {features.shape[0]} sites × {features.shape[1]} features")
    return df, features, feature_cols


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4: Train XGBoost & Score
# ═══════════════════════════════════════════════════════════════════════════

def generate_labels(features, feature_cols):
    """Self-supervised proxy scores from known relationships."""
    f = features
    n = len(f)
    np.random.seed(42)

    col = {name: i for i, name in enumerate(feature_cols)}

    solar = f[:, col["solar_kwh"]]
    sqft = f[:, col["sqft"]]
    ej = f[:, col["is_ej"]]
    roof_g = f[:, col["roof_good"]]
    ev_1km = f[:, col["ev_1km"]]
    e7_total = f[:, col["e7_avg_total"]]
    w1_refuse = f[:, col["w1_refuse"]]
    w1_div = f[:, col["w1_diversion"]]
    compost = f[:, col["compost_1km"]]
    transfer = f[:, col["nearest_transfer_m"]]
    ghg = f[:, col["ghg"]]

    def rank_pct(arr):
        from scipy.stats import rankdata
        return rankdata(arr, method="average") / len(arr)

    energy_raw = (
        rank_pct(solar) * 25 +
        rank_pct(e7_total) * 25 +
        ej * 15 +
        roof_g * 10 +
        rank_pct(sqft) * 10 +
        rank_pct(ghg) * 15
    )
    energy = energy_raw / energy_raw.max() * 100 + np.random.normal(0, 2, n)

    waste_raw = (
        ej * 25 +
        (1 - np.clip(compost, 0, 1)) * 15 +
        rank_pct(w1_refuse) * 20 +
        (1 - np.clip(w1_div, 0, 1)) * 20 +
        rank_pct(transfer) * 10 +
        rank_pct(sqft) * 10
    )
    waste = waste_raw / waste_raw.max() * 100 + np.random.normal(0, 2, n)

    nexus_raw = (
        energy_raw * 0.4 +
        waste_raw * 0.4 +
        (ej * rank_pct(solar)) * 30 +
        (rank_pct(ev_1km) * rank_pct(solar)) * 20
    )
    nexus = nexus_raw / nexus_raw.max() * 100 + np.random.normal(0, 2, n)

    return np.clip(energy, 0, 100), np.clip(waste, 0, 100), np.clip(nexus, 0, 100)


def train_and_score(features, feature_cols, energy_y, waste_y, nexus_y):
    """Train 3 XGBoost models and predict."""
    X = features

    # Detect GPU
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
            "max_depth": 6, "eta": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 5, "eval_metric": "rmse",
            **gpu_params,
        }
        model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
        preds = np.clip(model.predict(dtrain), 0, 100).astype(int)
        elapsed = time.time() - t0
        results[name] = {"model": model, "preds": preds, "time": elapsed}
        print(f"    {name}_score: {elapsed:.2f}s  mean={preds.mean():.1f}  range=[{preds.min()}, {preds.max()}]")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5: Output
# ═══════════════════════════════════════════════════════════════════════════

def save_results(sites_df, results, feature_cols):
    """Build ranked output and save."""
    ranked = sites_df[["Site", "Address", "Borough", "Agency",
                        "Environmental Justice Area", "lat", "lon", "bbl"]].copy()
    ranked["energy_score"] = results["energy"]["preds"]
    ranked["waste_score"] = results["waste"]["preds"]
    ranked["nexus_score"] = results["nexus"]["preds"]

    # BESS recommendation
    solar = sites_df["solar_kwh"].fillna(0).values
    ranked["recommended_bess_kwh"] = np.where(solar > 100000, 750,
                                     np.where(solar > 10000, 500,
                                     np.where(ranked["energy_score"] > 60, 250, 100)))
    ranked["estimated_annual_savings_usd"] = np.where(solar > 100000, 50000 + (solar * 0.05).astype(int),
                                             np.where(solar > 10000, 20000, 5000))

    # Recommendations
    recs = []
    for _, r in ranked.iterrows():
        is_ej = str(r.get("Environmental Justice Area", "")).lower() == "yes"
        bess = r["recommended_bess_kwh"]
        if r["nexus_score"] >= 75:
            recs.append(f"Deploy {bess} kWh BESS + partner with AD facility for organics-to-energy")
        elif r["energy_score"] >= 70:
            recs.append(f"Install {bess} kWh BESS paired with solar")
        elif r["waste_score"] >= 70:
            recs.append("Prioritize organic waste diversion to nearest composting/AD facility")
        elif is_ej:
            recs.append(f"EJ priority: evaluate {bess} kWh BESS + community solar")
        else:
            recs.append("Monitor for future solar-readiness assessment")
    ranked["top_recommendation"] = recs

    ranked = ranked.sort_values("nexus_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)

    # Save parquet
    out_path = os.path.join(OUT_DIR, "ranked_sites_xgboost_raw.parquet")
    ranked.to_parquet(out_path, index=False)
    print(f"\n[SAVED] {out_path} — {len(ranked)} sites")

    # Save top 50 JSON
    top50 = ranked.head(50).to_dict(orient="records")
    json_path = os.path.join(OUT_DIR, "top50_xgboost_raw.json")
    with open(json_path, "w") as f:
        json.dump(top50, f, indent=2, default=str)
    print(f"[SAVED] {json_path}")

    # Feature importance
    print("\n[IMPORTANCE] Top 10 features for nexus_score:")
    imp = results["nexus"]["model"].get_score(importance_type="gain")
    for fname, gain in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {fname:30s}  gain: {gain:.1f}")

    # Top 10
    print(f"\n  TOP 10 (by Nexus Score):")
    print("  " + "─" * 65)
    for _, r in ranked.head(10).iterrows():
        print(f"  #{int(r['rank']):>3}  E:{int(r['energy_score']):>3}  W:{int(r['waste_score']):>3}  "
              f"N:{int(r['nexus_score']):>3}  {str(r['Borough']):>12}  {str(r['Site'])[:35]}")
    print("  " + "─" * 65)

    return ranked


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  XGBoost END-TO-END: Raw CSVs → Scores")
    print(f"  Raw data: {RAW_DIR}")
    print("=" * 65)

    t_total = time.time()

    # Phase 1: Load raw
    print("\n── PHASE 1: LOAD RAW DATA ──")
    e5 = load_e5()
    e3 = load_e3()
    e4 = load_e4()
    e7 = load_e7()
    e10 = load_e10()
    w1 = load_w1()
    w2 = load_w2()
    w7 = load_w7()
    w8 = load_w8()

    # Phase 2+3: Features
    sites_df, features_df, feature_cols = build_feature_matrix(e5, e3, e4, e7, e10, w1, w2, w7, w8)
    X = features_df.values

    # Phase 4: Train & score
    print("\n── PHASE 4: TRAIN & SCORE ──")
    energy_y, waste_y, nexus_y = generate_labels(X, feature_cols)
    results = train_and_score(X, feature_cols, energy_y, waste_y, nexus_y)

    # Phase 5: Output
    print("\n── PHASE 5: SAVE RESULTS ──")
    save_results(sites_df, results, feature_cols)

    elapsed = time.time() - t_total
    train_time = sum(r["time"] for r in results.values())

    print(f"\n{'=' * 65}")
    print(f"  COMPLETE — {elapsed:.1f}s total")
    print(f"    Data loading:   {elapsed - train_time:.1f}s")
    print(f"    Training:       {train_time:.1f}s (3 models)")
    print(f"    Sites scored:   {len(sites_df)}")
    print(f"    Throughput:     {len(sites_df)/elapsed:.0f} sites/sec")
    print(f"    Features:       {len(feature_cols)}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
