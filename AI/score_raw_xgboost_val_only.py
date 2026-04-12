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

    # --- Final feature columns (NO is_ej — removed to see what else matters) ---
    feature_cols = [
        "solar_kwh", "solar_savings", "roof_good", "roof_fair", "roof_poor",
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
    roof_g = f[:, col["roof_good"]]
    ev_1km = f[:, col["ev_1km"]]
    e7_total = f[:, col["e7_avg_total"]]
    w1_refuse = f[:, col["w1_refuse"]]
    w1_div = f[:, col["w1_diversion"]]
    compost = f[:, col["compost_1km"]]
    transfer = f[:, col["nearest_transfer_m"]]
    ghg = f[:, col["ghg"]]
    site_eui = f[:, col["site_eui"]]
    e_star = f[:, col["energy_star"]]
    ev_500m = f[:, col["ev_500m"]]
    w1_organics = f[:, col["w1_organics"]]

    def rank_pct(arr):
        from scipy.stats import rankdata
        return rankdata(arr, method="average") / len(arr)

    energy_raw = (
        rank_pct(solar) * 20 +
        rank_pct(e7_total) * 20 +
        rank_pct(ghg) * 15 +
        rank_pct(site_eui) * 10 +
        roof_g * 10 +
        rank_pct(sqft) * 10 +
        rank_pct(ev_1km) * 10 +
        (1 - rank_pct(np.clip(e_star, 0, 100))) * 5  # low ENERGY STAR = more room to improve
    )
    energy = energy_raw / energy_raw.max() * 100 + np.random.normal(0, 3, n)

    waste_raw = (
        rank_pct(w1_refuse) * 25 +
        (1 - np.clip(w1_div, 0, 1)) * 20 +
        rank_pct(w1_organics) * 15 +
        (1 - np.clip(rank_pct(compost), 0, 1)) * 15 +  # fewer compost sites = more opportunity
        rank_pct(transfer) * 15 +  # farther from transfer = worse logistics
        rank_pct(sqft) * 10
    )
    waste = waste_raw / waste_raw.max() * 100 + np.random.normal(0, 3, n)

    nexus_raw = (
        energy_raw * 0.35 +
        waste_raw * 0.35 +
        (rank_pct(solar) * rank_pct(w1_refuse)) * 20 +  # solar + waste in same area
        (rank_pct(ev_1km) * rank_pct(solar)) * 10        # EV + solar synergy
    )
    nexus = nexus_raw / nexus_raw.max() * 100 + np.random.normal(0, 3, n)

    return np.clip(energy, 0, 100), np.clip(waste, 0, 100), np.clip(nexus, 0, 100)


def train_and_score(features, feature_cols, energy_y, waste_y, nexus_y, sites_df):
    """
    Train on train set, early stop on val set, predict ONLY on val set.
    The output file contains ONLY the 640 val sites — true unseen predictions.
    """
    X = features
    n = len(X)

    # Detect GPU
    try:
        gpu_params = {"device": "cuda", "tree_method": "hist"}
        xgb.DMatrix(X[:5], label=energy_y[:5])
        mode = "GPU"
    except Exception:
        gpu_params = {}
        mode = "CPU"
    print(f"  [XGBoost] {mode} mode")

    # ── Train / Validation / Test split: 70% / 15% / 15% ──
    np.random.seed(42)
    indices = np.random.permutation(n)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    print(f"\n  Split: Train={len(train_idx)} ({len(train_idx)/n*100:.0f}%) | "
          f"Val={len(val_idx)} ({len(val_idx)/n*100:.0f}%) | "
          f"Test={len(test_idx)} ({len(test_idx)/n*100:.0f}%)")
    print(f"\n  Model trains on Train, early-stops on Val.")
    print(f"  Output contains ONLY Val sites ({len(val_idx)}) — purely unseen predictions.")
    print(f"  Test set ({len(test_idx)}) held out as final verification.\n")

    base_params = {
        "objective": "reg:squarederror",
        "max_depth": 3,
        "eta": 0.1,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "eval_metric": "rmse",
        **gpu_params,
    }

    results = {}
    val_preds_all = {}
    test_preds_all = {}

    print(f"  {'Score':15s}  {'Train RMSE':>10s}  {'Val RMSE':>10s}  {'Val R²':>8s}  "
          f"{'Test RMSE':>10s}  {'Test R²':>8s}  {'Rounds':>6s}  {'Overfit':>8s}")
    print(f"  {'─'*90}")

    for name, y in [("energy", energy_y), ("waste", waste_y), ("nexus", nexus_y)]:
        t0 = time.time()

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

        # Early stopping on validation set
        model = xgb.train(
            base_params, dtrain,
            num_boost_round=500,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=20,
            verbose_eval=False,
        )

        best_rounds = model.best_iteration + 1

        # Predictions on all three splits
        train_preds = model.predict(dtrain)
        val_preds = model.predict(dval)
        test_preds = model.predict(dtest)

        val_preds_all[name] = np.clip(val_preds, 0, 100).astype(int)
        test_preds_all[name] = np.clip(test_preds, 0, 100).astype(int)

        # Metrics
        train_rmse = np.sqrt(np.mean((train_preds - y_train) ** 2))
        val_rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))
        test_rmse = np.sqrt(np.mean((test_preds - y_test) ** 2))

        ss_res_val = np.sum((y_val - val_preds) ** 2)
        ss_tot_val = np.sum((y_val - y_val.mean()) ** 2)
        val_r2 = 1 - ss_res_val / ss_tot_val if ss_tot_val > 0 else 0

        ss_res_test = np.sum((y_test - test_preds) ** 2)
        ss_tot_test = np.sum((y_test - y_test.mean()) ** 2)
        test_r2 = 1 - ss_res_test / ss_tot_test if ss_tot_test > 0 else 0

        overfit_gap = val_rmse - train_rmse
        overfit = "YES" if overfit_gap > 2 else "mild" if overfit_gap > 1 else "no"

        elapsed = time.time() - t0

        print(f"  {name+'_score':15s}  {train_rmse:>10.2f}  {val_rmse:>10.2f}  {val_r2:>8.3f}  "
              f"{test_rmse:>10.2f}  {test_r2:>8.3f}  {best_rounds:>6d}  {overfit:>8s}")

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
        }

    # Summary
    avg_val_r2 = np.mean([results[n]["val_r2"] for n in ["energy", "waste", "nexus"]])
    avg_test_r2 = np.mean([results[n]["test_r2"] for n in ["energy", "waste", "nexus"]])

    print(f"\n  RESULTS:")
    print(f"    Val R²  (model used this for early stopping): {avg_val_r2:.3f}")
    print(f"    Test R² (model NEVER saw this data):          {avg_test_r2:.3f}")

    if abs(avg_val_r2 - avg_test_r2) < 0.03:
        print(f"    ✅ Val ≈ Test — model is stable and generalizes")
    elif avg_val_r2 > avg_test_r2:
        print(f"    ⚠️ Val > Test by {avg_val_r2 - avg_test_r2:.3f} — slight optimism from early stopping")
    else:
        print(f"    ✅ Test > Val — model generalizes even better on fully unseen data")

    # Build output for VAL sites only
    val_sites = sites_df.iloc[val_idx].copy()
    val_sites["energy_score"] = val_preds_all["energy"]
    val_sites["waste_score"] = val_preds_all["waste"]
    val_sites["nexus_score"] = val_preds_all["nexus"]
    val_sites["split"] = "validation"

    # Also build test output for comparison
    test_sites = sites_df.iloc[test_idx].copy()
    test_sites["energy_score"] = test_preds_all["energy"]
    test_sites["waste_score"] = test_preds_all["waste"]
    test_sites["nexus_score"] = test_preds_all["nexus"]
    test_sites["split"] = "test"

    # Store for save_results
    results["_val_sites"] = val_sites
    results["_test_sites"] = test_sites
    results["_val_idx"] = val_idx
    results["_test_idx"] = test_idx
    results["_avg_val_r2"] = avg_val_r2
    results["_avg_test_r2"] = avg_test_r2

    return results


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5: Output
# ═══════════════════════════════════════════════════════════════════════════

def save_results(sites_df, results, feature_cols):
    """Save VAL-only and TEST-only results — purely unseen predictions."""
    val_sites = results["_val_sites"]
    test_sites = results["_test_sites"]

    def build_ranked(df):
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

    val_ranked = build_ranked(val_sites)
    test_ranked = build_ranked(test_sites)

    # Save val-only
    val_path = os.path.join(OUT_DIR, "ranked_sites_val_only.parquet")
    val_ranked.to_parquet(val_path, index=False)
    print(f"\n[SAVED] {val_path} — {len(val_ranked)} val sites")

    val_json = os.path.join(OUT_DIR, "top50_val_only.json")
    with open(val_json, "w") as f:
        json.dump(val_ranked.head(50).to_dict(orient="records"), f, indent=2, default=str)
    print(f"[SAVED] {val_json}")

    # Save test-only
    test_path = os.path.join(OUT_DIR, "ranked_sites_test_only.parquet")
    test_ranked.to_parquet(test_path, index=False)
    print(f"[SAVED] {test_path} — {len(test_ranked)} test sites")

    test_json = os.path.join(OUT_DIR, "top50_test_only.json")
    with open(test_json, "w") as f:
        json.dump(test_ranked.head(50).to_dict(orient="records"), f, indent=2, default=str)
    print(f"[SAVED] {test_json}")

    # Feature importance
    print("\n[IMPORTANCE] Top 10 features for nexus_score:")
    imp = results["nexus"]["model"].get_score(importance_type="gain")
    for fname, gain in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {fname:30s}  gain: {gain:.1f}")

    # Print val top 10
    print(f"\n  VAL TOP 10 ({len(val_ranked)} sites — model early-stopped on these):")
    print("  " + "─" * 75)
    for _, r in val_ranked.head(10).iterrows():
        print(f"  #{int(r['rank']):>3d}  E:{int(r['energy_score']):>3d}  W:{int(r['waste_score']):>3d}  "
              f"N:{int(r['nexus_score']):>3d}  {str(r['Borough']):>12s}  {str(r['Site'])[:35]}")

    # Print test top 10
    print(f"\n  TEST TOP 10 ({len(test_ranked)} sites — model NEVER saw these):")
    print("  " + "─" * 75)
    for _, r in test_ranked.head(10).iterrows():
        print(f"  #{int(r['rank']):>3d}  E:{int(r['energy_score']):>3d}  W:{int(r['waste_score']):>3d}  "
              f"N:{int(r['nexus_score']):>3d}  {str(r['Borough']):>12s}  {str(r['Site'])[:35]}")
    print("  " + "─" * 75)

    # Compare distributions
    print(f"\n  DISTRIBUTION COMPARISON:")
    print(f"  {'':15s}  {'Val mean':>8s}  {'Test mean':>9s}  {'Diff':>6s}")
    for col in ["energy_score", "waste_score", "nexus_score"]:
        v = val_ranked[col].mean()
        t = test_ranked[col].mean()
        print(f"  {col:15s}  {v:>8.1f}  {t:>9.1f}  {abs(v-t):>6.1f}")

    return val_ranked, test_ranked


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  XGBoost VAL-ONLY: Train → Predict on unseen Val & Test")
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
    print("\n── PHASE 4: TRAIN & SCORE (val+test only) ──")
    energy_y, waste_y, nexus_y = generate_labels(X, feature_cols)
    results = train_and_score(X, feature_cols, energy_y, waste_y, nexus_y, sites_df)

    # Phase 5: Output
    print("\n── PHASE 5: SAVE RESULTS ──")
    save_results(sites_df, results, feature_cols)

    elapsed = time.time() - t_total
    train_time = sum(results[n]["time"] for n in ["energy", "waste", "nexus"])
    avg_val_r2 = results["_avg_val_r2"]
    avg_test_r2 = results["_avg_test_r2"]

    print(f"\n{'=' * 65}")
    print(f"  COMPLETE — {elapsed:.1f}s total")
    print(f"    Training:     {train_time:.1f}s (3 models)")
    print(f"    Val sites:    {len(results['_val_idx'])} (output)")
    print(f"    Test sites:   {len(results['_test_idx'])} (verification)")
    print(f"    Val R²:       {avg_val_r2:.3f}")
    print(f"    Test R²:      {avg_test_r2:.3f}")
    print(f"    Features:     {len(feature_cols)}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
