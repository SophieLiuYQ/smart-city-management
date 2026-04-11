import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import cdist

DATA_ROOT = "/home/acergn100_6/smart-city-management/data/"
SILVER_DIR = os.path.join(DATA_ROOT, "silver")

def find_lat_lon_cols(df):
    cols = [c.lower() for c in df.columns]
    
    lat_col = next((df.columns[i] for i, c in enumerate(cols) if c in ['lat', 'latitude', 'y_coord']), None)
    lon_col = next((df.columns[i] for i, c in enumerate(cols) if c in ['lon', 'longitude', 'x_coord', 'long']), None)
    
    return lat_col, lon_col

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def calc_nearest_distances(base_df, poi_df, prefix="poi"):
    b_lat_col, b_lon_col = find_lat_lon_cols(base_df)
    p_lat_col, p_lon_col = find_lat_lon_cols(poi_df)

    if not all([b_lat_col, b_lon_col, p_lat_col, p_lon_col]):
        print(f"Warning: Could not find lat/lon columns for {prefix}. Skipping spatial join.")
        return base_df

    valid_pois = poi_df.dropna(subset=[p_lat_col, p_lon_col])
    
    base_lats = base_df[b_lat_col].fillna(0).values
    base_lons = base_df[b_lon_col].fillna(0).values
    poi_lats = valid_pois[p_lat_col].values
    poi_lons = valid_pois[p_lon_col].values
    
    nearest_dists = np.full(len(base_df), np.inf)
    
    for i, (blat, blon) in enumerate(zip(base_lats, base_lons)):
        dists = haversine_distance(blat, blon, poi_lats, poi_lons)
        nearest_dists[i] = np.min(dists)
    
    base_df[f'nearest_{prefix}_dist_m'] = nearest_dists
    
    mask = base_df[b_lat_col].isna() | base_df[b_lon_col].isna()
    base_df.loc[mask, f'nearest_{prefix}_dist_m'] = np.nan

    return base_df

def process_spatial_features():
    print("Loading datasets for spatial joins...")
    e5_path = os.path.join(SILVER_DIR, "E5_solar_readiness.parquet")
    if not os.path.exists(e5_path):
        print("Error: E5_solar_readiness not found. Run Phase 2 first.")
        return
    e5 = pd.read_parquet(e5_path)

    e4_path = os.path.join(SILVER_DIR, "E4_ev_fleet_stations.parquet")
    w7_path = os.path.join(SILVER_DIR, "W7_food_scrap_dropoffs.parquet")
    w8_path = os.path.join(SILVER_DIR, "W8_disposal_facilities.parquet")

    if os.path.exists(e4_path):
        print("Calculating distances to EV Stations...")
        e4 = pd.read_parquet(e4_path)
        e5 = calc_nearest_distances(e5, e4, prefix="ev")
        e5['ev_within_500m'] = e5['nearest_ev_dist_m'] <= 500.0
        e5['ev_within_1km'] = e5['nearest_ev_dist_m'] <= 1000.0

    if os.path.exists(w7_path):
        print("Calculating distances to Compost Sites...")
        w7 = pd.read_parquet(w7_path)
        e5 = calc_nearest_distances(e5, w7, prefix="compost")
        e5['compost_within_1km'] = e5['nearest_compost_dist_m'] <= 1000.0

    if os.path.exists(w8_path):
        print("Calculating distances to Transfer Facilities...")
        w8 = pd.read_parquet(w8_path)
        e5 = calc_nearest_distances(e5, w8, prefix="transfer")

    out_path = os.path.join(SILVER_DIR, "E5_solar_readiness_enriched.parquet")
    print(f"Saving enriched anchor table to {out_path}...")
    e5.to_parquet(out_path)
    print("Spatial geomathematics complete.")

if __name__ == "__main__":
    process_spatial_features()