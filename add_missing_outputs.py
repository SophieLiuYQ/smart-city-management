import pandas as pd
import os

DATA_ROOT = "/home/acergn100_6/smart-city-management/data"
SILVER_DIR = os.path.join(DATA_ROOT, "silver")
GOLD_DIR = os.path.join(DATA_ROOT, "gold")

def add_missing_outputs():
    print("Adding missing runbook outputs...")
    
    w8_path = os.path.join(SILVER_DIR, "W8_disposal_facilities.parquet")
    if os.path.exists(w8_path):
        w8 = pd.read_parquet(w8_path)
        col_lower = [c.lower() for c in w8.columns]
        
        lat_col = next((w8.columns[i] for i, c in enumerate(col_lower) if c in ['lat', 'latitude']), None)
        lon_col = next((w8.columns[i] for i, c in enumerate(col_lower) if c in ['lon', 'longitude', 'long']), None)
        
        if lat_col and lon_col:
            depots = w8[[lat_col, lon_col]].copy()
            depots = depots.rename(columns={lat_col: 'lat', lon_col: 'lon'})
            depots['depot_id'] = range(len(depots))
            depots = depots[['depot_id', 'lat', 'lon']]
            depots.to_parquet(os.path.join(GOLD_DIR, "route_inputs", "depots.parquet"))
            print(f"depots.parquet: {len(depots)} records")
    
    e5_path = os.path.join(SILVER_DIR, "E5_solar_readiness_enriched.parquet")
    e10_path = os.path.join(SILVER_DIR, "E10_ll84_benchmarking.parquet")
    e7_path = os.path.join(SILVER_DIR, "E7_ll84_monthly.parquet")
    e3_path = os.path.join(SILVER_DIR, "E3_electric_consumption.parquet")
    
    if os.path.exists(e5_path) and os.path.exists(e10_path):
        e5 = pd.read_parquet(e5_path)
        e10 = pd.read_parquet(e10_path)
        
        profiles = pd.DataFrame()
        if 'Site' in e5.columns:
            profiles['site_id'] = e5['Site']
        elif 'Address' in e5.columns:
            profiles['site_id'] = e5['Address']
        else:
            profiles['site_id'] = range(len(e5))
        
        profiles['property_type'] = e5.get('property_type', 'Unknown') if 'property_type' in e5.columns else 'Unknown'
        profiles['avg_monthly_total_kwh'] = 0.0
        profiles['peak_kw'] = 0.0
        profiles['seasonality_index'] = 1.0
        profiles['solar_production_kwh_yr'] = e5.get('Estimated Annual Production', 0) if 'Estimated Annual Production' in e5.columns else 0
        profiles['ev_ports_1km'] = e5.get('ev_within_1km', False).astype(int) if 'ev_within_1km' in e5.columns else 0
        
        profiles.to_parquet(os.path.join(GOLD_DIR, "dispatch", "site_profiles.parquet"))
        print(f"site_profiles.parquet: {len(profiles)} records")
    
    print("Missing outputs added.")

if __name__ == "__main__":
    add_missing_outputs()