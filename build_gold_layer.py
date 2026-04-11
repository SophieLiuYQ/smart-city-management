import pandas as pd
import pyarrow.dataset as ds
import os
import json

DATA_ROOT = "/home/acergn100_6/smart-city-management/data"
SILVER_DIR = os.path.join(DATA_ROOT, "silver")
GOLD_DIR = os.path.join(DATA_ROOT, "gold")

os.makedirs(GOLD_DIR, exist_ok=True)
os.makedirs(os.path.join(GOLD_DIR, "route_inputs"), exist_ok=True)
os.makedirs(os.path.join(GOLD_DIR, "nim"), exist_ok=True)
os.makedirs(os.path.join(GOLD_DIR, "dispatch"), exist_ok=True)

def build_time_series():
    print("Building Gold Table: Time Series...")
    e7_path = os.path.join(SILVER_DIR, "E7_ll84_monthly.parquet")
    if os.path.exists(e7_path):
        try:
            e7 = pd.read_parquet(e7_path)
        except Exception:
            dataset = ds.dataset(e7_path, format="parquet")
            table = dataset.to_table()
            e7 = table.to_pandas()
        out_path = os.path.join(GOLD_DIR, "time_series.parquet")
        e7.to_parquet(out_path)
        print(f" -> Saved {out_path}")
    else:
        print(" -> Error: E7 Silver table missing.")

def build_unified_sites():
    print("Building Gold Table: Unified Sites...")
    e5_path = os.path.join(SILVER_DIR, "E5_solar_readiness_enriched.parquet")
    bridge_path = os.path.join(SILVER_DIR, "property_to_bbl.parquet")
    e10_path = os.path.join(SILVER_DIR, "E10_ll84_benchmarking.parquet")

    e5_exists = os.path.exists(e5_path)
    bridge_exists = os.path.exists(bridge_path)
    e10_exists = os.path.exists(e10_path)
    
    if e5_exists and bridge_exists and e10_exists:
        e5 = pd.read_parquet(e5_path)
        bridge = pd.read_parquet(bridge_path)
        e10 = pd.read_parquet(e10_path)
        
        if 'bbl' in e5.columns:
            unified = e5.merge(bridge, on='bbl', how='left')
        elif 'property_id' in e5.columns:
            unified = e5.merge(e10[['property_id', 'bbl']], on='property_id', how='left')
        else:
            print(" -> Warning: No join key found. Saving E5 as-is.")
            unified = e5
        
        out_path = os.path.join(GOLD_DIR, "unified_sites.parquet")
        unified.to_parquet(out_path)
        print(f" -> Saved {out_path}")
        return unified
    else:
        print(f" -> Error: Missing files. E5:{e5_exists}, Bridge:{bridge_exists}, E10:{e10_exists}")
        return None

def build_district_waste_and_routing():
    print("Building Gold Table: District Waste & Routing Nodes...")
    w1_path = os.path.join(SILVER_DIR, "W1_dsny_monthly_tonnage.parquet")
    w2_path = os.path.join(SILVER_DIR, "W2_311_all.parquet")

    if os.path.exists(w1_path):
        w1 = pd.read_parquet(w1_path)
        
        district_col = next((c for c in w1.columns if 'district' in c.lower() or 'board' in c.lower() or 'community' in c.lower()), None)
        tonnage_col = next((c for c in w1.columns if 'ton' in c.lower() or 'weight' in c.lower()), None)

        if district_col and tonnage_col:
            district_waste = w1.groupby(district_col).agg({tonnage_col: 'sum'}).reset_index()
            district_waste = district_waste.rename(columns={tonnage_col: 'total_tons', district_col: 'district_id'})
            
            demand_nodes = district_waste.copy()
            demand_nodes['lat'] = 40.7128 
            demand_nodes['lon'] = -74.0060
            demand_nodes['complaint_intensity'] = 1.0
            
            dw_path = os.path.join(GOLD_DIR, "district_waste.parquet")
            district_waste.to_parquet(dw_path)
            print(f" -> Saved {dw_path}")

            rn_path = os.path.join(GOLD_DIR, "route_inputs", "demand_nodes.parquet")
            demand_nodes.to_parquet(rn_path)
            print(f" -> Saved {rn_path}")
        else:
            print(f" -> Error: Could not identify district or tonnage columns in W1. Available: {list(w1.columns)}")
    else:
        print(" -> Error: W1 Silver table missing.")

def generate_nim_batches(unified_sites):
    if unified_sites is None:
        return
        
    print("Generating NIM JSONL Batches...")
    
    bbl_col = 'bbl' if 'bbl' in unified_sites.columns else None
    
    if bbl_col:
        valid_sites = unified_sites.dropna(subset=[bbl_col])
    else:
        valid_sites = unified_sites
    
    top_sites = valid_sites.head(20)
    
    jsonl_path = os.path.join(GOLD_DIR, "nim", "site_batches.jsonl")
    with open(jsonl_path, 'w') as f:
        for _, row in top_sites.iterrows():
            site_record = {
                "site": str(row.get('Site', row.get('Address', 'Unknown'))),
                "property_type": str(row.get('property_type', 'Unknown')),
                "year_built": str(row.get('year_built', 'Unknown')),
                "has_ev_within_1km": bool(row.get('ev_within_1km', False)),
                "has_compost_within_1km": bool(row.get('compost_within_1km', False))
            }
            f.write(json.dumps(site_record) + '\n')
            
    print(f" -> Saved {jsonl_path} (Ready for local 8B inference)")

def main():
    print("--- Starting Phase 5 Gold Aggregation ---")
    build_time_series()
    unified = build_unified_sites()
    build_district_waste_and_routing()
    generate_nim_batches(unified)
    print("--- Phase 5 Complete! ---")

if __name__ == "__main__":
    main()