import pandas as pd
import os

DATA_ROOT = "/home/acergn100_6/smart-city-management/data"
SILVER_DIR = os.path.join(DATA_ROOT, "silver")

def build_bridge():
    e10_path = os.path.join(SILVER_DIR, "E10_ll84_benchmarking.parquet")
    
    if not os.path.exists(e10_path):
        print(f"Error: Could not find {e10_path}. Did Phase 2 finish successfully?")
        return

    print(f"Loading E10 from {e10_path}...")
    df = pd.read_parquet(e10_path)

    cols = df.columns
    
    prop_id_col = next((c for c in cols if 'property id' in c.lower() or 'property_id' in c.lower()), None)
    bbl_col = next((c for c in cols if 'bbl' in c.lower()), None)
    
    if not prop_id_col or not bbl_col:
        raise ValueError(f"Missing critical mapping keys! Available columns: {cols}")

    name_col = next((c for c in cols if 'property name' in c.lower()), None)
    type_col = next((c for c in cols if 'property type' in c.lower()), None)
    year_col = next((c for c in cols if 'year built' in c.lower()), None)
    lat_col = next((c for c in cols if 'latitude' in c.lower()), None)
    lon_col = next((c for c in cols if 'longitude' in c.lower()), None)

    target_cols = [prop_id_col, bbl_col, name_col, type_col, year_col, lat_col, lon_col]
    keep_cols = [c for c in target_cols if c is not None]

    print("Extracting the following columns:")
    for c in keep_cols:
        print(f" - {c}")

    bridge_df = df[keep_cols].copy()

    rename_map = {
        prop_id_col: 'property_id',
        bbl_col: 'bbl',
        name_col: 'property_name',
        type_col: 'property_type',
        year_col: 'year_built',
        lat_col: 'latitude',
        lon_col: 'longitude'
    }
    rename_map = {k: v for k, v in rename_map.items() if k is not None}
    bridge_df = bridge_df.rename(columns=rename_map)

    initial_rows = len(bridge_df)
    bridge_df = bridge_df.drop_duplicates(subset=['property_id'])
    final_rows = len(bridge_df)
    
    print(f"Dropped {initial_rows - final_rows} duplicate records.")

    out_path = os.path.join(SILVER_DIR, "property_to_bbl.parquet")
    bridge_df.to_parquet(out_path)
    
    print(f"Bridge table saved successfully to {out_path} with {final_rows} unique properties.")

if __name__ == "__main__":
    build_bridge()