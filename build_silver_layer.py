import pandas as pd
import os
import glob

DATA_ROOT = "/home/acergn100_6/smart-city-management/data"
RAW_DIR = os.path.join(DATA_ROOT, "raw")
SILVER_DIR = os.path.join(DATA_ROOT, "silver")

os.makedirs(SILVER_DIR, exist_ok=True)

def enforce_global_schema(df, dataset_name):
    bbl_cols = [c for c in df.columns if c.lower() == 'bbl']
    for col in bbl_cols:
        df[col] = df[col].fillna(0).astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
        df[col] = df[col].replace('0000000000', None)
        df[col] = df[col].replace('0000000nan', None)

    bin_cols = [c for c in df.columns if c.lower() == 'bin']
    for col in bin_cols:
        df[col] = df[col].fillna(0).astype(str).str.replace(r'\.0$', '', regex=True)
        df[col] = df[col].replace('0', None)
        df[col] = df[col].replace('nan', None)

    return df

def process_w2(file_path):
    print(f"Processing W2 (311 Data) from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    
    date_col = 'Created Date' if 'Created Date' in df.columns else None
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
        df['yyyy_mm'] = df[date_col].dt.strftime('%Y_%m')
        df['created_year'] = df[date_col].dt.year

    type_col = 'Complaint Type' if 'Complaint Type' in df.columns else 'Problem'
    desc_col = 'Descriptor' if 'Descriptor' in df.columns else 'Problem Detail'
    
    if type_col in df.columns and desc_col in df.columns:
        df['canonical_category'] = (
            df[type_col].fillna('Unknown').astype(str) + " - " + 
            df[desc_col].fillna('Unknown').astype(str)
        )

    df = enforce_global_schema(df, 'W2')
    
    out_path = os.path.join(SILVER_DIR, 'W2_311_dsny.parquet')
    print("Writing W2 Partitioned Parquet...")
    df.to_parquet(out_path, partition_cols=['created_year'])
    print("W2 Done.\n")

def process_e7(file_path):
    print(f"Processing E7 (Monthly LL84) from {file_path}...")
    df = pd.read_csv(file_path)
    
    date_col = 'Month' if 'Month' in df.columns else 'Date'
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['yyyy_mm'] = df[date_col].dt.strftime('%Y_%m')
        df['calendar_year'] = df[date_col].dt.year
        
    df = enforce_global_schema(df, 'E7')
    
    out_path = os.path.join(SILVER_DIR, 'E7_ll84_monthly.parquet')
    print("Writing E7 Partitioned Parquet...")
    df.to_parquet(out_path, partition_cols=['calendar_year'])
    print("E7 Done.\n")

def process_e10(file_path):
    print(f"Processing E10 (LL84 Benchmarking) from {file_path}...")
    df = pd.read_csv(file_path)
    
    required_cols = ['Property ID', 'NYC Borough, Block and Lot (BBL)', 'Property Name', 
                     'Primary Property Type - Self Selected', 'Year Built', 'Latitude', 'Longitude']
    existing = [c for c in required_cols if c in df.columns]
    df = df[existing]
    
    rename_map = {
        'Property ID': 'property_id',
        'NYC Borough, Block and Lot (BBL)': 'bbl',
        'Property Name': 'property_name',
        'Primary Property Type - Self Selected': 'property_type',
        'Year Built': 'year_built',
        'Latitude': 'latitude',
        'Longitude': 'longitude'
    }
    df = df.rename(columns=rename_map)
    df = enforce_global_schema(df, 'E10')
    
    out_path = os.path.join(SILVER_DIR, 'E10_ll84_benchmarking.parquet')
    df.to_parquet(out_path)
    print(f"E10 Done. {len(df)} records written.\n")

def process_standard_file(file_path, filename):
    print(f"Processing {filename}...")
    df = pd.read_csv(file_path)
    df = enforce_global_schema(df, filename)
    
    out_name = filename.replace('.csv', '.parquet')
    out_path = os.path.join(SILVER_DIR, out_name)
    df.to_parquet(out_path)
    print(f"{filename} Done.\n")

def process_large_311(file_path):
    print(f"Processing large 311 dump from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    
    date_col = 'Created Date' if 'Created Date' in df.columns else None
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
        df['yyyy_mm'] = df[date_col].dt.strftime('%Y_%m')
        df['created_year'] = df[date_col].dt.year

    df = enforce_global_schema(df, 'W2')
    
    out_path = os.path.join(SILVER_DIR, 'W2_311_dsny.parquet')
    print(f"Writing partitioned parquet...")
    df.to_parquet(out_path, partition_cols=['created_year'])
    print(f"W2 Done. {len(df)} records.\n")

def main():
    raw_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    
    if not raw_files:
        print(f"No CSVs found in {RAW_DIR}. Check your file paths.")
        return

    for file_path in raw_files:
        filename = os.path.basename(file_path)
        
        if '311_Service_Requests' in filename:
            try:
                process_large_311(file_path)
            except Exception as e:
                print(f"Skipping {filename}: {e}")
            continue
        
        if filename.startswith('W2'):
            process_w2(file_path)
        elif filename.startswith('E7'):
            process_e7(file_path)
        elif 'E10' in filename or 'LL84_2023' in filename or 'Benchmarking' in filename or 'Building_Energy_and_Water_Data_Disclosure' in filename:
            process_e10(file_path)
        elif 'NYC_EV_Fleet' in filename:
            df = pd.read_csv(file_path)
            lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
            lon_col = next((c for c in df.columns if 'lon' in c.lower() or 'lng' in c.lower()), None)
            df = enforce_global_schema(df, filename)
            df.to_parquet(os.path.join(SILVER_DIR, 'E4_ev_fleet_stations.parquet'))
            print(f"E4 done.\n")
        elif 'Solar-Readiness' in filename or 'Local_Law_24' in filename:
            df = pd.read_csv(file_path)
            df = enforce_global_schema(df, filename)
            df.to_parquet(os.path.join(SILVER_DIR, 'E5_solar_readiness.parquet'))
            print(f"E5 done.\n")
        elif 'Food_Scrap_Drop-Off' in filename:
            df = pd.read_csv(file_path)
            df = enforce_global_schema(df, filename)
            df.to_parquet(os.path.join(SILVER_DIR, 'W7_food_scrap_dropoffs.parquet'))
            print(f"W7 done.\n")
        elif 'Location_of_Disposal_Facilities' in filename:
            df = pd.read_csv(file_path)
            df = enforce_global_schema(df, filename)
            df.to_parquet(os.path.join(SILVER_DIR, 'W8_disposal_facilities.parquet'))
            print(f"W8 done.\n")
        elif 'Local_Law_84_Monthly_Data' in filename and 'Calendar_Year' in filename:
            df = pd.read_csv(file_path)
            date_col = 'Month' if 'Month' in df.columns else 'Date'
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df['calendar_year'] = df[date_col].dt.year
            df = enforce_global_schema(df, filename)
            df.to_parquet(os.path.join(SILVER_DIR, 'E7_ll84_monthly.parquet'), partition_cols=['calendar_year'])
            print(f"E7 done.\n")
        elif 'DSNY_Monthly_Tonnage' in filename:
            df = pd.read_csv(file_path)
            df = enforce_global_schema(df, filename)
            df.to_parquet(os.path.join(SILVER_DIR, 'W1_dsny_monthly_tonnage.parquet'))
            print(f"W1 done.\n")
        elif 'Electric_Consumption' in filename:
            df = pd.read_csv(file_path, low_memory=False)
            if 'BBL' in df.columns:
                df['bbl'] = df['BBL'].fillna(0).astype(str).str.zfill(10)
                df['bbl'] = df['bbl'].replace('0000000000', None)
            df = enforce_global_schema(df, filename)
            df.to_parquet(os.path.join(SILVER_DIR, 'E3_electric_consumption.parquet'))
            print(f"E3 done.\n")
        else:
            try:
                process_standard_file(file_path, filename)
            except Exception as e:
                print(f"Skipping {filename}: {e}")

    print("Silver pipeline complete. All files serialized to Parquet.")

if __name__ == "__main__":
    main()