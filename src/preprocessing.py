import os
import pandas as pd
import numpy as np

def clean_ecommerce_data(df):
    """
    Clean the e-commerce website logs:
    - Parse accessed_date to datetime
    - Convert age to numeric, replacing '--' and other non-numeric values with NaN
    - Drop duplicates if any
    """
    df_clean = df.copy()
    
    # Parse accessed_date
    df_clean['accessed_date'] = pd.to_datetime(df_clean['accessed_date'], errors='coerce')
    
    # Handle age column: convert '--' and others to NaN, then float
    df_clean['age'] = df_clean['age'].replace('--', np.nan)
    df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce')
    
    # Drop duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    dropped_dupes = initial_rows - len(df_clean)
    if dropped_dupes > 0:
        print(f"Dropped {dropped_dupes} duplicate rows from E-Commerce logs.")
        
    return df_clean

def clean_web_server_data(df_access, df_servers):
    """
    Clean the web server access logs:
    - Parse timestamp to datetime
    - Drop duplicates if any
    - Join with server metadata on server_id = id
    """
    df_access_clean = df_access.copy()
    
    # Parse timestamp
    df_access_clean['timestamp'] = pd.to_datetime(df_access_clean['timestamp'], errors='coerce')
    
    # Drop duplicates
    initial_rows = len(df_access_clean)
    df_access_clean = df_access_clean.drop_duplicates()
    dropped_dupes = initial_rows - len(df_access_clean)
    if dropped_dupes > 0:
        print(f"Dropped {dropped_dupes} duplicate rows from web server access logs.")
        
    # Join with server metadata
    df_joined = df_access_clean.merge(df_servers, left_on='server_id', right_on='id', suffixes=('', '_server'))
    # Drop the duplicate 'id' column from servers
    if 'id' in df_joined.columns:
        df_joined = df_joined.drop(columns=['id'])
        
    return df_joined

def run_preprocessing_pipeline(df_ecommerce, df_access, df_servers, output_dir="data/processed"):
    """
    Run the preprocessing pipeline on raw data and save results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Preprocessing E-Commerce logs...")
    df_ec_clean = clean_ecommerce_data(df_ecommerce)
    ec_output_path = os.path.join(output_dir, "clean_ecommerce_logs.csv")
    df_ec_clean.to_csv(ec_output_path, index=False)
    print(f"Saved clean E-Commerce logs to {ec_output_path}")
    
    print("Preprocessing Web Server access logs...")
    df_ws_clean = clean_web_server_data(df_access, df_servers)
    ws_output_path = os.path.join(output_dir, "clean_web_server_logs.csv")
    df_ws_clean.to_csv(ws_output_path, index=False)
    print(f"Saved clean Web Server logs to {ws_output_path}")
    
    return df_ec_clean, df_ws_clean
