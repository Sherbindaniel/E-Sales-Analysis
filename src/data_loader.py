import os
import pandas as pd

def get_data_path(relative_path):
    """
    Get the absolute path to a file in the data folder.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, relative_path)

def load_raw_ecommerce_data():
    """
    Load raw e-commerce logs from the csv.
    """
    path = get_data_path("data/raw/ecommerce_log/E-commerce Website Logs.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"E-commerce logs not found at {path}")
    # Read with low_memory=False because age has mixed types
    return pd.read_csv(path, low_memory=False)

def load_raw_web_server_data():
    """
    Load raw web server access logs and server metadata.
    """
    access_path = get_data_path("data/raw/web_server_log/access_logs.csv")
    servers_path = get_data_path("data/raw/web_server_log/servers.csv")
    
    if not os.path.exists(access_path):
        raise FileNotFoundError(f"Access logs not found at {access_path}")
    if not os.path.exists(servers_path):
        raise FileNotFoundError(f"Servers metadata not found at {servers_path}")
        
    df_access = pd.read_csv(access_path)
    df_servers = pd.read_csv(servers_path)
    
    return df_access, df_servers
