import pandas as pd
import numpy as np

def aggregate_ecommerce_to_minutes(df):
    """
    Aggregate E-Commerce logs into 1-minute intervals.
    Includes date-range reindexing to ensure minutes with 0 requests are included.
    """
    df = df.copy()
    
    # Extract minute floor
    df['minute_bin'] = df['accessed_date'].dt.floor('min')
    
    # Group and aggregate
    agg_df = df.groupby('minute_bin').agg(
        requests_per_minute=('ip', 'count'),
        unique_ips=('ip', 'nunique'),
        total_sales=('sales', 'sum'),
        avg_duration_sec=('duration_(secs)', 'mean'),
        total_bytes=('bytes', 'sum'),
        avg_bytes=('bytes', 'mean')
    ).reset_index()
    
    # Create a complete minute index from min to max date
    min_time = df['accessed_date'].min().floor('min')
    max_time = df['accessed_date'].max().floor('min')
    full_time_index = pd.date_range(start=min_time, end=max_time, freq='min')
    full_df = pd.DataFrame({'minute_bin': full_time_index})
    
    # Merge to fill gaps
    merged_df = pd.merge(full_df, agg_df, on='minute_bin', how='left')
    
    # Fill missing values
    merged_df['requests_per_minute'] = merged_df['requests_per_minute'].fillna(0).astype(int)
    merged_df['unique_ips'] = merged_df['unique_ips'].fillna(0).astype(int)
    merged_df['total_sales'] = merged_df['total_sales'].fillna(0.0)
    merged_df['avg_duration_sec'] = merged_df['avg_duration_sec'].fillna(0.0)
    merged_df['total_bytes'] = merged_df['total_bytes'].fillna(0).astype(int)
    merged_df['avg_bytes'] = merged_df['avg_bytes'].fillna(0.0)
    
    # Add rolling features
    merged_df['rolling_mean'] = merged_df['requests_per_minute'].rolling(window=15, min_periods=1).mean()
    merged_df['rolling_std'] = merged_df['requests_per_minute'].rolling(window=15, min_periods=1).std().fillna(0.0)
    merged_df['traffic_volatility'] = merged_df['rolling_std']
    
    # Extract temporal features
    merged_df['hour'] = merged_df['minute_bin'].dt.hour
    merged_df['day_of_week'] = merged_df['minute_bin'].dt.dayofweek
    merged_df['is_weekend'] = merged_df['day_of_week'].isin([5, 6]).astype(int)
    
    # Classify traffic levels: operationally define peak traffic as >= 90th percentile
    # If p90 is 0 (due to sparse data), set threshold to 1 to separate active vs idle traffic
    p90 = np.percentile(merged_df['requests_per_minute'], 90)
    threshold = max(p90, 1.0)
    merged_df['traffic_level'] = np.where(merged_df['requests_per_minute'] >= threshold, 'peak', 'normal')
    
    return merged_df, threshold

def aggregate_web_server_to_minutes(df_access):
    """
    Aggregate Web Server access logs into 1-minute intervals.
    Includes date-range reindexing to ensure minutes with 0 requests are included.
    """
    df = df_access.copy()
    
    # Extract minute floor
    df['minute_bin'] = df['timestamp'].dt.floor('min')
    
    # Create error indicator
    df['is_error'] = (df['status_code'] >= 400).astype(int)
    
    # Group and aggregate
    agg_df = df.groupby('minute_bin').agg(
        requests_per_minute=('ip_address', 'count'),
        unique_ips=('ip_address', 'nunique'),
        avg_response_time_ms=('response_time_ms', 'mean'),
        total_bytes_sent=('bytes_sent', 'sum'),
        error_rate=('is_error', 'mean')
    ).reset_index()
    
    # Create a complete minute index from min to max date
    min_time = df['timestamp'].min().floor('min')
    max_time = df['timestamp'].max().floor('min')
    full_time_index = pd.date_range(start=min_time, end=max_time, freq='min')
    full_df = pd.DataFrame({'minute_bin': full_time_index})
    
    # Merge to fill gaps
    merged_df = pd.merge(full_df, agg_df, on='minute_bin', how='left')
    
    # Fill missing values
    merged_df['requests_per_minute'] = merged_df['requests_per_minute'].fillna(0).astype(int)
    merged_df['unique_ips'] = merged_df['unique_ips'].fillna(0).astype(int)
    merged_df['avg_response_time_ms'] = merged_df['avg_response_time_ms'].fillna(0.0)
    merged_df['total_bytes_sent'] = merged_df['total_bytes_sent'].fillna(0).astype(int)
    merged_df['error_rate'] = merged_df['error_rate'].fillna(0.0)
    
    # Add rolling features
    merged_df['rolling_mean'] = merged_df['requests_per_minute'].rolling(window=15, min_periods=1).mean()
    merged_df['rolling_std'] = merged_df['requests_per_minute'].rolling(window=15, min_periods=1).std().fillna(0.0)
    merged_df['traffic_volatility'] = merged_df['rolling_std']
    
    # Extract temporal features
    merged_df['hour'] = merged_df['minute_bin'].dt.hour
    merged_df['day_of_week'] = merged_df['minute_bin'].dt.dayofweek
    merged_df['is_weekend'] = merged_df['day_of_week'].isin([5, 6]).astype(int)
    
    # Classify traffic levels: operationally define peak traffic as >= 90th percentile
    # If p90 is 0 (due to sparse data), set threshold to 1 to separate active vs idle traffic
    p90 = np.percentile(merged_df['requests_per_minute'], 90)
    threshold = max(p90, 1.0)
    merged_df['traffic_level'] = np.where(merged_df['requests_per_minute'] >= threshold, 'peak', 'normal')
    
    return merged_df, threshold

def calculate_interarrival_times(df, timestamp_col):
    """
    Calculate the time difference between consecutive requests in seconds.
    This is useful for fitting exponential distributions.
    """
    timestamps = pd.to_datetime(df[timestamp_col]).sort_values()
    interarrival = timestamps.diff().dt.total_seconds().dropna()
    return interarrival
