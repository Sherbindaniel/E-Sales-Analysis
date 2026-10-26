import pandas as pd
import numpy as np
from scipy import stats

def calculate_summary_statistics(series):
    """
    Calculate comprehensive summary statistics for a given pandas series.
    """
    stats_dict = {
        'count': len(series),
        'mean': series.mean(),
        'median': series.median(),
        'mode': series.mode().iloc[0] if not series.empty and len(series.mode()) > 0 else np.nan,
        'variance': series.var(),
        'std_dev': series.std(),
        'min': series.min(),
        'max': series.max(),
        'p25': series.quantile(0.25),
        'p50': series.quantile(0.50),
        'p75': series.quantile(0.75),
        'p90': series.quantile(0.90),
        'p95': series.quantile(0.95),
        'p99': series.quantile(0.99),
        'skewness': series.skew(),
        'kurtosis': series.kurtosis(),
        'dispersion_index': series.var() / series.mean() if series.mean() != 0 else np.nan
    }
    return stats_dict

def compare_regimes_statistics(df, val_col, group_col='traffic_level'):
    """
    Compare summary statistics of a variable across normal and peak regimes.
    """
    groups = df.groupby(group_col)[val_col]
    comparison = {}
    
    # Calculate for each group
    for name, group in groups:
        comparison[name] = calculate_summary_statistics(group)
        
    # Calculate for overall
    comparison['overall'] = calculate_summary_statistics(df[val_col])
    
    return pd.DataFrame(comparison)

def analyze_correlations(df, numeric_cols):
    """
    Calculate the correlation matrix for specified numeric columns.
    """
    # Select only columns present in the df
    cols = [c for c in numeric_cols if c in df.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    return df[cols].corr()
