import numpy as np
import pandas as pd
from scipy import stats

def detect_zscore_anomalies(series, threshold=3.0):
    """
    Detect anomalies using the Z-score method.
    """
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series(False, index=series.index)
    z_scores = (series - mean) / std
    return np.abs(z_scores) > threshold

def detect_iqr_anomalies(series):
    """
    Detect anomalies using the Interquartile Range (IQR) method.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    return series > upper_bound

def calculate_survival_probability(threshold, dist_name, params, was_shifted=False):
    """
    Calculate the probability P(X > threshold) under a given distribution and parameters.
    Handles shifted data for positive continuous distributions.
    """
    # Map friendly names to scipy names
    scipy_dist_map = {
        'poisson': 'poisson',
        'negative binomial': 'nbinom',
        'normal': 'norm',
        'exponential': 'expon',
        'gamma': 'gamma',
        'log-normal': 'lognorm',
        'weibull': 'weibull_min'
    }
    
    code = scipy_dist_map.get(dist_name.lower(), dist_name.lower())
    
    if code in ['poisson', 'nbinom']:
        # Discrete survival function: sf(k) = P(X > k)
        dist = getattr(stats, code)
        return dist.sf(threshold, *params)
    else:
        # Continuous survival function: sf(x) = P(X > x)
        dist = getattr(stats, code)
        val_to_eval = threshold + 1.0 if was_shifted else threshold
        return dist.sf(val_to_eval, *params)

def analyze_extreme_event_frequency(series, thresholds):
    """
    Compute empirical frequencies of exceeding specified thresholds.
    """
    n_samples = len(series)
    empirical_probs = {}
    for t in thresholds:
        count_exceed = np.sum(series > t)
        empirical_probs[t] = count_exceed / n_samples
    return empirical_probs
