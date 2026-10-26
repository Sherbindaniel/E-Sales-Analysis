import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

def check_overdispersion(data):
    """
    Check if the count data exhibits overdispersion.
    Returns:
    - mean: Sample mean
    - var: Sample variance
    - dispersion: var / mean
    - is_overdispersed: True if dispersion > 1.2 (heuristic)
    """
    mean = np.mean(data)
    var = np.var(data)
    dispersion = var / mean if mean != 0 else np.nan
    is_overdispersed = dispersion > 1.2 if not np.isnan(dispersion) else False
    return mean, var, dispersion, is_overdispersed

def fit_poisson(data):
    """
    Fit a Poisson distribution using MLE.
    The MLE for lambda (mu) is the sample mean.
    Returns parameters: (mu,)
    """
    mu = float(np.mean(data))
    if mu <= 0:
        mu = 1e-5
    return (mu,)

def fit_nbinom(data):
    """
    Fit a Negative Binomial distribution using MLE.
    Uses Method of Moments to initialize, then minimizes the negative log-likelihood.
    Returns parameters: (n, p)
    """
    mean = np.mean(data)
    var = np.var(data)
    
    # Method of moments initialization
    if var <= mean:
        # Fallback for under/equidispersed data
        p_init = 0.99
        n_init = mean * p_init / (1.0 - p_init)
    else:
        p_init = mean / var
        n_init = (mean ** 2) / (var - mean)
        
    # Negative Log-Likelihood function to minimize
    def nll(params):
        n_val, p_val = params
        if n_val <= 0 or p_val <= 0 or p_val >= 1:
            return 1e10
        # stats.nbinom uses n (number of successes) and p (probability of success)
        # scipy pmf: stats.nbinom.pmf(k, n, p)
        logpmf = stats.nbinom.logpmf(data, n_val, p_val)
        # Handle logpmf returning -inf or nan
        logpmf = np.where(np.isinf(logpmf) | np.isnan(logpmf), -1e5, logpmf)
        return -np.sum(logpmf)
        
    res = minimize(nll, [n_init, p_init], method='Nelder-Mead', bounds=[(1e-5, None), (1e-5, 0.99999)])
    if res.success:
        return (float(res.x[0]), float(res.x[1]))
    return (float(n_init), float(p_init))

def fit_continuous(data, dist_name):
    """
    Fit a continuous distribution to the data.
    Distributions supported: 'norm', 'expon', 'gamma', 'lognorm', 'weibull_min'
    """
    # Shift data by 1 if there are any non-positive values and we're fitting a distribution
    # that requires positive values (gamma, lognorm, weibull_min)
    needs_positive = dist_name in ['gamma', 'lognorm', 'weibull_min']
    if needs_positive and (data <= 0).any():
        data_to_fit = data + 1.0
    else:
        data_to_fit = data
        
    dist = getattr(stats, dist_name)
    params = dist.fit(data_to_fit)
    return tuple(float(p) for p in params)

def calculate_log_likelihood(data, dist_name, params, is_discrete=False):
    """
    Calculate the log-likelihood of the data given a distribution and its parameters.
    """
    if is_discrete:
        if dist_name == 'poisson':
            log_lik = np.sum(stats.poisson.logpmf(data, *params))
        elif dist_name == 'nbinom':
            log_lik = np.sum(stats.nbinom.logpmf(data, *params))
        else:
            raise ValueError(f"Unknown discrete distribution: {dist_name}")
    else:
        # Shift data if needed
        needs_positive = dist_name in ['gamma', 'lognorm', 'weibull_min']
        if needs_positive and (data <= 0).any():
            data_to_fit = data + 1.0
        else:
            data_to_fit = data
            
        dist = getattr(stats, dist_name)
        log_lik = np.sum(dist.logpdf(data_to_fit, *params))
        
    return log_lik

def calculate_ic(log_lik, n_params, n_samples):
    """
    Calculate AIC and BIC.
    """
    if np.isinf(log_lik) or np.isnan(log_lik):
        return np.nan, np.nan
    aic = 2 * n_params - 2 * log_lik
    bic = n_params * np.log(n_samples) - 2 * log_lik
    return aic, bic

def chi_square_gof(data, dist_name, params):
    """
    Perform a Chi-Square Goodness-of-Fit test for discrete count distributions.
    Bins the counts dynamically to ensure expected counts are >= 5.
    """
    unique_vals, obs_freqs = np.unique(data, return_counts=True)
    obs_dict = dict(zip(unique_vals, obs_freqs))
    
    max_val = int(np.max(data))
    n_samples = len(data)
    
    observed_bins = []
    expected_bins = []
    
    current_obs = 0
    current_exp = 0.0
    
    if dist_name == 'poisson':
        mu = params[0]
        cdf_fn = lambda x: stats.poisson.cdf(x, mu)
        pmf_fn = lambda x: stats.poisson.pmf(x, mu)
    elif dist_name == 'nbinom':
        n_param, p_param = params
        cdf_fn = lambda x: stats.nbinom.cdf(x, n_param, p_param)
        pmf_fn = lambda x: stats.nbinom.pmf(x, n_param, p_param)
    else:
        raise ValueError(f"Chi-square test only supported for discrete distributions, got {dist_name}")
        
    # Iterate counts and bin them
    for x in range(max_val + 1):
        current_obs += obs_dict.get(x, 0)
        
        if x == max_val:
            prob = 1.0 - cdf_fn(x - 1) if x > 0 else 1.0
        else:
            prob = pmf_fn(x)
            
        current_exp += prob * n_samples
        
        # Close bin if expected frequency >= 5, or at end
        if current_exp >= 5 or x == max_val:
            observed_bins.append(current_obs)
            expected_bins.append(current_exp)
            current_obs = 0
            current_exp = 0.0
            
    # Merge backward if last bin has expected frequency < 5
    if len(expected_bins) > 1 and expected_bins[-1] < 5:
        expected_bins[-2] += expected_bins[-1]
        observed_bins[-2] += observed_bins[-1]
        expected_bins.pop()
        observed_bins.pop()
        
    obs = np.array(observed_bins)
    exp = np.array(expected_bins)
    
    n_params = len(params)
    dof = len(obs) - 1 - n_params
    
    if dof <= 0:
        return np.nan, np.nan, np.nan
        
    # Compute Chi-square statistic and p-value
    chi_stat, p_val = stats.chisquare(f_obs=obs, f_exp=exp, ddof=n_params)
    return chi_stat, p_val, dof

def ks_gof(data, dist_name, params):
    """
    Perform a Kolmogorov-Smirnov test for continuous distributions.
    """
    needs_positive = dist_name in ['gamma', 'lognorm', 'weibull_min']
    if needs_positive and (data <= 0).any():
        data_to_fit = data + 1.0
    else:
        data_to_fit = data
        
    dist = getattr(stats, dist_name)
    
    # We define the CDF function with parameters unpacked
    cdf_fn = lambda x: dist.cdf(x, *params)
    
    ks_stat, p_val = stats.kstest(data_to_fit, cdf_fn)
    return ks_stat, p_val

def run_distribution_fitting(data, name_prefix):
    """
    Fit discrete and continuous distributions and compare GoF.
    Returns a pandas DataFrame comparison.
    """
    n_samples = len(data)
    if n_samples == 0:
        return pd.DataFrame(columns=[
            'Dataset', 'Distribution', 'Type', 'Parameters', 'Log-Likelihood', 
            'AIC', 'BIC', 'GoF Test', 'GoF Stat', 'p-value', 'DoF'
        ])
    results = []
    
    # 1. Discrete Models
    # Poisson
    poi_params = fit_poisson(data)
    poi_ll = calculate_log_likelihood(data, 'poisson', poi_params, is_discrete=True)
    poi_aic, poi_bic = calculate_ic(poi_ll, 1, n_samples)
    poi_chi, poi_p, poi_dof = chi_square_gof(data, 'poisson', poi_params)
    results.append({
        'Dataset': name_prefix,
        'Distribution': 'Poisson',
        'Type': 'Discrete',
        'Parameters': str(poi_params),
        'Log-Likelihood': poi_ll,
        'AIC': poi_aic,
        'BIC': poi_bic,
        'GoF Test': 'Chi-Square',
        'GoF Stat': poi_chi,
        'p-value': poi_p,
        'DoF': poi_dof
    })
    
    # Negative Binomial
    nb_params = fit_nbinom(data)
    nb_ll = calculate_log_likelihood(data, 'nbinom', nb_params, is_discrete=True)
    nb_aic, nb_bic = calculate_ic(nb_ll, 2, n_samples)
    nb_chi, nb_p, nb_dof = chi_square_gof(data, 'nbinom', nb_params)
    results.append({
        'Dataset': name_prefix,
        'Distribution': 'Negative Binomial',
        'Type': 'Discrete',
        'Parameters': str(nb_params),
        'Log-Likelihood': nb_ll,
        'AIC': nb_aic,
        'BIC': nb_bic,
        'GoF Test': 'Chi-Square',
        'GoF Stat': nb_chi,
        'p-value': nb_p,
        'DoF': nb_dof
    })
    
    # 2. Continuous Models (treated as continuous)
    continuous_dists = {
        'norm': ('Normal', 2),
        'expon': ('Exponential', 2),
        'gamma': ('Gamma', 3),
        'lognorm': ('Log-Normal', 3),
        'weibull_min': ('Weibull', 3)
    }
    
    for dist_code, (dist_name, n_params) in continuous_dists.items():
        try:
            params = fit_continuous(data, dist_code)
            ll = calculate_log_likelihood(data, dist_code, params, is_discrete=False)
            aic, bic = calculate_ic(ll, n_params, n_samples)
            ks_stat, ks_p = ks_gof(data, dist_code, params)
            
            results.append({
                'Dataset': name_prefix,
                'Distribution': dist_name,
                'Type': 'Continuous',
                'Parameters': str(params),
                'Log-Likelihood': ll,
                'AIC': aic,
                'BIC': bic,
                'GoF Test': 'Kolmogorov-Smirnov',
                'GoF Stat': ks_stat,
                'p-value': ks_p,
                'DoF': np.nan
            })
        except Exception as e:
            print(f"Error fitting {dist_name} on {name_prefix}: {e}")
            
    return pd.DataFrame(results)
