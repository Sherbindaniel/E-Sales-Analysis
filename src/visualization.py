import os
import ast
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def _parse_params(raw):
    if isinstance(raw, (list, tuple)):
        return tuple(float(x) for x in raw)
    s = re.sub(r'np\.\w+\(([^)]+)\)', r'\1', str(raw))
    parsed = ast.literal_eval(s)
    if isinstance(parsed, (int, float)):
        return (float(parsed),)
    return tuple(float(x) for x in parsed)

# Set styling defaults
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#EEEEEE'
plt.rcParams['grid.linewidth'] = 0.5

def plot_traffic_over_time(df, output_path, name_prefix):
    """
    Plot aggregated traffic over time, highlighting peak traffic levels.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot normal and peak segments
    normal_df = df[df['traffic_level'] == 'normal']
    peak_df = df[df['traffic_level'] == 'peak']
    
    plt.plot(df['minute_bin'], df['requests_per_minute'], color='#5A9BD5', alpha=0.5, label='Requests/min (All)')
    plt.scatter(peak_df['minute_bin'], peak_df['requests_per_minute'], color='#E06666', s=12, label='Peak Traffic (>=90th Pctl)')
    
    plt.title(f"{name_prefix} - Web Traffic Over Time", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Timestamp", fontsize=11, labelpad=10)
    plt.ylabel("Requests per Minute", fontsize=11, labelpad=10)
    plt.legend(frameon=True, facecolor='white', edgecolor='#E5E5E5')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_rolling_statistics(df, output_path, name_prefix):
    """
    Plot requests per minute along with rolling mean and volatility (rolling std).
    """
    plt.figure(figsize=(12, 5))
    
    plt.plot(df['minute_bin'], df['requests_per_minute'], color='#E5E5E5', alpha=0.6, label='Raw Requests/min')
    plt.plot(df['minute_bin'], df['rolling_mean'], color='#4F81BD', linewidth=1.5, label='Rolling Mean (15m)')
    plt.plot(df['minute_bin'], df['rolling_std'], color='#F1A340', linewidth=1.2, linestyle='--', label='Rolling Volatility (15m)')
    
    plt.title(f"{name_prefix} - Rolling Traffic Mean and Volatility", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Timestamp", fontsize=11, labelpad=10)
    plt.ylabel("Requests per Minute", fontsize=11, labelpad=10)
    plt.legend(frameon=True, facecolor='white', edgecolor='#E5E5E5')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_distribution_fits(df, dist_results, output_path, name_prefix, regime='all'):
    """
    Plot the observed histogram of requests/minute overlaid with fitted distribution curves.
    """
    data = df['requests_per_minute'].values
    
    plt.figure(figsize=(10, 6))
    
    # Calculate appropriate bins
    max_val = int(np.max(data))
    bins = np.arange(-0.5, max_val + 1.5, 1) if max_val < 50 else 50
    
    # Plot empirical density
    # If the range of values is small, plot as probability mass (bar chart), else density histogram
    if max_val < 50:
        counts, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.bar(bin_centers, counts, width=0.8, color='#A5A5A5', alpha=0.5, edgecolor='#7F7F7F', label='Observed Data')
    else:
        plt.hist(data, bins=50, density=True, color='#A5A5A5', alpha=0.5, edgecolor='#7F7F7F', label='Observed Data')
        
    x_discrete = np.arange(0, max_val + 1)
    x_continuous = np.linspace(0.01, max_val, 500)
    
    # Plot up to 4 top distributions
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000', '#7030A0']
    color_idx = 0
    
    for _, row in dist_results.iterrows():
        dist_name = row['Distribution']
        params = _parse_params(row['Parameters'])
        is_discrete = row['Type'] == 'Discrete'
        
        try:
            if is_discrete:
                if dist_name == 'Poisson':
                    y = stats.poisson.pmf(x_discrete, *params)
                    plt.step(x_discrete, y, where='mid', color=colors[color_idx % len(colors)], linewidth=2, label=f"Poisson (fit)")
                elif dist_name == 'Negative Binomial':
                    y = stats.nbinom.pmf(x_discrete, *params)
                    plt.step(x_discrete, y, where='mid', color=colors[color_idx % len(colors)], linewidth=2, label=f"Neg Binomial (fit)")
            else:
                # Continuous distributions
                dist_code_map = {
                    'Normal': stats.norm,
                    'Exponential': stats.expon,
                    'Gamma': stats.gamma,
                    'Log-Normal': stats.lognorm,
                    'Weibull': stats.weibull_min
                }
                dist = dist_code_map.get(dist_name)
                
                # Check if data was shifted (non-positive data gets +1.0 for positive-only dists)
                needs_positive = dist_name in ['Gamma', 'Log-Normal', 'Weibull']
                was_shifted = needs_positive and (data <= 0).any()
                
                if was_shifted:
                    # Plot density shifted back
                    y = dist.pdf(x_continuous + 1.0, *params)
                else:
                    y = dist.pdf(x_continuous, *params)
                    
                plt.plot(x_continuous, y, color=colors[color_idx % len(colors)], linewidth=2, label=f"{dist_name} (fit)")
            color_idx += 1
        except Exception as e:
            print(f"Error plotting {dist_name} curve: {e}")
            
    title_suffix = f"({regime.capitalize()} Traffic)" if regime != 'all' else "(All Traffic)"
    plt.title(f"{name_prefix} - Observed Traffic vs. Fitted Distributions {title_suffix}", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Requests per Minute", fontsize=11)
    plt.ylabel("Probability Density/Mass", fontsize=11)
    plt.legend(frameon=True, facecolor='white', edgecolor='#E5E5E5')
    plt.xlim(left=-0.5, right=max_val * 1.05)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_qq_plots(df, dist_results, output_path, name_prefix):
    """
    Generate Q-Q plots for continuous distributions to check fit accuracy in details.
    """
    data = df['requests_per_minute'].values
    
    # Select continuous models
    cont_results = dist_results[dist_results['Type'] == 'Continuous']
    if cont_results.empty:
        return
        
    num_models = len(cont_results)
    cols = 2
    rows = (num_models + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()
    
    dist_code_map = {
        'Normal': ('norm', stats.norm),
        'Exponential': ('expon', stats.expon),
        'Gamma': ('gamma', stats.gamma),
        'Log-Normal': ('lognorm', stats.lognorm),
        'Weibull': ('weibull_min', stats.weibull_min)
    }
    
    for i, (_, row) in enumerate(cont_results.iterrows()):
        dist_name = row['Distribution']
        params = _parse_params(row['Parameters'])
        
        try:
            dist_code, dist = dist_code_map.get(dist_name)
            
            # Shift data if needed
            needs_positive = dist_name in ['Gamma', 'Log-Normal', 'Weibull']
            was_shifted = needs_positive and (data <= 0).any()
            data_fit = data + 1.0 if was_shifted else data
            
            # Generate QQ plot
            stats.probplot(data_fit, sparams=params, dist=dist_code, plot=axes[i])
            axes[i].set_title(f"{dist_name} Q-Q Plot", fontsize=11, fontweight='bold')
            axes[i].get_lines()[0].set_color('#5A9BD5')
            axes[i].get_lines()[0].set_marker('o')
            axes[i].get_lines()[0].set_markersize(3)
            axes[i].get_lines()[0].set_alpha(0.5)
            axes[i].get_lines()[1].set_color('#C00000')
            axes[i].get_lines()[1].set_linewidth(1.5)
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
            
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle(f"{name_prefix} - Probability Q-Q Plots", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_temporal_patterns(df, output_path, name_prefix):
    """
    Plot traffic distribution grouped by Hour of Day and Day of Week.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Hour of Day Boxplot
    df.boxplot(column='requests_per_minute', by='hour', ax=axes[0], grid=False,
               patch_artist=True, boxprops=dict(facecolor='#D9E1F2', color='#4F81BD'),
               whiskerprops=dict(color='#4F81BD'), medianprops=dict(color='#C00000', linewidth=1.5))
    axes[0].set_title("Traffic Volume by Hour of Day", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Requests/min")
    
    # 2. Day of Week Boxplot
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    df.boxplot(column='requests_per_minute', by='day_of_week', ax=axes[1], grid=False,
               patch_artist=True, boxprops=dict(facecolor='#FCE4D6', color='#ED7D31'),
               whiskerprops=dict(color='#ED7D31'), medianprops=dict(color='#C00000', linewidth=1.5))
    axes[1].set_title("Traffic Volume by Day of Week", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Day of Week")
    axes[1].set_xticklabels([days[i] for i in sorted(df['day_of_week'].unique())])
    axes[1].set_ylabel("Requests/min")
    
    plt.suptitle(f"{name_prefix} - Temporal Traffic Distributions", fontsize=13, fontweight='bold', y=0.98)
    # Remove pandas auto title
    fig.axes[0].set_title("Traffic Volume by Hour of Day")
    fig.axes[1].set_title("Traffic Volume by Day of Week")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
