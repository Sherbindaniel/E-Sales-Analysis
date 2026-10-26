import numpy as np
import pandas as pd
from scipy import stats


def compute_capacity_report(df_min, server_capacity_per_unit, num_servers,
                             dist_fit_df, lognorm_params, nb_params):
    """
    Core SRE capacity planning report.

    Parameters
    ----------
    df_min               : aggregated 1-min DataFrame with 'requests_per_minute'
    server_capacity_per_unit : int  — max req/min a single server handles
    num_servers          : int  — current number of servers
    dist_fit_df          : distribution fit results DataFrame (for best model)
    lognorm_params       : fitted Log-Normal params tuple
    nb_params            : fitted Negative Binomial params tuple

    Returns
    -------
    dict with all capacity planning fields
    """
    rpm = df_min['requests_per_minute']
    total_capacity = server_capacity_per_unit * num_servers
    n = len(rpm)

    # ── Basic traffic stats ──────────────────────────────────────
    mean_rpm   = float(rpm.mean())
    median_rpm = float(rpm.median())
    p90_rpm    = float(np.percentile(rpm, 90))
    p95_rpm    = float(np.percentile(rpm, 95))
    p99_rpm    = float(np.percentile(rpm, 99))
    max_rpm    = float(rpm.max())

    # ── Utilization at different percentiles ─────────────────────
    util_mean = mean_rpm / total_capacity * 100
    util_p90  = p90_rpm  / total_capacity * 100
    util_p95  = p95_rpm  / total_capacity * 100
    util_p99  = p99_rpm  / total_capacity * 100
    util_max  = max_rpm  / total_capacity * 100

    # ── Bottleneck range ─────────────────────────────────────────
    # Bottleneck starts when utilization > 70% (industry standard warning threshold)
    # and becomes critical at > 90%
    bottleneck_warning_rpm  = total_capacity * 0.70
    bottleneck_critical_rpm = total_capacity * 0.90

    # ── Overload frequency (empirical) ───────────────────────────
    overload_count    = int((rpm > total_capacity).sum())
    overload_freq_pct = overload_count / n * 100
    # Mean time between overloads in minutes
    if overload_count > 0:
        mtbo_minutes = n / overload_count
        mtbo_hours   = mtbo_minutes / 60
    else:
        mtbo_minutes = None
        mtbo_hours   = None

    # ── Uptime estimate ──────────────────────────────────────────
    # Minutes where traffic <= total_capacity = "healthy" minutes
    healthy_minutes = int((rpm <= total_capacity).sum())
    uptime_pct = healthy_minutes / n * 100

    # ── Model-based survival probability at total_capacity ───────
    # Using Log-Normal (best overall fit)
    try:
        from scipy.stats import lognorm as _lognorm
        p_exceed_lognorm = float(_lognorm.sf(total_capacity + 1.0, *lognorm_params))
    except Exception:
        p_exceed_lognorm = None

    # Using Negative Binomial
    try:
        from scipy.stats import nbinom as _nbinom
        p_exceed_nb = float(_nbinom.sf(total_capacity, *nb_params))
    except Exception:
        p_exceed_nb = None

    # ── Recommended server count ─────────────────────────────────
    # To handle p99 traffic with 20% headroom
    recommended_for_p99 = int(np.ceil(p99_rpm * 1.20 / server_capacity_per_unit))
    # To handle max observed traffic with 20% headroom
    recommended_for_max = int(np.ceil(max_rpm * 1.20 / server_capacity_per_unit))

    # ── Risk level ───────────────────────────────────────────────
    if util_p99 > 100:
        risk_level = 'CRITICAL'
        risk_desc  = 'Current capacity is insufficient for p99 traffic. Overloads are frequent.'
    elif util_p95 > 90:
        risk_level = 'HIGH'
        risk_desc  = 'Capacity is strained at p95. Expect degradation during traffic spikes.'
    elif util_p90 > 80:
        risk_level = 'MEDIUM'
        risk_desc  = 'Capacity is adequate for normal traffic but tight during peak periods.'
    elif util_p90 > 60:
        risk_level = 'LOW'
        risk_desc  = 'Capacity is healthy. Monitor p99 events and plan for growth.'
    else:
        risk_level = 'SAFE'
        risk_desc  = 'Capacity is well-provisioned. No immediate action required.'

    # ── Measures / Recommendations ───────────────────────────────
    measures = _generate_measures(
        risk_level, util_p90, util_p99, util_max,
        overload_count, recommended_for_p99, num_servers,
        bottleneck_warning_rpm, bottleneck_critical_rpm,
        p99_rpm, total_capacity
    )

    # ── Percentile breakdown table ────────────────────────────────
    percentile_table = []
    for pct_label, val in [('P50 (Median)', median_rpm), ('P90', p90_rpm),
                            ('P95', p95_rpm), ('P99', p99_rpm), ('Max', max_rpm)]:
        util = val / total_capacity * 100
        status = 'Normal' if util < 70 else ('Warning' if util < 90 else 'Critical')
        percentile_table.append({
            'Percentile': pct_label,
            'Traffic (req/min)': round(val, 1),
            'Utilization %': round(util, 1),
            'Status': status,
            'Headroom (req/min)': round(max(0, total_capacity - val), 1)
        })

    return {
        'total_capacity':           total_capacity,
        'mean_rpm':                 round(mean_rpm, 2),
        'p90_rpm':                  round(p90_rpm, 2),
        'p95_rpm':                  round(p95_rpm, 2),
        'p99_rpm':                  round(p99_rpm, 2),
        'max_rpm':                  round(max_rpm, 2),
        'util_mean':                round(util_mean, 2),
        'util_p90':                 round(util_p90, 2),
        'util_p95':                 round(util_p95, 2),
        'util_p99':                 round(util_p99, 2),
        'util_max':                 round(util_max, 2),
        'bottleneck_warning_rpm':   round(bottleneck_warning_rpm, 1),
        'bottleneck_critical_rpm':  round(bottleneck_critical_rpm, 1),
        'overload_count':           overload_count,
        'overload_freq_pct':        round(overload_freq_pct, 4),
        'mtbo_minutes':             round(mtbo_minutes, 1) if mtbo_minutes else None,
        'mtbo_hours':               round(mtbo_hours, 2)   if mtbo_hours   else None,
        'uptime_pct':               round(uptime_pct, 4),
        'p_exceed_lognorm':         p_exceed_lognorm,
        'p_exceed_nb':              p_exceed_nb,
        'recommended_for_p99':      recommended_for_p99,
        'recommended_for_max':      recommended_for_max,
        'risk_level':               risk_level,
        'risk_desc':                risk_desc,
        'measures':                 measures,
        'percentile_table':         percentile_table,
        'server_capacity_per_unit': server_capacity_per_unit,
        'num_servers':              num_servers,
    }


def _generate_measures(risk_level, util_p90, util_p99, util_max,
                        overload_count, recommended_for_p99, num_servers,
                        warn_rpm, crit_rpm, p99_rpm, total_capacity):
    measures = []

    if risk_level == 'CRITICAL':
        measures.append({
            'priority': 'IMMEDIATE',
            'action': f'Scale up to at least {recommended_for_p99} servers immediately.',
            'reason': f'p99 traffic exceeds total capacity. Overloads occurred {overload_count} times.'
        })
        measures.append({
            'priority': 'IMMEDIATE',
            'action': 'Enable auto-scaling with trigger at 70% utilization.',
            'reason': 'Manual scaling cannot respond fast enough to flash-sale spikes.'
        })
        measures.append({
            'priority': 'SHORT-TERM',
            'action': 'Implement request queuing and graceful degradation.',
            'reason': 'Prevents hard failures when capacity is exceeded.'
        })

    elif risk_level == 'HIGH':
        measures.append({
            'priority': 'URGENT',
            'action': f'Add {recommended_for_p99 - num_servers} servers to reach {recommended_for_p99} total.',
            'reason': f'p95 utilization is {util_p99:.1f}%. Headroom is insufficient for spikes.'
        })
        measures.append({
            'priority': 'SHORT-TERM',
            'action': 'Set auto-scaling trigger at 65% utilization with 3-minute cooldown.',
            'reason': 'Provides buffer time before capacity is exhausted.'
        })
        measures.append({
            'priority': 'SHORT-TERM',
            'action': 'Configure CDN caching for static assets to reduce origin load.',
            'reason': 'Can reduce origin traffic by 30–60% for e-commerce workloads.'
        })

    elif risk_level == 'MEDIUM':
        measures.append({
            'priority': 'PLANNED',
            'action': f'Plan capacity expansion to {recommended_for_p99} servers before next peak event.',
            'reason': f'p90 utilization is {util_p90:.1f}%. Adequate now but insufficient for growth.'
        })
        measures.append({
            'priority': 'PLANNED',
            'action': f'Set alerting threshold at {warn_rpm:.0f} req/min (70% capacity).',
            'reason': 'Early warning before reaching critical utilization.'
        })
        measures.append({
            'priority': 'MONITORING',
            'action': 'Track rolling 15-minute traffic volatility. Alert if std dev > 2× baseline.',
            'reason': 'High volatility precedes traffic spikes by 5–10 minutes.'
        })

    else:
        measures.append({
            'priority': 'MONITORING',
            'action': f'Maintain alerting at {warn_rpm:.0f} req/min (70% capacity).',
            'reason': 'Current capacity is healthy. Proactive monitoring prevents surprises.'
        })
        measures.append({
            'priority': 'PLANNED',
            'action': 'Review capacity quarterly against traffic growth trend.',
            'reason': 'Prevents reactive scaling during high-growth periods.'
        })

    # Universal measures
    measures.append({
        'priority': 'BEST PRACTICE',
        'action': f'Never provision below p99 traffic ({p99_rpm:.0f} req/min). Use Log-Normal model for planning.',
        'reason': 'Poisson model underestimates tail risk by orders of magnitude for overdispersed traffic.'
    })
    measures.append({
        'priority': 'BEST PRACTICE',
        'action': f'Set hard rate limit at {crit_rpm:.0f} req/min (90% capacity) with 429 responses.',
        'reason': 'Controlled degradation is better than uncontrolled failure.'
    })

    return measures


def get_anomaly_log(df_min, timestamp_col='minute_bin'):
    """
    Return a detailed table of anomalous minutes with timestamps, severity, and z-scores.
    """
    rpm = df_min['requests_per_minute']
    mean = rpm.mean()
    std  = rpm.std()

    if std == 0:
        return []

    z_scores = (rpm - mean) / std
    q3  = rpm.quantile(0.75)
    iqr = rpm.quantile(0.75) - rpm.quantile(0.25)
    iqr_bound = q3 + 1.5 * iqr

    anomaly_mask = (np.abs(z_scores) > 3) | (rpm > iqr_bound)
    anomaly_df = df_min[anomaly_mask].copy()

    if anomaly_df.empty:
        return []

    anomaly_df = anomaly_df.sort_values('requests_per_minute', ascending=False)

    rows = []
    for _, row in anomaly_df.head(50).iterrows():
        z = float(z_scores[row.name])
        rpm_val = int(row['requests_per_minute'])

        if abs(z) > 6 or rpm_val > mean + 6 * std:
            severity = 'EXTREME'
        elif abs(z) > 4:
            severity = 'SEVERE'
        else:
            severity = 'MODERATE'

        rows.append({
            'Timestamp':        str(row[timestamp_col])[:16],
            'Requests/min':     rpm_val,
            'Z-Score':          round(z, 2),
            'vs Mean':          f'+{rpm_val - mean:.0f}' if rpm_val > mean else f'{rpm_val - mean:.0f}',
            'Severity':         severity,
            'IQR Outlier':      'Yes' if rpm_val > iqr_bound else 'No',
        })

    return rows


def get_plain_english_verdicts(dist_fit_df, dispersion_rows):
    """
    Generate plain-English interpretation for each distribution fit and dispersion result.
    """
    verdicts = {}

    # Dispersion verdict
    for row in dispersion_rows:
        name = row['Dataset']
        d    = row.get('Dispersion Index')
        if d is None:
            verdicts[f'disp_{name}'] = 'Insufficient data to assess dispersion.'
            continue
        if d > 5:
            verdicts[f'disp_{name}'] = (
                f'Severely overdispersed (D={d:.2f}). Poisson is invalid. '
                f'Use Negative Binomial or Log-Normal for all capacity calculations.'
            )
        elif d > 1.2:
            verdicts[f'disp_{name}'] = (
                f'Overdispersed (D={d:.2f}). Poisson underestimates variance. '
                f'Negative Binomial is the appropriate count model.'
            )
        else:
            verdicts[f'disp_{name}'] = (
                f'Approximately equidispersed (D={d:.2f}). '
                f'Poisson may be acceptable, but verify with GoF test.'
            )

    # Distribution fit verdicts
    if dist_fit_df is not None and not dist_fit_df.empty:
        aic_vals = dist_fit_df['AIC'].dropna()
        if not aic_vals.empty:
            best_idx  = aic_vals.idxmin()
            worst_idx = aic_vals.idxmax()
            best_name  = dist_fit_df.loc[best_idx,  'Distribution']
            worst_name = dist_fit_df.loc[worst_idx, 'Distribution']

            for _, row in dist_fit_df.iterrows():
                dist = row['Distribution']
                aic  = row.get('AIC')
                pval = row.get('p-value')

                if aic is None:
                    verdicts[f'dist_{dist}'] = 'Could not be fitted on this data.'
                    continue

                delta_aic = float(aic) - float(aic_vals.min()) if aic_vals.min() else 0

                if dist == best_name:
                    verdicts[f'dist_{dist}'] = (
                        f'BEST FIT — Use this model for capacity planning and survival probability estimates.'
                    )
                elif dist == 'Poisson' and delta_aic > 10000:
                    verdicts[f'dist_{dist}'] = (
                        f'REJECTED — AIC is {delta_aic:,.0f} worse than best model. '
                        f'Poisson catastrophically underestimates tail risk. Never use for provisioning.'
                    )
                elif delta_aic < 200:
                    verdicts[f'dist_{dist}'] = (
                        f'COMPETITIVE — Within {delta_aic:.0f} AIC of best model. '
                        f'Acceptable alternative for sensitivity analysis.'
                    )
                elif delta_aic < 2000:
                    verdicts[f'dist_{dist}'] = (
                        f'POOR FIT — {delta_aic:.0f} AIC worse than best. '
                        f'Do not use for tail risk estimation.'
                    )
                else:
                    verdicts[f'dist_{dist}'] = (
                        f'REJECTED — {delta_aic:,.0f} AIC worse than best model. '
                        f'Significantly misrepresents the traffic distribution.'
                    )

    return verdicts
