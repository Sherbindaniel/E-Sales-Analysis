import os
import sys
import io
import csv
import base64
import json
import traceback
import threading
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, request, render_template, Response, stream_with_context, make_response

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_raw_ecommerce_data, load_raw_web_server_data
from preprocessing import run_preprocessing_pipeline
from feature_engineering import aggregate_ecommerce_to_minutes, aggregate_web_server_to_minutes
from eda import compare_regimes_statistics, analyze_correlations
from distributions import check_overdispersion, run_distribution_fitting
from anomaly_detection import (
    detect_zscore_anomalies, detect_iqr_anomalies,
    calculate_survival_probability, analyze_extreme_event_frequency
)
from visualization import (
    plot_traffic_over_time, plot_rolling_statistics,
    plot_distribution_fits, plot_qq_plots, plot_temporal_patterns
)
from capacity import compute_capacity_report, get_anomaly_log, get_plain_english_verdicts

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory cache for last run results (for export)
_last_results = {}


# ── Helpers ──────────────────────────────────────────────────────────────────

def df_to_records(df):
    return json.loads(df.to_json(orient='records', default_handler=str))


def safe_float(val):
    if val is None:
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    except Exception:
        return str(val)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def safe_json(obj):
    return json.dumps(obj, cls=NumpyEncoder)


def capture_figure(plot_fn, leading_args, trailing_args):
    tmp = os.path.join(UPLOAD_FOLDER, f'_tmp_{threading.get_ident()}.png')
    try:
        plot_fn(*leading_args, tmp, *trailing_args)
        with open(tmp, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run_full_pipeline(df_ec_raw, df_ws_raw, df_servers,
                      server_capacity, num_servers, dataset_choice,
                      progress_cb):
    results = {}

    # ── Stage 1: Preprocessing ───────────────────────────────────
    progress_cb(4, 'Preprocessing raw data…')
    df_ec, df_ws = run_preprocessing_pipeline(df_ec_raw, df_ws_raw, df_servers)
    results['ingestion'] = {
        'ec_rows': len(df_ec), 'ws_rows': len(df_ws),
        'ec_cols': df_ec.shape[1], 'ws_cols': df_ws.shape[1],
    }

    # ── Stage 2: Aggregation ─────────────────────────────────────
    progress_cb(10, 'Aggregating to 1-minute intervals…')
    df_ec_min, p90_ec = aggregate_ecommerce_to_minutes(df_ec)
    df_ws_min, p90_ws = aggregate_web_server_to_minutes(df_ws)
    results['aggregation'] = {
        'ec_minutes': len(df_ec_min),
        'ec_req_min': int(df_ec_min['requests_per_minute'].min()),
        'ec_req_max': int(df_ec_min['requests_per_minute'].max()),
        'ec_p90': float(p90_ec),
        'ws_minutes': len(df_ws_min),
        'ws_req_min': int(df_ws_min['requests_per_minute'].min()),
        'ws_req_max': int(df_ws_min['requests_per_minute'].max()),
        'ws_p90': float(p90_ws),
    }

    # Choose primary dataset for capacity analysis
    df_primary = df_ec_min if dataset_choice in ('ec', 'both') else df_ws_min
    primary_label = 'E-Commerce' if dataset_choice in ('ec', 'both') else 'Web Server'

    # ── Stage 3: EDA ─────────────────────────────────────────────
    progress_cb(18, 'Running exploratory data analysis…')
    ec_regimes = compare_regimes_statistics(df_ec_min, 'requests_per_minute')
    ws_regimes = compare_regimes_statistics(df_ws_min, 'requests_per_minute')

    ec_num_cols = ['requests_per_minute', 'unique_ips', 'total_sales',
                   'avg_duration_sec', 'total_bytes', 'rolling_std']
    ws_num_cols = ['requests_per_minute', 'unique_ips', 'avg_response_time_ms',
                   'total_bytes_sent', 'error_rate', 'rolling_std']
    ec_corr = analyze_correlations(df_ec_min, ec_num_cols)
    ws_corr = analyze_correlations(df_ws_min, ws_num_cols)

    results['eda'] = {
        'ec_regimes': df_to_records(ec_regimes.reset_index().rename(columns={'index': 'Metric'})),
        'ws_regimes': df_to_records(ws_regimes.reset_index().rename(columns={'index': 'Metric'})),
        'ec_corr': df_to_records(ec_corr.reset_index().rename(columns={'index': 'Feature'})),
        'ws_corr': df_to_records(ws_corr.reset_index().rename(columns={'index': 'Feature'})),
        'ec_corr_cols': ['Feature'] + list(ec_corr.columns),
        'ws_corr_cols': ['Feature'] + list(ws_corr.columns),
    }

    # ── Stage 4: Dispersion ──────────────────────────────────────
    progress_cb(26, 'Checking overdispersion…')
    dispersion_rows = []
    for name, s in [
        ('E-Commerce All',    df_ec_min['requests_per_minute']),
        ('E-Commerce Normal', df_ec_min[df_ec_min['traffic_level'] == 'normal']['requests_per_minute']),
        ('E-Commerce Peak',   df_ec_min[df_ec_min['traffic_level'] == 'peak']['requests_per_minute']),
        ('Web Server All',    df_ws_min['requests_per_minute']),
        ('Web Server Normal', df_ws_min[df_ws_min['traffic_level'] == 'normal']['requests_per_minute']),
        ('Web Server Peak',   df_ws_min[df_ws_min['traffic_level'] == 'peak']['requests_per_minute']),
    ]:
        mean, var, disp, is_od = check_overdispersion(s.values)
        dispersion_rows.append({
            'Dataset': name, 'Mean': safe_float(mean),
            'Variance': safe_float(var), 'Dispersion Index': safe_float(disp),
            'Overdispersed': bool(is_od),
        })
    results['dispersion'] = dispersion_rows

    # ── Stage 5: Distribution Fitting ────────────────────────────
    progress_cb(34, 'Fitting distributions — E-Commerce All…')
    df_ec_fit_all = run_distribution_fitting(df_ec_min['requests_per_minute'].values, 'E-Commerce All')

    progress_cb(40, 'Fitting distributions — E-Commerce Normal…')
    df_ec_fit_normal = run_distribution_fitting(
        df_ec_min[df_ec_min['traffic_level'] == 'normal']['requests_per_minute'].values, 'E-Commerce Normal')

    progress_cb(46, 'Fitting distributions — E-Commerce Peak…')
    df_ec_fit_peak = run_distribution_fitting(
        df_ec_min[df_ec_min['traffic_level'] == 'peak']['requests_per_minute'].values, 'E-Commerce Peak')

    progress_cb(52, 'Fitting distributions — Web Server All…')
    df_ws_fit_all = run_distribution_fitting(df_ws_min['requests_per_minute'].values, 'Web Server All')

    progress_cb(57, 'Fitting distributions — Web Server Normal…')
    df_ws_fit_normal = run_distribution_fitting(
        df_ws_min[df_ws_min['traffic_level'] == 'normal']['requests_per_minute'].values, 'Web Server Normal')

    progress_cb(62, 'Fitting distributions — Web Server Peak…')
    df_ws_fit_peak = run_distribution_fitting(
        df_ws_min[df_ws_min['traffic_level'] == 'peak']['requests_per_minute'].values, 'Web Server Peak')

    def clean_fit(df):
        d = df.copy()
        for col in ['Log-Likelihood', 'AIC', 'BIC', 'GoF Stat', 'p-value']:
            if col in d.columns:
                d[col] = d[col].apply(safe_float)
        return df_to_records(d)

    results['distributions'] = {
        'ec_all': clean_fit(df_ec_fit_all),   'ec_normal': clean_fit(df_ec_fit_normal),
        'ec_peak': clean_fit(df_ec_fit_peak), 'ws_all': clean_fit(df_ws_fit_all),
        'ws_normal': clean_fit(df_ws_fit_normal), 'ws_peak': clean_fit(df_ws_fit_peak),
    }

    # ── Stage 6: Anomaly Detection ───────────────────────────────
    progress_cb(66, 'Running anomaly detection…')
    anomaly_summary = []
    for label, s in [
        ('E-Commerce Requests/min', df_ec_min['requests_per_minute']),
        ('Web Server Requests/min', df_ws_min['requests_per_minute']),
    ]:
        z   = detect_zscore_anomalies(s)
        iqr = detect_iqr_anomalies(s)
        anomaly_summary.append({
            'Dataset': label, 'Total Points': len(s),
            'Z-Score Anomalies': int(z.sum()), 'Z-Score %': round(z.sum() / len(s) * 100, 3),
            'IQR Anomalies': int(iqr.sum()), 'IQR %': round(iqr.sum() / len(s) * 100, 3),
        })

    # Detailed anomaly log for primary dataset
    anomaly_log = get_anomaly_log(df_primary)
    results['anomalies'] = {'summary': anomaly_summary, 'log': anomaly_log}

    # ── Stage 7: Survival Probabilities ──────────────────────────
    progress_cb(70, 'Computing survival probabilities…')
    primary_fit = df_ec_fit_all if dataset_choice in ('ec', 'both') else df_ws_fit_all

    best_all_row  = primary_fit.loc[primary_fit['AIC'].dropna().idxmin()]
    best_peak_row = (df_ec_fit_peak if dataset_choice in ('ec', 'both') else df_ws_fit_peak)
    best_peak_row = best_peak_row.loc[best_peak_row['AIC'].dropna().idxmin()]

    thresholds = [40, 60, 80, 100, 150, 200]
    emp_probs  = analyze_extreme_event_frequency(df_primary['requests_per_minute'], thresholds)

    def parse_params(fit_df, dist_name):
        row = fit_df[fit_df['Distribution'] == dist_name]
        if row.empty:
            return None
        raw = row['Parameters'].values[0]
        if isinstance(raw, (list, tuple)):
            return tuple(float(x) for x in raw)
        import ast, re
        s = str(raw)
        # Strip numpy type wrappers e.g. np.float64(1.23) -> 1.23
        s = re.sub(r'np\.\w+\(([^)]+)\)', r'\1', s)
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (int, float)):
            return (float(parsed),)
        return tuple(float(x) for x in parsed)

    poi_params     = parse_params(primary_fit, 'Poisson')
    nb_params      = parse_params(primary_fit, 'Negative Binomial')
    lognorm_params = parse_params(primary_fit, 'Log-Normal')

    survival_rows = []
    for t in thresholds:
        survival_rows.append({
            'Threshold (req/min)': t,
            'Empirical Frequency': safe_float(emp_probs[t]),
            'Poisson P(X>T)':      safe_float(calculate_survival_probability(t, 'poisson', poi_params)),
            'Neg-Binomial P(X>T)': safe_float(calculate_survival_probability(t, 'negative binomial', nb_params)),
            'Log-Normal P(X>T)':   safe_float(calculate_survival_probability(t, 'log-normal', lognorm_params, was_shifted=True)),
        })

    results['survival'] = {
        'rows': survival_rows,
        'best_all':      str(best_all_row['Distribution']),
        'best_all_aic':  safe_float(best_all_row['AIC']),
        'best_peak':     str(best_peak_row['Distribution']),
        'best_peak_aic': safe_float(best_peak_row['AIC']),
    }

    # ── Stage 8: Capacity Planning ───────────────────────────────
    progress_cb(74, 'Computing capacity planning report…')
    cap = compute_capacity_report(
        df_primary, server_capacity, num_servers,
        primary_fit, lognorm_params, nb_params
    )
    results['capacity'] = cap

    # ── Stage 9: Plain-English Verdicts ──────────────────────────
    progress_cb(78, 'Generating interpretations…')
    verdicts = get_plain_english_verdicts(primary_fit, dispersion_rows)
    results['verdicts'] = verdicts

    # ── Stage 10: Charts ─────────────────────────────────────────
    progress_cb(82, 'Generating E-Commerce charts…')
    figs = {}
    figs['ec_traffic']   = capture_figure(plot_traffic_over_time,  [df_ec_min], ['E-Commerce Logs'])
    figs['ec_rolling']   = capture_figure(plot_rolling_statistics, [df_ec_min], ['E-Commerce Logs'])
    figs['ec_temporal']  = capture_figure(plot_temporal_patterns,  [df_ec_min], ['E-Commerce Logs'])
    figs['ec_dist_all']  = capture_figure(plot_distribution_fits,  [df_ec_min, df_ec_fit_all],  ['E-Commerce Logs', 'all'])
    figs['ec_dist_peak'] = capture_figure(plot_distribution_fits,  [df_ec_min, df_ec_fit_peak], ['E-Commerce Logs', 'peak'])
    figs['ec_qq']        = capture_figure(plot_qq_plots,           [df_ec_min, df_ec_fit_all],  ['E-Commerce Logs'])

    progress_cb(91, 'Generating Web Server charts…')
    figs['ws_traffic']   = capture_figure(plot_traffic_over_time,  [df_ws_min], ['Web Server Logs'])
    figs['ws_rolling']   = capture_figure(plot_rolling_statistics, [df_ws_min], ['Web Server Logs'])
    figs['ws_temporal']  = capture_figure(plot_temporal_patterns,  [df_ws_min], ['Web Server Logs'])
    figs['ws_dist_all']  = capture_figure(plot_distribution_fits,  [df_ws_min, df_ws_fit_all],  ['Web Server Logs', 'all'])
    figs['ws_dist_peak'] = capture_figure(plot_distribution_fits,  [df_ws_min, df_ws_fit_peak], ['Web Server Logs', 'peak'])
    figs['ws_qq']        = capture_figure(plot_qq_plots,           [df_ws_min, df_ws_fit_all],  ['Web Server Logs'])
    results['figures'] = figs

    progress_cb(100, 'Pipeline complete.')
    return results


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run', methods=['POST'])
def run_pipeline():
    use_default      = request.form.get('use_default', 'true') == 'true'
    server_capacity  = int(request.form.get('server_capacity', 100))
    num_servers      = int(request.form.get('num_servers', 2))
    dataset_choice   = request.form.get('dataset_choice', 'ec')

    ec_file  = request.files.get('ec_file')
    ws_file  = request.files.get('ws_file')
    srv_file = request.files.get('srv_file')

    def event_stream():
        events = []
        lock   = threading.Lock()

        def progress_cb(pct, msg):
            with lock:
                events.append((pct, msg))

        result_holder = {}
        error_holder  = {}

        def worker():
            try:
                if use_default:
                    df_ec_raw = load_raw_ecommerce_data()
                    df_ws_raw, df_servers = load_raw_web_server_data()
                else:
                    if not ec_file or not ws_file or not srv_file:
                        raise ValueError('Please upload all three CSV files.')
                    df_ec_raw = pd.read_csv(ec_file, low_memory=False)
                    df_ws_raw = pd.read_csv(ws_file)
                    df_servers = pd.read_csv(srv_file)

                data = run_full_pipeline(
                    df_ec_raw, df_ws_raw, df_servers,
                    server_capacity, num_servers, dataset_choice,
                    progress_cb
                )
                result_holder['data'] = data
                _last_results['data'] = data

            except Exception as e:
                error_holder['msg'] = str(e)
                error_holder['trace'] = traceback.format_exc()
                progress_cb(100, f'ERROR: {e}')

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        sent = 0
        while t.is_alive() or sent < len(events):
            with lock:
                while sent < len(events):
                    pct, msg = events[sent]
                    yield f"data: {safe_json({'type': 'progress', 'pct': pct, 'msg': msg})}\n\n"
                    sent += 1
            time.sleep(0.15)

        if error_holder:
            yield f"data: {safe_json({'type': 'error', 'msg': error_holder['msg']})}\n\n"
        else:
            data = result_holder['data']
            # Send all data except figures first (small payload)
            figs = data.pop('figures', {})
            yield f"data: {safe_json({'type': 'result', 'data': data})}\n\n"
            # Send each figure individually
            for key, b64 in figs.items():
                yield f"data: {safe_json({'type': 'figure', 'key': key, 'b64': b64})}\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )


@app.route('/export')
def export_csv():
    """Export last run results as a multi-section CSV."""
    if 'data' not in _last_results:
        return 'No results available. Run the pipeline first.', 404

    data = _last_results['data']
    buf  = io.StringIO()
    w    = csv.writer(buf)

    # Capacity Planning
    cap = data.get('capacity', {})
    w.writerow(['=== CAPACITY PLANNING REPORT ==='])
    w.writerow(['Metric', 'Value'])
    for k, v in cap.items():
        if k not in ('measures', 'percentile_table'):
            w.writerow([k, v])
    w.writerow([])

    # Percentile Table
    w.writerow(['=== TRAFFIC PERCENTILE BREAKDOWN ==='])
    pt = cap.get('percentile_table', [])
    if pt:
        w.writerow(list(pt[0].keys()))
        for row in pt:
            w.writerow(list(row.values()))
    w.writerow([])

    # Measures
    w.writerow(['=== RECOMMENDED ACTIONS ==='])
    w.writerow(['Priority', 'Action', 'Reason'])
    for m in cap.get('measures', []):
        w.writerow([m['priority'], m['action'], m['reason']])
    w.writerow([])

    # Survival Probabilities
    w.writerow(['=== SURVIVAL PROBABILITIES ==='])
    surv = data.get('survival', {}).get('rows', [])
    if surv:
        w.writerow(list(surv[0].keys()))
        for row in surv:
            w.writerow(list(row.values()))
    w.writerow([])

    # Anomaly Log
    w.writerow(['=== ANOMALY LOG ==='])
    alog = data.get('anomalies', {}).get('log', [])
    if alog:
        w.writerow(list(alog[0].keys()))
        for row in alog:
            w.writerow(list(row.values()))
    w.writerow([])

    # Distribution Fits
    w.writerow(['=== DISTRIBUTION FITS (E-Commerce All) ==='])
    fits = data.get('distributions', {}).get('ec_all', [])
    if fits:
        w.writerow(list(fits[0].keys()))
        for row in fits:
            w.writerow(list(row.values()))

    buf.seek(0)
    resp = make_response(buf.getvalue())
    resp.headers['Content-Type']        = 'text/csv'
    resp.headers['Content-Disposition'] = 'attachment; filename=trafficstat_report.csv'
    return resp


if __name__ == '__main__':
    import os
    os.environ['PYTHONUNBUFFERED'] = '1'
    app.run(debug=True, threaded=True)
