'use strict';

// State
let currentMode = 'default';
let distData    = {};
let running     = false;

// Init
document.addEventListener('DOMContentLoaded', () => {
  setupSidebar();
  setupUploads();
  setupScrollSpy();
  setInterval(() => {
    const el = document.getElementById('tb-time');
    if (el) el.textContent = new Date().toLocaleTimeString();
  }, 1000);
});

// Sidebar
function setupSidebar() {
  const btn  = document.getElementById('hamburger');
  const side = document.getElementById('sidebar');
  btn.addEventListener('click', () => side.classList.toggle('open'));
  document.addEventListener('click', e => {
    if (!side.contains(e.target) && !btn.contains(e.target))
      side.classList.remove('open');
  });
}

function setNav(el) {
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  el.classList.add('active');
  const href = el.getAttribute('href');
  const id   = href ? href.replace('#', '') : '';
  const sec  = document.getElementById(id);
  if (sec) {
    const label = sec.querySelector('.sec-title');
    const bc = document.getElementById('breadcrumb-current');
    if (bc && label) bc.textContent = label.textContent;
  }
}

function setupScrollSpy() {
  const sections = document.querySelectorAll('section[id]');
  const navItems = document.querySelectorAll('.nav-item');
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        navItems.forEach(n => n.classList.remove('active'));
        const a = document.querySelector(`.nav-item[href="#${e.target.id}"]`);
        if (a) a.classList.add('active');
      }
    });
  }, { threshold: 0.25 });
  sections.forEach(s => obs.observe(s));
}

// Tab switching
function switchTab(mode) {
  currentMode = mode;
  ['default','custom'].forEach(m => {
    document.getElementById(`panel-${m}`).classList.toggle('active', m === mode);
    document.getElementById(`tab-${m}`).classList.toggle('active', m === mode);
  });
}

// File uploads
function setupUploads() {
  [['drop-ec','file-ec','name-ec'],
   ['drop-ws','file-ws','name-ws'],
   ['drop-srv','file-srv','name-srv']].forEach(([dropId, inputId, nameId]) => {
    const box   = document.getElementById(dropId);
    const input = document.getElementById(inputId);
    const label = document.getElementById(nameId);
    if (!box || !input) return;
    box.addEventListener('click', () => input.click());
    input.addEventListener('change', () => {
      if (input.files[0]) { label.textContent = input.files[0].name; box.classList.add('has-file'); }
    });
    box.addEventListener('dragover',  e => { e.preventDefault(); box.classList.add('drag-over'); });
    box.addEventListener('dragleave', () => box.classList.remove('drag-over'));
    box.addEventListener('drop', e => {
      e.preventDefault(); box.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file) {
        const dt = new DataTransfer(); dt.items.add(file); input.files = dt.files;
        label.textContent = file.name; box.classList.add('has-file');
      }
    });
  });
}

// Run Pipeline
function runPipeline() {
  if (running) return;
  running = true;

  const btn = document.getElementById('run-btn');
  btn.disabled = true;

  // Reset sections to placeholder state
  ['capacity','anomaly','survival','ingestion','eda','dispersion','dist','charts'].forEach(key => {
    const ph = document.getElementById(`placeholder-${key}`);
    if (ph) ph.style.display = 'flex';
  });
  ['eda-content','dispersion-content','dist-content','charts-content'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  });
  // Clear previous results from always-visible content areas
  ['cap-kpi-row','tbl-percentile','bottleneck-panel','uptime-panel','measures-list',
   'risk-badge-top','anomaly-summary-row','tbl-anomaly-log','best-model-row',
   'tbl-survival','kpi-ingestion','kpi-aggregation'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = '';
  });
  // Clear chart images
  ['ec-traffic','ec-rolling','ec-temporal','ec-dist-all','ec-dist-peak','ec-qq',
   'ws-traffic','ws-rolling','ws-temporal','ws-dist-all','ws-dist-peak','ws-qq'].forEach(key => {
    const img = document.getElementById('fig-' + key);
    if (img) img.src = '';
  });
  document.getElementById('error-box').style.display = 'none';
  document.getElementById('progress-steps').innerHTML = '';
  document.getElementById('export-btn').style.display = 'none';

  const pw = document.getElementById('progress-wrap');
  pw.style.display = 'block';
  setProgress(0, 'Starting pipeline...');
  setStatus('running');

  const spinner = document.getElementById('spinner');
  if (spinner) spinner.classList.remove('done');

  const fd = new FormData();
  fd.append('use_default', currentMode === 'default' ? 'true' : 'false');
  fd.append('server_capacity', document.getElementById('server-capacity').value || '100');
  fd.append('num_servers',     document.getElementById('num-servers').value     || '2');
  fd.append('dataset_choice',  document.getElementById('dataset-choice').value  || 'ec');

  if (currentMode === 'custom') {
    const ec  = document.getElementById('file-ec').files[0];
    const ws  = document.getElementById('file-ws').files[0];
    const srv = document.getElementById('file-srv').files[0];
    if (!ec || !ws || !srv) { showError('Please upload all three CSV files before running.'); resetBtn(); return; }
    fd.append('ec_file', ec);
    fd.append('ws_file', ws);
    fd.append('srv_file', srv);
  }

  fetch('/run', { method: 'POST', body: fd })
    .then(res => {
      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';

      function processLines() {
        let idx;
        while ((idx = buf.indexOf('\n\n')) !== -1) {
          const block = buf.slice(0, idx).trim();
          buf = buf.slice(idx + 2);
          if (!block.startsWith('data: ')) continue;
          try { handleMsg(JSON.parse(block.slice(6))); } catch(e) { console.error('SSE parse error:', e, block.slice(6, 200)); }
        }
      }

      function read() {
        reader.read().then(({ done, value }) => {
          if (done) { resetBtn(); return; }
          buf += decoder.decode(value, { stream: true });
          processLines();
          read();
        });
      }
      read();
    })
    .catch(err => { showError('Network error: ' + err.message); resetBtn(); });
}

function handleMsg(msg) {
  if (msg.type === 'progress') {
    setProgress(msg.pct, msg.msg);
    addLog(msg.msg, msg.pct >= 100);
  } else if (msg.type === 'result') {
    if (!msg.data) { showError('Server returned empty result.'); return; }
    addLog('Result received — rendering...', false);
    renderAll(msg.data);
  } else if (msg.type === 'figure') {
    const id  = 'fig-' + msg.key.replace(/_/g, '-');
    const img = document.getElementById(id);
    if (img) { img.src = 'data:image/png;base64,' + msg.b64; img.onclick = () => openLightbox(img.src); }
  } else if (msg.type === 'error') {
    showError('Pipeline error: ' + msg.msg);
    setStatus('error');
    resetBtn();
  }
}

function setProgress(pct, msg) {
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-pct').textContent = pct + '%';
  document.getElementById('progress-msg').textContent = msg;
}

function addLog(msg, done) {
  const el = document.createElement('div');
  el.className = 'log-line' + (done ? ' done' : '');
  el.innerHTML = `<span class="log-dot"></span>${msg}`;
  const steps = document.getElementById('progress-steps');
  steps.appendChild(el);
  steps.scrollTop = steps.scrollHeight;
}

function showError(msg) {
  const b = document.getElementById('error-box');
  b.textContent = 'Error: ' + msg;
  b.style.display = 'block';
}

function setStatus(state) {
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  dot.className = 'status-dot ' + state;
  text.textContent = state === 'running' ? 'Running...' : state === 'done' ? 'Complete' : state === 'error' ? 'Error' : 'Ready';
}

function resetBtn() {
  running = false;
  document.getElementById('run-btn').disabled = false;
  const sp = document.getElementById('spinner');
  if (sp) sp.classList.add('done');
}

function exportCSV() { window.location.href = '/export'; }

// RENDER ALL
function renderAll(data) {
  ['capacity','anomaly','survival','ingestion','eda','dispersion','dist','charts'].forEach(key => {
    const ph = document.getElementById(`placeholder-${key}`);
    if (ph) ph.style.display = 'none';
  });
  ['eda-content','dispersion-content','dist-content','charts-content'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'block';
  });

  const run = (fn, label) => { try { fn(); } catch(e) { console.error(label + ' failed:', e); } };
  run(() => renderCapacity(data.capacity),                         'renderCapacity');
  run(() => renderAnomalyLog(data.anomalies),                      'renderAnomalyLog');
  run(() => renderSurvival(data.survival),                         'renderSurvival');
  run(() => renderIngestion(data.ingestion, data.aggregation),     'renderIngestion');
  run(() => renderEDA(data.eda),                                   'renderEDA');
  run(() => renderDispersion(data.dispersion, data.verdicts),      'renderDispersion');
  run(() => renderDistributions(data.distributions, data.verdicts),'renderDistributions');

  // Show chart containers (images arrive via figure messages)
  const ec = document.getElementById('charts-ec');
  const ws = document.getElementById('charts-ws');
  if (ec) ec.style.display = 'block';
  if (ws) ws.style.display = 'none';

  document.getElementById('export-btn').style.display = 'flex';
  setStatus('done');
  setTimeout(() => {
    document.getElementById('sec-capacity').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 200);
}

// Capacity Report
function renderCapacity(cap) {
  if (!cap) return;
  document.getElementById('risk-badge-top').innerHTML =
    `<span class="risk-badge risk-${cap.risk_level}">${cap.risk_level} RISK</span>`;

  document.getElementById('cap-kpi-row').innerHTML = [
    kpi('Total Capacity',    fmt(cap.total_capacity) + ' req/min', `${cap.num_servers} servers x ${fmt(cap.server_capacity_per_unit)}`, 'accent'),
    kpi('Mean Utilization',  cap.util_mean.toFixed(1) + '%', `${fmt(cap.mean_rpm)} req/min avg`, utilClass(cap.util_mean)),
    kpi('P90 Utilization',   cap.util_p90.toFixed(1)  + '%', `${fmt(cap.p90_rpm)} req/min at p90`, utilClass(cap.util_p90)),
    kpi('P99 Utilization',   cap.util_p99.toFixed(1)  + '%', `${fmt(cap.p99_rpm)} req/min at p99`, utilClass(cap.util_p99)),
    kpi('Peak Observed',     fmt(cap.max_rpm) + ' req/min', `${cap.util_max.toFixed(1)}% of capacity`, utilClass(cap.util_max)),
    kpi('Uptime Estimate',   cap.uptime_pct.toFixed(2) + '%', 'Minutes within capacity', cap.uptime_pct > 99.9 ? 'green' : cap.uptime_pct > 99 ? 'yellow' : 'red'),
  ].join('');

  const ptCols = ['Percentile','Traffic (req/min)','Utilization %','Status','Headroom (req/min)'];
  document.getElementById('tbl-percentile').innerHTML = buildTable(ptCols, cap.percentile_table, (col, val) => {
    if (col === 'Status') {
      const cls = val === 'Normal' ? 'badge-green' : val === 'Warning' ? 'badge-yellow' : 'badge-red';
      return `<span class="badge ${cls}">${val}</span>`;
    }
    if (col === 'Utilization %') {
      const n = parseFloat(val);
      const color = n >= 90 ? 'var(--red)' : n >= 70 ? 'var(--yellow)' : 'var(--green)';
      return `<span style="color:${color};font-family:var(--mono);font-weight:600">${n.toFixed(1)}%</span>`;
    }
    if (col === 'Traffic (req/min)' || col === 'Headroom (req/min)')
      return `<span class="mono">${fmtN(parseFloat(val))}</span>`;
    return `<strong>${val}</strong>`;
  });

  document.getElementById('bottleneck-panel').innerHTML = `
    <div class="ip-row"><span class="ip-label">Warning threshold (70% capacity)</span><span class="ip-value yellow">${fmt(cap.bottleneck_warning_rpm)} req/min</span></div>
    <div class="ip-row"><span class="ip-label">Critical threshold (90% capacity)</span><span class="ip-value red">${fmt(cap.bottleneck_critical_rpm)} req/min</span></div>
    <div class="ip-row"><span class="ip-label">Bottleneck range</span><span class="ip-value">${fmt(cap.bottleneck_warning_rpm)} - ${fmt(cap.bottleneck_critical_rpm)} req/min</span></div>
    <div class="ip-row"><span class="ip-label">Servers needed for p99 (+20% headroom)</span><span class="ip-value accent">${cap.recommended_for_p99} servers</span></div>
    <div class="ip-row"><span class="ip-label">Servers needed for observed max (+20%)</span><span class="ip-value">${cap.recommended_for_max} servers</span></div>
    <div class="ip-row" style="border:none"><span class="ip-label" style="color:var(--muted);font-size:11px">${cap.risk_desc}</span></div>`;

  const mtbo = cap.mtbo_hours != null
    ? `Every <strong>${cap.mtbo_hours.toFixed(1)} hours</strong> (${cap.mtbo_minutes.toFixed(0)} min)`
    : '<span style="color:var(--green)">No overloads observed</span>';
  const pLN = cap.p_exceed_lognorm != null ? cap.p_exceed_lognorm.toExponential(3) : '-';
  const pNB = cap.p_exceed_nb      != null ? cap.p_exceed_nb.toExponential(3)      : '-';
  document.getElementById('uptime-panel').innerHTML = `
    <div class="ip-row"><span class="ip-label">Empirical uptime</span><span class="ip-value ${cap.uptime_pct > 99.9 ? 'green' : cap.uptime_pct > 99 ? 'yellow' : 'red'}">${cap.uptime_pct.toFixed(3)}%</span></div>
    <div class="ip-row"><span class="ip-label">Overload events observed</span><span class="ip-value ${cap.overload_count > 0 ? 'red' : 'green'}">${cap.overload_count} events (${cap.overload_freq_pct.toFixed(3)}%)</span></div>
    <div class="ip-row"><span class="ip-label">Mean time between overloads</span><span class="ip-value">${mtbo}</span></div>
    <div class="ip-row"><span class="ip-label">P(exceed capacity) - Log-Normal model</span><span class="ip-value mono">${pLN}</span></div>
    <div class="ip-row" style="border:none"><span class="ip-label">P(exceed capacity) - Neg-Binomial model</span><span class="ip-value mono">${pNB}</span></div>`;

  document.getElementById('measures-list').innerHTML = '<div class="measures-list">' +
    cap.measures.map(m => `
      <div class="measure-item">
        <span class="measure-priority pri-${CSS.escape(m.priority)}">${m.priority}</span>
        <div class="measure-body">
          <div class="measure-action">${m.action}</div>
          <div class="measure-reason">${m.reason}</div>
        </div>
      </div>`).join('') + '</div>';
}

function utilClass(pct) {
  return pct >= 90 ? 'red' : pct >= 70 ? 'yellow' : pct >= 50 ? 'orange' : 'green';
}

// Anomaly Log
function renderAnomalyLog(anomalies) {
  if (!anomalies) return;
  const summary = anomalies.summary || [];
  const kpis = [];
  summary.forEach(row => {
    const prefix = row['Dataset'].includes('E-Commerce') ? 'EC' : 'WS';
    kpis.push(kpi(`${prefix} Z-Score Anomalies`, row['Z-Score Anomalies'], `${row['Z-Score %']}% of intervals`, row['Z-Score %'] > 5 ? 'red' : row['Z-Score %'] > 2 ? 'yellow' : 'green'));
    kpis.push(kpi(`${prefix} IQR Anomalies`,     row['IQR Anomalies'],     `${row['IQR %']}% of intervals`,     row['IQR %'] > 8 ? 'red' : row['IQR %'] > 4 ? 'yellow' : 'green'));
  });
  document.getElementById('anomaly-summary-row').innerHTML = kpis.join('');

  const log = anomalies.log || [];
  if (!log.length) {
    document.getElementById('tbl-anomaly-log').innerHTML = '<p style="padding:16px;color:var(--muted);font-size:12px">No anomalies detected.</p>';
    return;
  }
  const cols = Object.keys(log[0]);
  document.getElementById('tbl-anomaly-log').innerHTML = buildTable(cols, log, (col, val) => {
    if (col === 'Severity') {
      const cls = val === 'EXTREME' ? 'badge-red' : val === 'SEVERE' ? 'badge-yellow' : 'badge-gray';
      return `<span class="badge ${cls}">${val}</span>`;
    }
    if (col === 'IQR Outlier') return val === 'Yes' ? '<span class="badge badge-yellow">Yes</span>' : '<span style="color:var(--dim)">No</span>';
    if (col === 'Requests/min') return `<span class="mono" style="color:var(--red);font-weight:600">${fmt(val)}</span>`;
    if (col === 'Z-Score')      return `<span class="mono">${val}</span>`;
    if (col === 'vs Mean')      return `<span class="mono" style="color:var(--orange)">${val}</span>`;
    if (col === 'Timestamp')    return `<span class="mono">${val}</span>`;
    return val;
  });
}

// Survival Probabilities
function renderSurvival(surv) {
  if (!surv) return;
  document.getElementById('best-model-row').innerHTML = `
    <div class="card" style="padding:14px 18px">
      <div class="kpi-label">Best Model - All Traffic</div>
      <div class="kpi-value" style="font-size:18px;color:var(--green)">${surv.best_all}</div>
      <div class="kpi-sub">AIC: ${fmtN(surv.best_all_aic)} - Use for capacity planning</div>
    </div>
    <div class="card" style="padding:14px 18px">
      <div class="kpi-label">Best Model - Peak Traffic</div>
      <div class="kpi-value" style="font-size:18px;color:var(--green)">${surv.best_peak}</div>
      <div class="kpi-sub">AIC: ${fmtN(surv.best_peak_aic)} - Use for flash-sale provisioning</div>
    </div>`;

  const cols = ['Threshold (req/min)','Empirical Frequency','Poisson P(X>T)','Neg-Binomial P(X>T)','Log-Normal P(X>T)'];
  document.getElementById('tbl-survival').innerHTML = buildTable(cols, surv.rows, (col, val) => {
    if (col === 'Threshold (req/min)') return `<strong class="mono">${val}</strong>`;
    if (val === null || val === undefined) return '-';
    const n = parseFloat(val);
    if (isNaN(n)) return val;
    const str = n.toExponential(4);
    if (col === 'Empirical Frequency') return `<span class="badge badge-blue mono">${str}</span>`;
    if (col === 'Poisson P(X>T)')      return `<span class="mono" style="color:var(--red)">${str}</span>`;
    return `<span class="mono">${str}</span>`;
  });
}

// Ingestion
function renderIngestion(ing, agg) {
  document.getElementById('kpi-ingestion').innerHTML = [
    kpi('EC Events',   fmt(ing.ec_rows), 'E-Commerce raw rows', 'accent'),
    kpi('WS Records',  fmt(ing.ws_rows), 'Web Server raw rows', 'accent'),
    kpi('EC Features', ing.ec_cols,      'Columns after cleaning'),
    kpi('WS Features', ing.ws_cols,      'Columns after join'),
  ].join('');
  document.getElementById('kpi-aggregation').innerHTML = [
    kpi('EC Intervals',      fmt(agg.ec_minutes), '1-min buckets', 'accent'),
    kpi('EC Range',          `${agg.ec_req_min}-${agg.ec_req_max}`, 'req/min observed'),
    kpi('EC P90 Threshold',  agg.ec_p90, 'Peak cutoff (req/min)'),
    kpi('WS Intervals',      fmt(agg.ws_minutes), '1-min buckets', 'accent'),
    kpi('WS Range',          `${agg.ws_req_min}-${agg.ws_req_max}`, 'req/min observed'),
    kpi('WS P90 Threshold',  agg.ws_p90, 'Peak cutoff (req/min)'),
  ].join('');
}

// EDA
function renderEDA(eda) {
  renderRegime('tbl-ec-regimes', eda.ec_regimes);
  renderRegime('tbl-ws-regimes', eda.ws_regimes);
  renderCorr('tbl-ec-corr', eda.ec_corr, eda.ec_corr_cols);
  renderCorr('tbl-ws-corr', eda.ws_corr, eda.ws_corr_cols);
}

function renderRegime(id, rows) {
  if (!rows || !rows.length) return;
  const cols = Object.keys(rows[0]);
  document.getElementById(id).innerHTML = buildTable(cols, rows, (col, val) => {
    if (col === 'Metric') return `<strong>${val}</strong>`;
    if (val === null || val === undefined) return '-';
    const n = parseFloat(val);
    return !isNaN(n) ? `<span class="mono">${fmtN(n)}</span>` : val;
  });
}

function renderCorr(id, rows, cols) {
  if (!rows || !rows.length) return;
  document.getElementById(id).innerHTML = buildTable(cols, rows, (col, val) => {
    if (col === 'Feature') return `<strong>${val}</strong>`;
    if (val === null || val === undefined) return '-';
    const n = parseFloat(val);
    if (!isNaN(n)) {
      const abs = Math.abs(n);
      const color = abs > 0.7 ? 'var(--green)' : abs > 0.4 ? 'var(--yellow)' : 'var(--muted)';
      return `<span class="mono" style="color:${color}">${n.toFixed(4)}</span>`;
    }
    return val;
  });
}

// Dispersion
function renderDispersion(rows, verdicts) {
  const cols = ['Dataset','Mean','Variance','Dispersion Index','Overdispersed','Interpretation'];
  const enriched = rows.map(r => ({ ...r, 'Interpretation': (verdicts && verdicts[`disp_${r.Dataset}`]) || '-' }));
  document.getElementById('tbl-dispersion').innerHTML = buildTable(cols, enriched, (col, val) => {
    if (col === 'Overdispersed')
      return val ? '<span class="badge badge-red">Overdispersed</span>' : '<span class="badge badge-green">Equidispersed</span>';
    if (col === 'Dispersion Index' && val !== null) {
      const n = parseFloat(val);
      const color = n > 5 ? 'var(--red)' : n > 1.2 ? 'var(--yellow)' : 'var(--green)';
      return `<span class="mono" style="color:${color};font-weight:600">${fmtN(n)}</span>`;
    }
    if (col === 'Interpretation') return `<span style="font-size:11px;color:var(--muted);white-space:normal;max-width:300px;display:block">${val}</span>`;
    if (val === null || val === undefined) return '-';
    const n = parseFloat(val);
    return !isNaN(n) ? `<span class="mono">${fmtN(n)}</span>` : val;
  });
}

// Distributions
function renderDistributions(dists, verdicts) {
  distData = { dists, verdicts };
  renderDistTable('ec_all');
}

function switchDist(key, btn) {
  document.querySelectorAll('.seg').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderDistTable(key);
}

function renderDistTable(key) {
  const rows = distData.dists ? distData.dists[key] : null;
  if (!rows || !rows.length) {
    document.getElementById('tbl-dist').innerHTML = '<p style="padding:16px;color:var(--muted);font-size:12px">No data.</p>';
    return;
  }
  let bestIdx = -1, bestAIC = Infinity;
  rows.forEach((r, i) => { const a = parseFloat(r['AIC']); if (!isNaN(a) && a < bestAIC) { bestAIC = a; bestIdx = i; } });

  const cols = ['Distribution','Type','AIC','BIC','GoF Stat','p-value','Verdict'];
  const verdicts = distData.verdicts || {};
  const enriched = rows.map(r => ({ ...r, 'Verdict': verdicts[`dist_${r['Distribution']}`] || '-' }));

  document.getElementById('tbl-dist').innerHTML = buildTable(cols, enriched, (col, val, ri) => {
    if (val === null || val === undefined) return '-';
    if (col === 'Type')
      return val === 'Discrete' ? '<span class="badge badge-blue">Discrete</span>' : '<span class="badge badge-purple">Continuous</span>';
    if (col === 'AIC' || col === 'BIC' || col === 'GoF Stat') {
      const n = parseFloat(val); return !isNaN(n) ? `<span class="mono">${fmtN(n)}</span>` : '-';
    }
    if (col === 'p-value') {
      const n = parseFloat(val); if (isNaN(n)) return '-';
      return `<span class="badge ${n > 0.05 ? 'badge-green' : 'badge-red'} mono">${n.toExponential(3)}</span>`;
    }
    if (col === 'Verdict') {
      const isBest = val.startsWith('BEST'), isRej = val.startsWith('REJECTED');
      const c = isBest ? 'var(--green)' : isRej ? 'var(--red)' : 'var(--muted)';
      return `<span style="color:${c};font-size:11px;white-space:normal;max-width:280px;display:block">${val}</span>`;
    }
    return val;
  }, ri => ri === bestIdx ? 'row-best' : '');
}

// Charts
function renderCharts(figs) {
  if (!figs) return;
  const map = {
    'fig-ec-traffic': figs.ec_traffic, 'fig-ec-rolling': figs.ec_rolling,
    'fig-ec-temporal': figs.ec_temporal, 'fig-ec-dist-all': figs.ec_dist_all,
    'fig-ec-dist-peak': figs.ec_dist_peak, 'fig-ec-qq': figs.ec_qq,
    'fig-ws-traffic': figs.ws_traffic, 'fig-ws-rolling': figs.ws_rolling,
    'fig-ws-temporal': figs.ws_temporal, 'fig-ws-dist-all': figs.ws_dist_all,
    'fig-ws-dist-peak': figs.ws_dist_peak, 'fig-ws-qq': figs.ws_qq,
  };
  Object.entries(map).forEach(([id, b64]) => {
    if (!b64) return;
    const img = document.getElementById(id);
    if (img) { img.src = 'data:image/png;base64,' + b64; img.onclick = () => openLightbox(img.src); }
  });
}

function switchCharts(group, btn) {
  document.querySelectorAll('.seg').forEach(b => {
    if (b.closest('.seg-ctrl') === btn.closest('.seg-ctrl')) b.classList.remove('active');
  });
  btn.classList.add('active');
  document.getElementById('charts-ec').style.display = group === 'ec' ? 'block' : 'none';
  document.getElementById('charts-ws').style.display = group === 'ws' ? 'block' : 'none';
}

// Lightbox
function openLightbox(src) {
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox').classList.add('open');
}
function closeLightbox() { document.getElementById('lightbox').classList.remove('open'); }
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

// Helpers
function kpi(label, value, sub, colorClass) {
  return `<div class="kpi ${colorClass || ''}">
    <div class="kpi-label">${label}</div>
    <div class="kpi-value">${value}</div>
    <div class="kpi-sub">${sub || ''}</div>
  </div>`;
}

function buildTable(cols, rows, cellFn, rowClassFn) {
  const head = `<thead><tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr></thead>`;
  const body = rows.map((row, ri) => {
    const cls   = rowClassFn ? rowClassFn(ri) : '';
    const cells = cols.map(c => `<td>${cellFn(c, row[c], ri)}</td>`).join('');
    return `<tr class="${cls}">${cells}</tr>`;
  }).join('');
  return `<table>${head}<tbody>${body}</tbody></table>`;
}

function fmt(n) {
  if (n === null || n === undefined) return '-';
  return Number(n).toLocaleString();
}

function fmtN(n) {
  if (n === null || n === undefined || isNaN(n)) return '-';
  const abs = Math.abs(n);
  if (abs >= 1e6 || (abs < 0.001 && n !== 0)) return n.toExponential(3);
  if (Number.isInteger(n)) return n.toLocaleString();
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 });
}
