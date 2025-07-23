// Dashboard rendering logic for the home page

function createElem(tag, className, html) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (html !== undefined) el.innerHTML = html;
  return el;
}

function summaryCard(title, value, subtitle) {
  const card = createElem('div', 'stat-card');
  card.innerHTML = `<h3>${title}</h3><p class="stat-value">${value}</p><p class="stat-subtitle">${subtitle}</p>`;
  return card;
}

function progressCard(title, done, total) {
  const pct = total ? (done / total) * 100 : 100;
  const card = createElem('div', 'progress-card');
  card.innerHTML = `<h3>${title}</h3>` +
    `<div class="progress-bar"><div class="progress-fill" style="width:${pct}%">${Math.round(pct)}%</div></div>` +
    `<div class="progress-stats"><span>${done} / ${total} files</span><span>${total - done} pending</span></div>`;
  return card;
}

function buildBarChart(data, container, valueKey, unit, gradient) {
  container.innerHTML = '';
  if (!data.length) {
    container.innerHTML = '<div style="text-align:center;color:#999;padding:2em;">No data available</div>';
    return;
  }
  const maxVal = Math.max(...data.map(d => d[valueKey])) || 1;
  const skip = Math.ceil(data.length / 30);
  data.forEach((d, i) => {
    if (i % skip !== 0) return;
    const bar = createElem('div', 'bar');
    bar.style.height = `${(d[valueKey] / maxVal) * 100}%`;
    if (gradient) bar.style.background = gradient;
    const label = createElem('div', 'bar-label', d.day);
    bar.appendChild(label);
    if (d[valueKey] > 0) {
      const val = createElem('div', 'bar-value', d[valueKey] + (unit || ''));
      bar.appendChild(val);
    }
    container.appendChild(bar);
  });
}

function buildHeatmap(data) {
  const container = document.getElementById('heatmap');
  container.innerHTML = '';
  if (!data.length) return;
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  let maxVal = 0;
  data.forEach(row => row.forEach(v => { if (v > maxVal) maxVal = v; }));
  container.appendChild(document.createElement('div'));
  const header = createElem('div', 'heatmap-header');
  for (let h = 0; h < 24; h++) header.appendChild(createElem('div', 'heatmap-hour', h));
  container.appendChild(header);
  for (let d = 0; d < 7; d++) {
    container.appendChild(createElem('div', 'heatmap-label', days[d]));
    for (let h = 0; h < 24; h++) {
      const intensity = maxVal ? data[d][h] / maxVal : 0;
      const cell = createElem('div', 'heatmap-cell');
      cell.style.background = `rgba(102,126,234,${intensity})`;
      cell.title = `${days[d]} ${h}:00 - ${Math.round(data[d][h])} min`;
      container.appendChild(cell);
    }
  }
}

function buildTopics(counts, minutes) {
  const names = Object.keys(counts || {});
  if (!names.length) return;
  const section = document.getElementById('topicsSection');
  section.innerHTML = '<h2>Topics</h2><div class="topics-grid" id="topicsGrid"></div>';
  const grid = document.getElementById('topicsGrid');
  names.sort((a, b) => counts[b] - counts[a]);
  names.forEach(name => {
    const card = createElem('div', 'topic-card');
    card.innerHTML = `<div class="topic-name">${name}</div>` +
      `<div class="topic-stats"><span>${counts[name]} occurrences</span>` +
      `<span>${Math.round(minutes[name] || 0)}m</span></div>`;
    grid.appendChild(card);
  });
}

function buildRepairs(totals) {
  const categories = {
    repair_hear: 'Audio',
    repair_see: 'Screenshots',
    repair_reduce: 'Summaries',
    repair_entity: 'Entities',
    repair_ponder: 'Ponder'
  };
  const any = Object.keys(categories).some(k => (totals[k] || 0) > 0);
  if (!any) return;
  const section = document.getElementById('repairSection');
  section.innerHTML = '<div class="chart-section" style="background:#fff3cd;border:1px solid #ffeaa7;">' +
    '<h2>Items Needing Repair</h2><div class="stats-grid" id="repairGrid" style="margin-bottom:0;"></div></div>';
  const grid = document.getElementById('repairGrid');
  Object.keys(categories).forEach(key => {
    const count = totals[key] || 0;
    if (!count) return;
    const card = createElem('div', 'stat-card');
    card.innerHTML = `<h3>${categories[key]}</h3><p class="stat-value" style="color:#f5576c;">${count}</p>`;
    grid.appendChild(card);
  });
}

function renderDashboard(data) {
  if (!data || !data.days || Object.keys(data.days).length === 0) {
    document.getElementById('notice').innerHTML = '<div style="background:#fff3cd;border:1px solid #ffeaa7;border-radius:8px;padding:1em;margin-bottom:2em;">' +
      '<strong>No data available.</strong> Run journal_stats.py to generate statistics.</div>';
    return;
  }
  const days = Object.keys(data.days).sort();
  const totals = data.totals || {};
  const totalAudioMB = ((data.total_audio_bytes || 0) + (data.total_image_bytes || 0)) / (1024 * 1024);

  const statsGrid = document.getElementById('statsGrid');
  statsGrid.appendChild(summaryCard('Total Days', days.length, 'days recorded'));
  statsGrid.appendChild(summaryCard('Audio Duration', (data.total_audio_seconds / 3600).toFixed(1), 'hours recorded'));
  statsGrid.appendChild(summaryCard('Storage Used', Math.round(totalAudioMB), 'MB total'));
  const completion = totals.audio_flac ? (totals.audio_json / totals.audio_flac) * 100 : 100;
  statsGrid.appendChild(summaryCard('Processing Status', Math.round(completion) + '%', 'complete'));

  const progressSection = document.getElementById('progressSection');
  progressSection.appendChild(progressCard('Audio Transcription', totals.audio_json || 0, totals.audio_flac || 0));
  progressSection.appendChild(progressCard('Screenshot Analysis', totals.desc_json || 0, totals.diff_png || 0));

  const activityData = days.map(day => ({ day, value: data.days[day].activity || 0 }));
  buildBarChart(activityData, document.getElementById('activityChart'), 'value');

  const audioData = days.map(day => ({ day, hours: (data.days[day].audio_seconds || 0) / 3600 }));
  buildBarChart(audioData, document.getElementById('audioChart'), 'hours', 'h', 'linear-gradient(to top,#f093fb,#f5576c)');

  buildHeatmap(data.heatmap || []);
  buildTopics(data.topic_counts || {}, data.topic_minutes || {});
  buildRepairs(totals);
}

function loadStats(url) {
  fetch(url).then(r => r.json()).then(renderDashboard).catch(() => {
    document.getElementById('notice').textContent = 'Failed to load stats';
  });
}

export { loadStats };
