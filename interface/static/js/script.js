'use strict';

// ── Config ───────────────────────────────────────────
const WS_URL = `ws://${location.host}/ws`;

const ACTIVITY_KEYS = ['memperhatikan', 'menulis', 'membaca', 'angkat_tangan', 'bermain_hp', 'tidur'];
const LEARN_KEYS    = ['memperhatikan', 'menulis', 'membaca', 'angkat_tangan'];
const NONLEARN_KEYS = ['bermain_hp', 'tidur'];

// ── State ────────────────────────────────────────────
let ws             = null;
let reconnectTimer = null;
let sessionStart   = Date.now();
let frameCount     = 0;
let fpsTimer       = null;
let lastFpsCount   = 0;

// ── DOM refs (cached once) ───────────────────────────
const el = {
  connBadge:    document.getElementById('conn-badge'),
  wsDot:        document.getElementById('ws-dot'),
  wsStatus:     document.getElementById('ws-status'),
  sessionTime:  document.getElementById('session-time'),
  detected:     document.getElementById('m-detected'),
  engagement:   document.getElementById('m-engagement'),
  engSub:       document.getElementById('m-eng-sub'),
  learning:     document.getElementById('m-learning'),
  nonlearning:  document.getElementById('m-nonlearning'),
  detBadge:     document.getElementById('det-badge'),
  camFeed:      document.getElementById('cam-feed'),
  camPlaceholder: document.getElementById('cam-placeholder'),
  engBar:       document.getElementById('eng-bar'),
  engLabelText: document.getElementById('eng-label-text'),
  engLabelPct:  document.getElementById('eng-label-pct'),
  camFps:       document.getElementById('cam-fps'),
  reportStats:  document.getElementById('report-stats'),
  rptEngagement: document.getElementById('rpt-engagement'),
  rptFrames:    document.getElementById('rpt-frames'),
  rptLearning:  document.getElementById('rpt-learning'),
  reportMessage: document.getElementById('report-message'),
  previewBtn:   document.getElementById('preview-report'),
  downloadBtn:  document.getElementById('generate-report'),
};

// ── Helpers ──────────────────────────────────────────
function safe(val, fallback = 0) {
  const n = Number(val);
  return isFinite(n) ? n : fallback;
}

function setConn(state, text) {
  el.connBadge.dataset.state = state;
  el.connBadge.textContent   = text;
  el.wsDot.className = `ws-dot ${state}`;
  el.wsStatus.textContent = `WebSocket: ${text.toLowerCase()}`;
}

function engLabel(pct) {
  if (pct >= 70) return 'Tinggi';
  if (pct >= 40) return 'Sedang';
  return 'Rendah';
}

function engColor(pct) {
  if (pct >= 70) return '#4ade80';
  if (pct >= 40) return '#fbbf24';
  return '#f87171';
}

// ── Session Timer ────────────────────────────────────
function startSessionTimer() {
  setInterval(() => {
    const diff = Math.floor((Date.now() - sessionStart) / 1000);
    const h = String(Math.floor(diff / 3600)).padStart(2, '0');
    const m = String(Math.floor((diff % 3600) / 60)).padStart(2, '0');
    const s = String(diff % 60).padStart(2, '0');
    el.sessionTime.textContent = `${h}:${m}:${s}`;
  }, 1000);
}

// ── FPS Counter ──────────────────────────────────────
function startFpsCounter() {
  fpsTimer = setInterval(() => {
    const fps = frameCount - lastFpsCount;
    lastFpsCount = frameCount;
    el.camFps.textContent = `${fps} fps`;
  }, 1000);
}

// ── UI Update ────────────────────────────────────────
function updateUI(d) {
  frameCount++;

  const act        = d.activities || {};
  const detected   = safe(d.detected);
  const engIdx     = safe(d.engagement_index);

  const learningPct    = LEARN_KEYS.reduce((s, k)    => s + safe(act[k]), 0);
  const nonlearningPct = NONLEARN_KEYS.reduce((s, k) => s + safe(act[k]), 0);

  // Metric cards
  el.detected.textContent    = detected;
  el.engagement.textContent  = `${engIdx}%`;
  el.learning.textContent    = `${learningPct}%`;
  el.nonlearning.textContent = `${nonlearningPct}%`;
  el.detBadge.textContent    = `${detected} detected`;

  // Metric value colors
  el.engagement.style.color  = engColor(engIdx);
  el.learning.style.color    = '#4ade80';
  el.nonlearning.style.color = nonlearningPct > 30 ? '#f87171' : '#94a3b8';

  // Activity bars
  ACTIVITY_KEYS.forEach(k => {
    const v   = safe(act[k]);
    const bar = document.getElementById('b-' + k);
    const pct = document.getElementById('p-' + k);
    if (bar) bar.style.width   = `${v}%`;
    if (pct) pct.textContent   = `${v}%`;
  });

  // Engagement bar
  el.engBar.style.width      = `${engIdx}%`;
  el.engBar.style.background = engColor(engIdx);
  el.engLabelText.textContent = engLabel(engIdx);
  el.engLabelPct.textContent  = `${engIdx}%`;
  el.engLabelPct.style.color  = engColor(engIdx);

  // Camera frame
  if (d.frame) {
    el.camFeed.src              = `data:image/jpeg;base64,${d.frame}`;
    el.camFeed.hidden           = false;
    el.camPlaceholder.hidden    = true;
  }
}

// ── WebSocket ────────────────────────────────────────
function connect() {
  setConn('disconnected', 'Connecting...');

  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    clearTimeout(reconnectTimer);
    setConn('connected', 'Connected');
  };

  ws.onmessage = (e) => {
    try {
      const d = JSON.parse(e.data);
      if (d.error) { console.error('[WS]', d.error); return; }
      updateUI(d);
    } catch (err) {
      console.error('[WS] Parse error:', err);
    }
  };

  ws.onclose = () => {
    setConn('reconnecting', 'Reconnecting...');
    reconnectTimer = setTimeout(connect, 3000);
  };

  ws.onerror = () => ws.close();
}

// ── Report ───────────────────────────────────────────
function showMessage(text, type = 'success') {
  el.reportMessage.textContent  = text;
  el.reportMessage.className    = `report-message ${type}`;
  el.reportMessage.hidden       = false;
  setTimeout(() => { el.reportMessage.hidden = true; }, 4000);
}

el.previewBtn.addEventListener('click', async () => {
  el.previewBtn.disabled = true;
  el.previewBtn.textContent = 'Memuat...';
  try {
    const res  = await fetch('/api/summary');
    const data = await res.json();

    if (data.total_frames > 0) {
      el.rptEngagement.textContent = `${safe(data.avg_engagement)}%`;
      el.rptFrames.textContent     = data.total_frames;
      el.rptLearning.textContent   = `${safe(data.learning_time_pct)}%`;
      el.reportStats.hidden        = false;
    } else {
      showMessage('Belum ada data sesi.', 'error');
    }
  } catch {
    showMessage('Gagal mengambil data.', 'error');
  } finally {
    el.previewBtn.disabled    = false;
    el.previewBtn.textContent = 'Lihat Ringkasan';
  }
});

el.downloadBtn.addEventListener('click', async () => {
  el.downloadBtn.disabled    = true;
  el.downloadBtn.textContent = 'Memproses...';
  try {
    const res  = await fetch('/api/summary');
    const data = await res.json();

    if (data.total_frames > 0) {
      // Download JSON — nanti bisa diganti PDF/CSV
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href     = url;
      a.download = `classwatch_report_${new Date().toISOString().slice(0,10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
      showMessage('Report berhasil diunduh.', 'success');
    } else {
      showMessage('Belum ada data sesi.', 'error');
    }
  } catch {
    showMessage('Gagal mengunduh report.', 'error');
  } finally {
    el.downloadBtn.disabled    = false;
    el.downloadBtn.textContent = 'Unduh Report';
  }
});

// ── Session control ───────────────────────────────────
let sessionRunning = false;
let sessionTick    = null;
let sessionSec     = 0;

const toggleBtn    = document.getElementById('session-toggle');
const btnLabel     = document.getElementById('session-btn-label');
const sessionBadge = document.getElementById('session-time');

function setSessionState(state) {
  toggleBtn.dataset.state = state;
  if (state === 'idle') {
    btnLabel.textContent   = 'Mulai Sesi';
    sessionRunning         = false;
    toggleBtn.disabled     = false;
    clearInterval(sessionTick);
  } else if (state === 'active') {
    btnLabel.textContent   = 'Hentikan Sesi';
    sessionRunning         = true;
    toggleBtn.disabled     = false;
  } else if (state === 'saving') {
    btnLabel.textContent   = 'Menyimpan...';
    toggleBtn.disabled     = true;
  }
}

function startSessionClock(fromSec = 0) {
  clearInterval(sessionTick);
  sessionSec = fromSec;
  sessionTick = setInterval(() => {
    sessionSec++;
    const h = String(Math.floor(sessionSec / 3600)).padStart(2, '0');
    const m = String(Math.floor((sessionSec % 3600) / 60)).padStart(2, '0');
    const s = String(sessionSec % 60).padStart(2, '0');
    sessionBadge.textContent = `${h}:${m}:${s}`;
  }, 1000);
}

toggleBtn.addEventListener('click', async () => {
  const camBox = document.querySelector('.cam-box');

  if (!sessionRunning) {
    // START
    try {
      const res = await fetch('/api/session/start', { method: 'POST' });
      if (!res.ok) throw new Error(await res.text());
      setSessionState('active');
      startSessionClock(0);
      camBox.classList.remove('session-idle');
      showMessage('Sesi pemantauan dimulai.', 'success');
    } catch (e) {
      console.error('[Session] Gagal start:', e);
      showMessage('Gagal memulai sesi.', 'error');
    }
  } else {
    // STOP
    setSessionState('saving');
    try {
      const res  = await fetch('/api/session/stop', { method: 'POST' });
      const data = await res.json();
      clearInterval(sessionTick);
      sessionBadge.textContent = '00:00:00';
      camBox.classList.add('session-idle');
      setSessionState('idle');
      showMessage(`Sesi disimpan: ${data.saved_as || '—'} (${data.total_frames} frame)`, 'success');
    } catch (e) {
      console.error('[Session] Gagal stop:', e);
      setSessionState('active');
      showMessage('Gagal menghentikan sesi.', 'error');
    }
  }
});

async function syncSessionState() {
  try {
    const res  = await fetch('/api/session/status');
    const data = await res.json();
    const camBox = document.querySelector('.cam-box');
    if (data.active) {
      setSessionState('active');
      startSessionClock(data.duration_seconds || 0);
      camBox.classList.remove('session-idle');
    } else {
      camBox.classList.add('session-idle');
    }
  } catch { /* server belum siap */ }
}

// ── Init ─────────────────────────────────────────────
startFpsCounter();
connect();
syncSessionState();