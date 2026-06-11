/**
 * Execution Lab driver for the static (precomputed) deployment.
 *
 * Results are precomputed at build time by web/export.py into HTML fragments:
 *   {basePath}/lab/run/{policy}-r{n_reg}-t{T}-{date}.html
 *   {basePath}/lab/regimes/{split}-r{n_reg}.html
 * This file maps the configuration panel onto those URLs. The URL scheme
 * must stay in sync with web/export.py.
 */

document.addEventListener('DOMContentLoaded', function () {
  const configEl = document.getElementById('lab-config');
  if (!configEl) return;
  const config = JSON.parse(configEl.textContent);
  const basePath = config.basePath || '';

  const splitEl = document.getElementById('split');
  const nRegEl = document.getElementById('n_reg');
  const horizonEl = document.getElementById('horizon');
  const policyEl = document.getElementById('policy');
  const dateEl = document.getElementById('start_date');
  const dateHint = document.getElementById('start-date-hint');
  const runBtn = document.getElementById('run-all-btn');
  const regimesBtn = document.getElementById('regimes-btn');
  const overlay = document.getElementById('pipeline-overlay');
  const container = document.getElementById('results-container');

  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function maxStartIndex(T) {
    const idx = config.maxStartIndex[String(T)];
    return typeof idx === 'number' ? idx : config.dates.length - 1;
  }

  // Snap an arbitrary calendar date to the latest trading day on or before
  // it, clamped to the valid start range for the chosen horizon. Mirrors
  // the ffill logic of the dynamic server.
  function snapDate(value, T) {
    const dates = config.dates;
    const last = maxStartIndex(T);
    if (!value || value < dates[0]) return dates[0];
    if (value >= dates[last]) return dates[last];
    let lo = 0, hi = last;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (dates[mid] <= value) lo = mid; else hi = mid - 1;
    }
    return dates[lo];
  }

  function updateDateBounds() {
    const T = horizonEl.value;
    dateEl.min = config.dates[0];
    dateEl.max = config.dates[maxStartIndex(T)];
    if (dateHint) dateHint.textContent = 'Available: ' + dateEl.min + ' – ' + dateEl.max;
  }

  function updateRunAvailability() {
    const isTest = splitEl.value === 'test';
    runBtn.disabled = !isTest;
    runBtn.classList.toggle('opacity-50', !isTest);
    runBtn.classList.toggle('cursor-not-allowed', !isTest);
    runBtn.title = isTest ? '' : 'Episode rollout and benchmarks are available on the Test split';
  }

  function showOverlay(show) {
    if (!overlay) return;
    overlay.classList.toggle('hidden', !show);
    overlay.classList.toggle('flex', show);
  }

  function scrollToResults() {
    const wrapper = document.getElementById('results-wrapper');
    if (!wrapper) return;
    const y = wrapper.getBoundingClientRect().top + window.scrollY - 74;
    window.scrollTo({ top: Math.max(0, y), behavior: prefersReducedMotion ? 'auto' : 'smooth' });
  }

  // innerHTML does not execute inline scripts, so re-create each script tag
  // to trigger the chart render calls inside the fragment.
  function insertFragment(target, html) {
    target.innerHTML = html;
    target.querySelectorAll('script').forEach(function (old) {
      const fresh = document.createElement('script');
      fresh.textContent = old.textContent;
      old.replaceWith(fresh);
    });
  }

  function errorBox(message) {
    return '<div class="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700 text-sm">' + message + '</div>';
  }

  async function loadFragment(url, target) {
    target.classList.add('opacity-60');
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error('HTTP ' + res.status);
      insertFragment(target, await res.text());
    } catch (err) {
      insertFragment(target, errorBox('This configuration is not precomputed (' + err.message + '). Try a different date or horizon.'));
    } finally {
      target.classList.remove('opacity-60');
    }
  }

  // Once a full pipeline run replaces the placeholder panels, the regimes
  // button needs a panel of its own to render into.
  function regimeTarget() {
    let el = document.getElementById('regime-result');
    if (el) return el;
    container.innerHTML =
      '<div class="result-panel mb-6">' +
      '<h3 class="font-bold text-navy-900 mb-4">Regime Detection</h3>' +
      '<div id="regime-result"></div></div>';
    return document.getElementById('regime-result');
  }

  runBtn.addEventListener('click', async function () {
    const T = horizonEl.value;
    const date = snapDate(dateEl.value, T);
    dateEl.value = date;
    const url = basePath + '/lab/run/' + policyEl.value + '-r' + nRegEl.value + '-t' + T + '-' + date + '.html';
    showOverlay(true);
    try {
      await loadFragment(url, container);
    } finally {
      showOverlay(false);
    }
    scrollToResults();
  });

  regimesBtn.addEventListener('click', function () {
    const url = basePath + '/lab/regimes/' + splitEl.value + '-r' + nRegEl.value + '.html';
    loadFragment(url, regimeTarget());
  });

  splitEl.addEventListener('change', updateRunAvailability);
  horizonEl.addEventListener('change', updateDateBounds);
  dateEl.addEventListener('change', function () {
    dateEl.value = snapDate(dateEl.value, horizonEl.value);
  });

  updateDateBounds();
  updateRunAvailability();
});
