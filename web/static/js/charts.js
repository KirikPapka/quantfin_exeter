/**
 * Plotly chart builders for QuantFin Exeter.
 * Shared between case study (pre-computed) and run page (dynamic).
 */

const QF_COLORS = {
  navy:     '#0f172a',
  navyMid:  '#334e68',
  navyLight:'#829ab1',
  accent:   '#f59e0b',
  accentDk: '#d97706',
  calm:     'rgba(52, 211, 153, 0.25)',
  calmBord: 'rgba(52, 211, 153, 0.5)',
  vol:      'rgba(251, 191, 36, 0.25)',
  volBord:  'rgba(251, 191, 36, 0.5)',
  stressed: 'rgba(248, 113, 113, 0.25)',
  stressBord:'rgba(248, 113, 113, 0.5)',
  white:    '#ffffff',
  bg:       '#f8fafc',
};

const QF_LAYOUT = {
  font:      { family: 'Sora, system-ui, sans-serif', size: 12, color: '#334e68' },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  '#ffffff',
  margin:    { l: 56, r: 24, t: 40, b: 44 },
  xaxis:     { gridcolor: '#e2e8f0', linecolor: '#d9e2ec', zeroline: false },
  yaxis:     { gridcolor: '#e2e8f0', linecolor: '#d9e2ec', zeroline: false },
  hoverlabel:{ bgcolor: '#0f172a', font: { color: '#fff', size: 12 } },
  modebar:   { bgcolor: 'transparent', color: '#9fb3c8', activecolor: '#f59e0b' },
};

const REGIME_FILLS = [QF_COLORS.calm, QF_COLORS.vol, QF_COLORS.stressed];
const REGIME_NAMES = ['Calm', 'Elevated Volatility', 'Stressed'];

function _isCompactViewport() {
  return window.matchMedia('(max-width: 768px)').matches;
}


function _regimeShapes(dates, regimes) {
  const shapes = [];
  if (!regimes || regimes.length === 0) return shapes;
  let current = regimes[0], start = 0;
  for (let i = 1; i <= regimes.length; i++) {
    if (i === regimes.length || regimes[i] !== current) {
      shapes.push({
        type: 'rect',
        xref: 'x', yref: 'paper',
        x0: dates[start], x1: dates[Math.min(i, dates.length - 1)],
        y0: 0, y1: 1,
        fillcolor: REGIME_FILLS[current] || REGIME_FILLS[0],
        line: { width: 0 },
        layer: 'below',
      });
      if (i < regimes.length) {
        current = regimes[i];
        start = i;
      }
    }
  }
  return shapes;
}


/**
 * Render regime-overlaid price chart.
 * @param {string} containerId  - DOM element id
 * @param {Object} data         - { dates: [], close: [], regimes: [] }
 */
function renderRegimeChart(containerId, data) {
  if (!data || !data.dates) return;
  const compact = _isCompactViewport();
  const trace = {
    x: data.dates,
    y: data.close,
    type: 'scatter',
    mode: 'lines',
    name: 'SPY Close',
    line: { color: QF_COLORS.navy, width: 1.8 },
    hovertemplate: '%{x}<br>$%{y:.2f}<extra></extra>',
  };
  const layout = {
    ...QF_LAYOUT,
    title: { text: 'SPY Price with Market Regimes', font: { size: compact ? 12 : 14, color: '#102a43' }, x: 0.02 },
    margin: compact ? { l: 46, r: 10, t: 34, b: 36 } : QF_LAYOUT.margin,
    shapes: _regimeShapes(data.dates, data.regimes),
    xaxis: { ...QF_LAYOUT.xaxis, type: 'date', tickfont: { size: compact ? 10 : 12 } },
    yaxis: { ...QF_LAYOUT.yaxis, title: { text: 'Price ($)', standoff: 8 }, tickfont: { size: compact ? 10 : 12 } },
    showlegend: false,
    height: null,
  };
  Plotly.newPlot(containerId, [trace], layout, { responsive: true, displayModeBar: !compact, modeBarButtonsToRemove: ['lasso2d','select2d'] });
}


/**
 * Render episode inventory + action chart.
 * @param {string} containerId  - DOM element id
 * @param {Object} data         - { steps: [], dates: [], inventory: [], actions: [] }
 */
function renderEpisodeChart(containerId, data) {
  if (!data || !data.steps) return;
  const compact = _isCompactViewport();
  const useDates = data.dates && data.dates.length === data.steps.length &&
                   typeof data.dates[0] === 'string' && data.dates[0].match(/^\d{4}-\d{2}-\d{2}/);
  const xValues = useDates ? data.dates : data.steps;
  const xType   = useDates ? 'date' : 'linear';
  const xTitle  = useDates ? 'Date' : 'Step';

  const invTrace = {
    x: xValues,
    y: data.inventory,
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Inventory remaining',
    line: { color: QF_COLORS.navy, width: 2.5 },
    marker: { size: 6, color: QF_COLORS.navy },
    yaxis: 'y',
    hovertemplate: (useDates ? '%{x|%b %d, %Y}' : 'Step %{x}') + '<br>Inventory: %{y:.3f}<extra></extra>',
  };
  const actTrace = {
    x: xValues,
    y: data.actions,
    type: 'bar',
    name: 'Action fraction',
    marker: { color: QF_COLORS.accent, opacity: 0.7 },
    yaxis: 'y2',
    hovertemplate: (useDates ? '%{x|%b %d, %Y}' : 'Step %{x}') + '<br>Action: %{y:.3f}<extra></extra>',
  };
  const layout = {
    ...QF_LAYOUT,
    margin: compact ? { l: 44, r: 52, t: 48, b: 36 } : { ...QF_LAYOUT.margin, t: 52, r: 64, b: 48 },
    xaxis: {
      ...QF_LAYOUT.xaxis,
      title: { text: xTitle, standoff: 4 },
      tickfont: { size: compact ? 10 : 12 },
      type: xType,
      ...(useDates ? { tickangle: compact ? -24 : 0 } : { dtick: 1 }),
    },
    yaxis: {
      ...QF_LAYOUT.yaxis,
      title: { text: 'Inventory', font: { color: QF_COLORS.navy }, standoff: 10 },
      tickfont: { size: compact ? 10 : 12 },
      side: 'left',
      rangemode: 'tozero',
    },
    yaxis2: {
      title: { text: 'Action', font: { color: QF_COLORS.accentDk }, standoff: 10 },
      overlaying: 'y',
      side: 'right',
      rangemode: 'tozero',
      range: [0, 1.05],
      gridcolor: 'rgba(0,0,0,0)',
      tickfont: { size: compact ? 10 : 12 },
    },
    legend: compact
      ? {
          x: 0,
          y: 1.1,
          xanchor: 'left',
          yanchor: 'bottom',
          orientation: 'h',
          font: { size: 10 },
          bgcolor: 'rgba(255,255,255,0.9)',
          bordercolor: '#e2e8f0',
          borderwidth: 1,
        }
      : {
          x: 0.5,
          y: 1.04,
          xanchor: 'center',
          yanchor: 'bottom',
          orientation: 'h',
          font: { size: 11 },
          bgcolor: 'rgba(255,255,255,0.95)',
          bordercolor: '#e2e8f0',
          borderwidth: 1,
        },
    barmode: 'overlay',
    bargap: compact ? 0.22 : 0.3,
    height: null,
  };
  Plotly.newPlot(containerId, [invTrace, actTrace], layout, { responsive: true, displayModeBar: false });
}


/**
 * Render benchmark comparison bar chart.
 * @param {string} containerId  - DOM element id
 * @param {Object} data         - { strategies: [], mean_is: [], std_is: [] }
 */
function renderBenchmarkChart(containerId, data) {
  if (!data || !data.strategies) return;
  const compact = _isCompactViewport();
  const colors = data.strategies.map(s =>
    s === 'RL' ? QF_COLORS.accent : QF_COLORS.navyMid
  );
  const trace = {
    x: data.strategies,
    y: data.mean_is,
    error_y: {
      type: 'data',
      array: data.std_is,
      visible: true,
      color: '#9fb3c8',
      thickness: 1.5,
    },
    type: 'bar',
    marker: { color: colors, line: { width: 0 }, cornerradius: 4 },
    hovertemplate: '%{x}<br>Mean IS: %{y:.2f} bps<extra></extra>',
  };
  const layout = {
    ...QF_LAYOUT,
    title: { text: 'Mean Implementation Shortfall by Strategy', font: { size: compact ? 12 : 14, color: '#102a43' }, x: 0.02 },
    margin: compact ? { l: 44, r: 10, t: 36, b: 40 } : QF_LAYOUT.margin,
    xaxis: { ...QF_LAYOUT.xaxis, title: '', tickangle: compact ? -24 : 0, tickfont: { size: compact ? 10 : 12 } },
    yaxis: { ...QF_LAYOUT.yaxis, title: { text: 'IS (bps)', standoff: 8 }, tickfont: { size: compact ? 10 : 12 } },
    showlegend: false,
    bargap: compact ? 0.22 : 0.35,
    height: null,
  };
  Plotly.newPlot(containerId, [trace], layout, { responsive: true, displayModeBar: false });
}


/**
 * Render two side-by-side heatmaps: Mean IS (bps) and Completion Rate.
 * @param {string} isContainerId    - DOM element id for IS heatmap
 * @param {string} crContainerId    - DOM element id for Completion Rate heatmap
 * @param {Object} data             - { strategies: [], mean_is: [], completion_rates: [] }
 */
function renderBenchmarkHeatmaps(isContainerId, crContainerId, data) {
  if (!data || !data.strategies) return;
  const compact = _isCompactViewport();

  const strats = data.strategies;
  const isVals = data.mean_is || [];
  const crVals = (data.completion_rates || []).map(v => v * 100);

  const heatmapConfig = { responsive: true, displayModeBar: false };

  // --- Mean IS heatmap ---
  const isTrace = {
    x: strats,
    y: ['Mean IS (bps)'],
    z: [isVals],
    type: 'heatmap',
    // Red (low IS, worse for a sell) → green (high IS, better)
    colorscale: [
      [0, '#b91c1c'],
      [0.35, '#f97316'],
      [0.65, '#eab308'],
      [1, '#15803d'],
    ],
    showscale: true,
    colorbar: { thickness: compact ? 10 : 14, len: compact ? 0.74 : 0.8, title: { text: 'bps', side: 'right', font: { size: compact ? 10 : 11 } } },
    hovertemplate: '%{x}<br>Mean IS: %{z:.2f} bps<extra></extra>',
    text: [isVals.map(v => v.toFixed(1) + ' bps')],
    texttemplate: '%{text}',
    textfont: { size: compact ? 11 : 13, color: '#ffffff' },
  };
  const isLayout = {
    ...QF_LAYOUT,
    margin: compact ? { l: 12, r: 52, t: 40, b: 48 } : { l: 20, r: 80, t: 50, b: 60 },
    title: { text: 'Mean IS by Strategy', font: { size: compact ? 11 : 13, color: '#102a43' }, x: 0.02 },
    xaxis: { ...QF_LAYOUT.xaxis, tickangle: compact ? -26 : -20, tickfont: { size: compact ? 10 : 12 } },
    yaxis: { ...QF_LAYOUT.yaxis, showticklabels: false },
    height: compact ? 150 : 180,
  };
  Plotly.newPlot(isContainerId, [isTrace], isLayout, heatmapConfig);

  // --- Completion Rate heatmap ---
  if (!crVals.length) return;
  const crTrace = {
    x: strats,
    y: ['Completion (%)'],
    z: [crVals],
    type: 'heatmap',
    // Red (low completion) → green (high completion)
    colorscale: [
      [0, '#b91c1c'],
      [0.35, '#f97316'],
      [0.65, '#eab308'],
      [1, '#15803d'],
    ],
    zmin: 0,
    zmax: 100,
    showscale: true,
    colorbar: { thickness: compact ? 10 : 14, len: compact ? 0.74 : 0.8, title: { text: '%', side: 'right', font: { size: compact ? 10 : 11 } } },
    hovertemplate: '%{x}<br>Completion: %{z:.0f}%<extra></extra>',
    text: [crVals.map(v => v.toFixed(0) + '%')],
    texttemplate: '%{text}',
    textfont: { size: compact ? 11 : 13, color: '#ffffff' },
  };
  const crLayout = {
    ...QF_LAYOUT,
    margin: compact ? { l: 12, r: 52, t: 40, b: 48 } : { l: 20, r: 80, t: 50, b: 60 },
    title: { text: 'Completion Rate by Strategy', font: { size: compact ? 11 : 13, color: '#102a43' }, x: 0.02 },
    xaxis: { ...QF_LAYOUT.xaxis, tickangle: compact ? -26 : -20, tickfont: { size: compact ? 10 : 12 } },
    yaxis: { ...QF_LAYOUT.yaxis, showticklabels: false },
    height: compact ? 150 : 180,
  };
  Plotly.newPlot(crContainerId, [crTrace], crLayout, heatmapConfig);
}
