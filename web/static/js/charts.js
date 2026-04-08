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
  font:      { family: 'Inter, system-ui, sans-serif', size: 12, color: '#334e68' },
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
    title: { text: 'SPY Price with Market Regimes', font: { size: 14, color: '#102a43' }, x: 0.02 },
    shapes: _regimeShapes(data.dates, data.regimes),
    xaxis: { ...QF_LAYOUT.xaxis, type: 'date' },
    yaxis: { ...QF_LAYOUT.yaxis, title: { text: 'Price ($)', standoff: 8 } },
    showlegend: false,
    height: null,
  };
  Plotly.newPlot(containerId, [trace], layout, { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d','select2d'] });
}


/**
 * Render episode inventory + action chart.
 * @param {string} containerId  - DOM element id
 * @param {Object} data         - { steps: [], inventory: [], actions: [] }
 */
function renderEpisodeChart(containerId, data) {
  if (!data || !data.steps) return;
  const invTrace = {
    x: data.steps,
    y: data.inventory,
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Inventory',
    line: { color: QF_COLORS.navy, width: 2.5 },
    marker: { size: 6, color: QF_COLORS.navy },
    yaxis: 'y',
    hovertemplate: 'Step %{x}<br>Inventory: %{y:.3f}<extra></extra>',
  };
  const actTrace = {
    x: data.steps,
    y: data.actions,
    type: 'bar',
    name: 'Action (fraction)',
    marker: { color: QF_COLORS.accent, opacity: 0.7 },
    yaxis: 'y2',
    hovertemplate: 'Step %{x}<br>Action: %{y:.3f}<extra></extra>',
  };
  const layout = {
    ...QF_LAYOUT,
    title: { text: 'Execution Episode: Inventory & Actions', font: { size: 14, color: '#102a43' }, x: 0.02 },
    xaxis: { ...QF_LAYOUT.xaxis, title: { text: 'Step', standoff: 4 }, dtick: 1 },
    yaxis: {
      ...QF_LAYOUT.yaxis,
      title: { text: 'Inventory Remaining', font: { color: QF_COLORS.navy }, standoff: 8 },
      side: 'left',
      rangemode: 'tozero',
    },
    yaxis2: {
      title: { text: 'Action Fraction', font: { color: QF_COLORS.accentDk }, standoff: 8 },
      overlaying: 'y',
      side: 'right',
      rangemode: 'tozero',
      range: [0, 1],
      gridcolor: 'rgba(0,0,0,0)',
    },
    legend: { x: 0.02, y: 1.15, orientation: 'h', font: { size: 11 } },
    barmode: 'overlay',
    bargap: 0.3,
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
    title: { text: 'Mean Implementation Shortfall by Strategy', font: { size: 14, color: '#102a43' }, x: 0.02 },
    xaxis: { ...QF_LAYOUT.xaxis, title: '' },
    yaxis: { ...QF_LAYOUT.yaxis, title: { text: 'IS (bps)', standoff: 8 } },
    showlegend: false,
    bargap: 0.35,
    height: null,
  };
  Plotly.newPlot(containerId, [trace], layout, { responsive: true, displayModeBar: false });
}
