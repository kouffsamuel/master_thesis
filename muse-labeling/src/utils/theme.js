// Theme palette for the <canvas> panels.
//
// The DOM chrome (topbar, cells, right panel) is themed with CSS variables in
// App.css. Canvas drawing calls can't read CSS variables, so every canvas
// colour lives here instead and is picked by theme name. Light-mode variants
// of the data palettes are derived by darkening the dark-mode hues — that keeps
// each track_id recognisably the same colour in both modes while staying
// readable on a white background.

const THEME_KEY = 'muse-theme'

export function loadTheme() {
  try {
    const t = localStorage.getItem(THEME_KEY)
    return t === 'light' || t === 'dark' ? t : 'dark'
  } catch { return 'dark' }
}

export function saveTheme(theme) {
  try { localStorage.setItem(THEME_KEY, theme) } catch { /* private mode */ }
}

// mix hex toward another hex; t=0 keeps hex, t=1 returns target
function mix(hex, target, t) {
  const ch = (h, i) => parseInt(h.slice(1 + i * 2, 3 + i * 2), 16)
  const out = i => Math.round(ch(hex, i) + (ch(target, i) - ch(hex, i)) * t)
    .toString(16).padStart(2, '0')
  return '#' + out(0) + out(1) + out(2)
}
const darken = (c, t = 0.3) => mix(c, '#000000', t)

// matplotlib tab20 — used for track_id colours
const TAB20 = [
  '#1f77b4','#aec7e8','#ff7f0e','#ffbb78','#2ca02c',
  '#98df8a','#d62728','#ff9896','#9467bd','#c5b0d5',
  '#8c564b','#c49c94','#e377c2','#f7b6d2','#7f7f7f',
  '#c7c7c7','#bcbd22','#dbdb8d','#17becf','#9edae5'
]
const PAIR_PALETTE = ['#f39c12','#e74c3c','#9b59b6','#1abc9c','#3498db','#e67e22','#16a085','#e91e63']

const DARK = {
  bg:        '#0d1117',   // canvas background (outside the axes)
  plot:      '#0d1117',   // plot area inside the square — flush with bg here
  frame:     '#22324a',   // square border
  grid:      '#1e2d3d',   // grid lines
  axis:      '#2c3e50',   // v = 0 axis
  tick:      '#555',      // tick + axis labels
  rdWhite:   255,         // RD map gray_r white point (low power) — see RDMapCanvas
  marker:    '#ffffff',   // selected / pending highlight
  sel:       '#e94560',   // selection ring
  candidate: '#f0c419',   // Stage 3 dashed ring
  obj:       '#2ecc71',
  noise:     '#7f8c8d',
  track:     TAB20,
  pair:      PAIR_PALETTE,
}

// Muted on purpose: a full-white plot area next to three other panels is
// glaring under long labelling sessions. bg sits a shade below the cell
// surface so each canvas reads as recessed; grid/axis are stepped down to
// keep the same contrast ratio they had against the dark background.
const LIGHT = {
  bg:        '#dfe4ec',
  plot:      '#c9d1de',   // a step below bg so the axes read as their own area
  frame:     '#8fa0b6',
  grid:      '#adb9ca',
  axis:      '#74839a',
  tick:      '#333c4a',   // axis numbers + "unconfirmed" — small text, keep it dark
  rdWhite:   224,   // dim the RD map's white (low-power) end to match bg
  marker:    '#111827',
  sel:       '#d02c4b',
  candidate: '#b8860b',
  obj:       darken('#2ecc71', 0.38),
  noise:     darken('#7f8c8d', 0.38),
  // 0.45, not the 0.32 that suited the markers alone: these same colours draw
  // the "ID n" labels, and tab20's pale half (#c7c7c7, #dbdb8d, #9edae5…)
  // was illegible as text on the light background at anything shallower.
  track:     TAB20.map(c => darken(c, 0.45)),
  pair:      PAIR_PALETTE.map(c => darken(c, 0.3)),
}

export function canvasTheme(theme) {
  return theme === 'light' ? LIGHT : DARK
}

// track_id -> colour (negative / unknown ids fall into the last slot)
export function trackColor(palette, trackId) {
  return palette.track[(trackId >= 0 ? trackId : 19) % 20]
}
