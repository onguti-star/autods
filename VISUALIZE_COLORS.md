# Light canvas + custom colors for Visualise

Two additions to the Visualise tab, both scoped to that tab only -- the rest
of the app is unaffected.

## Light canvas

A "\u2600 Light canvas" button in the Visualise ribbon switches the whole
Visualise workspace (KPI cards, tiles, filter/build panes) from dark to a
white background, and every chart's axis labels, gridlines and legend text
switch from light-on-dark to dark-on-light so they stay readable. It's
remembered across sessions (saved in the browser, not per-dataset).

## Vivid colours (saturated by default)

Charts are drawn with **saturated, high-contrast colours by default**, because
the original pastel palette plus translucent fills (e.g. a scatter drawn at
`#ef6f6faa`) washed out into the background — especially on the white Light
canvas. The Visualise ribbon's "◐ Vivid colours" button switches between that
default and the original calmer palette; the choice is remembered in the
browser (`localStorage`, key `autods.viz.contrast`).

What "vivid" means in practice:

- Two vivid palettes, one per canvas: bright hues on the dark canvas, deep
  saturated hues on the white Light canvas. A colour that pops off navy is far
  too pale on white, so `vizPalette()` picks the set that matches the canvas in
  use, and toggling the canvas re-draws with the other set.
- Series fills are painted **opaque** (`vizFill`) instead of at 60-70% alpha.
  Only fills that must stay see-through (area/radar/violin washes, `vizWash`)
  keep transparency, and even then stronger than before.
- Marks that can touch (scatter/bubble points, bars, line points) get a
  hairline edge in the canvas colour plus a slightly larger radius, so
  neighbouring marks stay visually separate instead of merging.
- Labels drawn inside a fill (treemap, violin median line) pick dark or white
  ink from the fill's lightness (`vizInkOn`) — previously they were hard-coded
  white, which vanished on the lighter hues.

Per-chart colour overrides still win over the palette: the pickers just start
from the vivid colours now. Switching either the canvas or the palette
re-draws every chart the Visualise tab has drawn (tiles, the suggestion
gallery and built visuals) from its own spec — maps and choropleths are left
alone since they colour by value gradient and fetch their own boundary data.

## Batch-predict preview no longer overflows its card

"Batch predict on another dataset" returns a preview row set that carries
*every* original column plus the prediction. The preview table is
`white-space:nowrap`, so a target dataset with many columns used to stretch it
wider than the model card and the numbers spilled outside the panel.

The table now lives in a `.batch-preview-wrap` scroll box
(`overflow:auto`, capped height, `max-width:100%`) with a short hint under it
pointing out that the table scrolls and how many columns it holds. The card
keeps its width no matter how many columns were scored, and the table scrolls
— horizontally and vertically, with the sticky header — inside its own box.

## Custom colors per chart

Charts that have a "🎨" button in their header (histogram, bar,
pie/donut, scatter, line, area, density, bubble, treemap, radar, violin,
word frequency, word cloud) can have their colors changed freely:
- Charts with one series (scatter, line, area, density, bubble) get a single
  color swatch.
- Charts with categories (pie, bar, treemap, word cloud, etc.) get one swatch
  per category, up to 8.
- "Reset to default colors" removes the override.

Chosen colors survive cross-filtering and are what gets included when you
download the HTML report or the notebook (see below) -- they're stored
directly on the same chart-spec objects already used for those downloads,
so no extra step is needed.

Not customizable: heatmap and choropleth (both use a gradient, not a fixed
color), the maps, and box plots in their normal (Q1/Median/Q3) mode.

## Downloads

- **HTML report**: the same color(s) you picked are baked into the
  downloaded file's own chart-drawing code.
- **Notebook (.ipynb)**: the same color(s) are passed to matplotlib's
  `color=` argument. Verified by actually executing the generated cells:
  a chosen color renders exactly on the right bar/point/line, including
  charts (like pie) that don't have their own matplotlib chart type in the
  notebook and fall back to a category bar chart -- colors are matched to
  categories by name there, not by position, so they stay correct even
  though that fallback recomputes and re-sorts the categories itself.

## Files touched
- `frontend/index.html` -- theme helpers, per-tile color picker UI, the
  ribbon toggle, `white-space` fix for multi-line assistant replies from the
  earlier "smarter assistant" change is unrelated and untouched here.
  Later pass: the two vivid palettes and the color helpers
  (`vizPalette`/`vizPaletteAt`/`vizFill`/`vizWash`/`vizInkOn`), every
  `drawChart` branch switched over to them, the "◐ Vivid colours" ribbon
  button, `vizSpecs` + `redrawVizCharts()` so a palette/canvas change
  re-draws gallery charts too, and the batch-predict preview scroll box.
- `backend/main_report.py` -- the downloaded HTML report's own embedded
  chart-drawing JS now reads the same `color`/`colors` fields, and its
  no-custom-colour fallbacks are the app's vivid palette (the light-canvas
  variant, since the report is a white page) so a downloaded chart matches
  what was on screen. The report's copy/notification script had string
  escapes (`\n`) that Python was turning into real newlines, which made that
  whole `<script>` block fail to parse and left the report's copy buttons
  dead; those are now escaped properly.
- `backend/nb.py` -- the generated notebook's matplotlib calls now read the
  same fields, including a label-name-based fallback for chart types nb.py
  renders generically (pie, treemap, radar, violin, density, heatmap all
  degrade to a category bar chart in the notebook, as they already did
  before this change -- only their coloring is new).

