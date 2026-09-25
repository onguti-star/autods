# Light canvas + custom colors for Visualise

Two additions to the Visualise tab, both scoped to that tab only -- the rest
of the app is unaffected.

## Light canvas

A "\u2600 Light canvas" button in the Visualise ribbon switches the whole
Visualise workspace (KPI cards, tiles, filter/build panes) from dark to a
white background, and every chart's axis labels, gridlines and legend text
switch from light-on-dark to dark-on-light so they stay readable. It's
remembered across sessions (saved in the browser, not per-dataset).

## Custom colors per chart

Charts that have a "\ud83c\udfa8" button in their header (histogram, bar,
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
- `backend/main_report.py` -- the downloaded HTML report's own embedded
  chart-drawing JS now reads the same `color`/`colors` fields.
- `backend/nb.py` -- the generated notebook's matplotlib calls now read the
  same fields, including a label-name-based fallback for chart types nb.py
  renders generically (pie, treemap, radar, violin, density, heatmap all
  degrade to a category bar chart in the notebook, as they already did
  before this change -- only their coloring is new).
