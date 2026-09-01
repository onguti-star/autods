"""
Visualization data layer — produces Chart.js-ready data for all chart types.
"""
import numpy as np
import pandas as pd

from . import nlp

# Hard cap on how many points a line/area chart will ever send to the
# browser. Sending every row of a 500k-row dataset crashes/freezes the tab
# (huge JSON payload + Chart.js curve smoothing over that many points), so
# large series get aggregated down to this many points instead. Trend shape
# is preserved by averaging y within evenly-sized x buckets rather than
# randomly dropping points.
MAX_LINE_POINTS = 3000


# ── helpers ────────────────────────────────────────────────────────────────

def _downsample_xy(sub: pd.DataFrame, x: str, y: str, max_points: int = MAX_LINE_POINTS):
    """
    Reduce a sorted (x, y) DataFrame to at most `max_points` rows so a line/
    area chart never has to render more than that many points. When x is
    numeric, points are aggregated into evenly-spaced x-buckets and averaged
    (keeps the trend shape instead of a misleading zig-zag from random
    sampling). When x is non-numeric (e.g. category/date-as-string), falls
    back to an evenly-spaced stride so the whole range is still represented.
    Returns (reduced_df, was_sampled, original_count, shown_count).
    """
    n = len(sub)
    if n <= max_points:
        return sub, False, n, n

    if pd.api.types.is_numeric_dtype(sub[x]):
        bin_idx = pd.cut(sub[x], bins=max_points, labels=False, duplicates="drop")
        agg = sub.groupby(bin_idx, observed=True).agg({x: "mean", y: "mean"}).reset_index(drop=True)
        return agg, True, n, len(agg)
    else:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        reduced = sub.iloc[idx].reset_index(drop=True)
        return reduced, True, n, len(reduced)


def _histogram_data(s: pd.Series, bins: int = 12,
                    x_min: float = None, x_max: float = None,
                    bin_width: float = None):
    clean = s.dropna()
    if x_min is not None:
        clean = clean[clean >= x_min]
    if x_max is not None:
        clean = clean[clean <= x_max]
    if len(clean) == 0:
        return {"labels": [], "values": []}
    if bin_width is not None:
        start = float(x_min) if x_min is not None else float(clean.min())
        end = float(x_max) if x_max is not None else float(clean.max())
        if start == end:
            end = start + bin_width
        edge_count = int(np.ceil((end - start) / bin_width)) + 1
        if edge_count > 501:
            raise ValueError("Bin width creates too many bars. Use a larger bin width or narrower X range.")
        edges = start + np.arange(edge_count) * bin_width
        if edges[-1] < end:
            edges = np.append(edges, edges[-1] + bin_width)
        counts, edges = np.histogram(clean, bins=edges)
    else:
        n_bins = min(bins, max(1, clean.nunique()))
        counts, edges = np.histogram(clean, bins=n_bins)
    labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(edges) - 1)]
    return {"labels": labels, "values": [int(c) for c in counts]}


def _bar_data(s: pd.Series, top_n: int = None):
    counts = s.value_counts(dropna=True)
    if top_n is not None:
        counts = counts.head(top_n)
    return {"labels": [str(i) for i in counts.index], "values": [int(v) for v in counts.values]}


def _scatter_data(df: pd.DataFrame, x: str, y: str, size_col: str = None, max_points: int = 600):
    cols = [c for c in [x, y, size_col] if c]
    sub = df[cols].dropna()
    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=42)
    points = []
    for _, r in sub.iterrows():
        pt = {"x": float(r[x]), "y": float(r[y])}
        if size_col:
            pt["r"] = float(r[size_col])
        points.append(pt)
    return {"points": points}


def _boxplot_data(df: pd.DataFrame, x: str, group_col: str = None):
    """Returns per-group box stats: min, q1, median, q3, max, outliers."""
    def stats(s):
        s = s.dropna()
        if len(s) == 0:
            return None
        q1, med, q3 = float(s.quantile(0.25)), float(s.median()), float(s.quantile(0.75))
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = [float(v) for v in s[(s < lo) | (s > hi)]]
        return {"min": float(s[s >= lo].min()), "q1": q1, "median": med,
                "q3": q3, "max": float(s[s <= hi].max()), "outliers": outliers[:50]}

    if group_col and group_col in df.columns:
        groups = df[group_col].dropna().unique()[:12]
        result = {}
        for g in groups:
            st = stats(df[df[group_col] == g][x])
            if st:
                result[str(g)] = st
        return {"groups": result, "grouped": True}
    else:
        st = stats(df[x])
        return {"groups": {"all": st} if st else {}, "grouped": False}


def _line_data(df: pd.DataFrame, x: str, y: str):
    sub = df[[x, y]].dropna().sort_values(x)
    sub, sampled, total_points, shown_points = _downsample_xy(sub, x, y)
    if pd.api.types.is_numeric_dtype(sub[x]):
        xs = [float(v) for v in sub[x]]
    else:
        xs = [str(v) for v in sub[x]]
    ys = [float(v) for v in sub[y]]
    return {
        "labels": xs, "values": ys,
        "sampled": sampled, "total_points": total_points, "shown_points": shown_points,
    }


def _pie_data(s: pd.Series, top_n: int = 10):
    counts = s.value_counts(dropna=True)
    top = counts.head(top_n)
    rest = int(counts.iloc[top_n:].sum()) if len(counts) > top_n else 0
    labels = [str(i) for i in top.index]
    values = [int(v) for v in top.values]
    if rest > 0:
        labels.append("Other")
        values.append(rest)
    return {"labels": labels, "values": values}


def _heatmap_data(matrix: list[list[float]], labels: list[str]) -> dict:
    """Generate data for correlation heatmap."""
    return {
        "type": "heatmap",
        "matrix": matrix,
        "labels": labels
    }


def _treemap_data(s: pd.Series, top_n: int = 15) -> dict:
    """Generate data for treemap - hierarchical view of categorical data."""
    counts = s.value_counts(dropna=True).head(top_n)
    labels = [str(i) for i in counts.index]
    values = [int(v) for v in counts.values]
    return {"type": "treemap", "labels": labels, "values": values}


def _radar_data(df: pd.DataFrame, cols: list[str], group_col: str = None) -> dict:
    """Generate data for radar chart - compare numeric columns."""
    if not cols:
        raise ValueError("No numeric columns provided for radar chart.")
    
    # Limit to top 8 columns for readability
    cols = cols[:8]
    
    if group_col and group_col in df.columns:
        # Group by categorical column
        groups = df[group_col].dropna().unique()[:5]
        datasets = []
        for g in groups:
            group_df = df[df[group_col] == g]
            values = [float(group_df[c].mean()) if not pd.isna(group_df[c].mean()) else 0 for c in cols]
            datasets.append({
                "label": str(g),
                "data": values
            })
        return {"type": "radar", "labels": cols, "datasets": datasets}
    else:
        # Single dataset - show mean of each column
        values = [float(df[c].mean()) if not pd.isna(df[c].mean()) else 0 for c in cols]
        return {"type": "radar", "labels": cols, "datasets": [{"label": "Mean", "data": values}]}


def _area_data(df: pd.DataFrame, x: str, y: str) -> dict:
    """Generate data for area chart - cumulative trends."""
    sub = df[[x, y]].dropna().sort_values(x)
    sub, sampled, total_points, shown_points = _downsample_xy(sub, x, y)
    if pd.api.types.is_numeric_dtype(sub[x]):
        xs = [float(v) for v in sub[x]]
    else:
        xs = [str(v) for v in sub[x]]
    ys = [float(v) for v in sub[y]]
    return {
        "type": "area", "labels": xs, "values": ys,
        "sampled": sampled, "total_points": total_points, "shown_points": shown_points,
    }


def _split_cols(value: str | None) -> list[str]:
    """Split a comma-separated column list (sent by the frontend's Y
    multi-select) into a clean list of column names."""
    if not value:
        return []
    return [c.strip() for c in value.split(",") if c.strip()]


def _multi_series_xy(df: pd.DataFrame, x: str, ys: list[str], max_points: int = MAX_LINE_POINTS) -> dict:
    """
    Build one aligned series per Y column so several numeric columns (e.g.
    separate 'male' and 'female' columns) can be compared against the same
    X-axis on one chart. Rows sharing an X value are aggregated (mean) so
    each X tick maps to exactly one point per series.
    """
    cols = [x] + ys
    sub = df[cols].dropna(subset=[x])
    sub = sub.dropna(subset=ys, how="all")
    if sub.empty:
        return {
            "labels": [], "series": [{"name": y, "values": []} for y in ys],
            "sampled": False, "total_points": 0, "shown_points": 0,
        }

    grouped = sub.groupby(x, as_index=False, sort=True)[ys].mean()
    total_points = len(grouped)
    sampled = False

    if total_points > max_points:
        sampled = True
        if pd.api.types.is_numeric_dtype(grouped[x]):
            bin_idx = pd.cut(grouped[x], bins=max_points, labels=False, duplicates="drop")
            grouped = grouped.groupby(bin_idx, observed=True).mean().reset_index(drop=True)
        else:
            idx = np.linspace(0, total_points - 1, max_points).astype(int)
            grouped = grouped.iloc[idx].reset_index(drop=True)

    if pd.api.types.is_numeric_dtype(grouped[x]):
        xs = [float(v) for v in grouped[x]]
    else:
        xs = [str(v) for v in grouped[x]]

    series = [
        {"name": y, "values": [None if pd.isna(v) else float(v) for v in grouped[y]]}
        for y in ys
    ]
    return {
        "labels": xs, "series": series,
        "sampled": sampled, "total_points": total_points, "shown_points": len(grouped),
    }


def _grouped_series_xy(df: pd.DataFrame, x: str, y: str, group_col: str,
                        max_points: int = MAX_LINE_POINTS, max_series: int = 12) -> dict:
    """
    Split a single numeric Y column into one series per distinct value of
    group_col, aligned on a shared X-axis — mirrors Power BI's "Legend"
    field (e.g. one line per 'Indicator_name', X-axis 'Year', Y-axis
    'Value'). Rows sharing the same (x, group) pair are summed, matching
    Power BI's default "Sum of Value" aggregation.
    """
    cols = [x, y, group_col]
    sub = df[cols].dropna(subset=[x, y, group_col])
    if sub.empty:
        return {"labels": [], "series": [], "sampled": False, "total_points": 0, "shown_points": 0, "truncated_groups": False}

    value_counts = sub[group_col].value_counts()
    top_groups = value_counts.head(max_series).index.tolist()
    truncated_groups = len(value_counts) > max_series
    sub = sub[sub[group_col].isin(top_groups)]

    pivot = sub.groupby([x, group_col], as_index=False)[y].sum()
    wide = pivot.pivot(index=x, columns=group_col, values=y).sort_index()

    total_points = len(wide)
    sampled = False
    if total_points > max_points:
        sampled = True
        if pd.api.types.is_numeric_dtype(wide.index):
            bin_idx = pd.cut(wide.index.to_series(index=wide.index), bins=max_points, labels=False, duplicates="drop")
            wide = wide.groupby(bin_idx, observed=True).mean()
        else:
            idx = np.linspace(0, total_points - 1, max_points).astype(int)
            wide = wide.iloc[idx]

    if pd.api.types.is_numeric_dtype(wide.index):
        xs = [float(v) for v in wide.index]
    else:
        xs = [str(v) for v in wide.index]

    series = []
    for g in top_groups:
        if g not in wide.columns:
            continue
        vals = wide[g]
        series.append({"name": str(g), "values": [None if pd.isna(v) else float(v) for v in vals]})

    return {
        "labels": xs, "series": series,
        "sampled": sampled, "total_points": total_points, "shown_points": len(wide),
        "truncated_groups": truncated_groups,
    }


def _violin_data(df: pd.DataFrame, x: str, group_col: str = None) -> dict:
    """Generate data for violin plot - distribution with density."""
    def compute_violin_stats(s):
        s = s.dropna()
        if len(s) == 0:
            return None
        
        # Create bins for violin shape
        try:
            bins = min(20, max(5, len(s) // 10))
            counts, edges = np.histogram(s, bins=bins)
            # Normalize counts
            max_count = max(counts) if max(counts) > 0 else 1
            normalized = [c / max_count for c in counts]
            
            return {
                "bins": [float((edges[i] + edges[i+1]) / 2) for i in range(len(edges) - 1)],
                "widths": normalized,
                "min": float(s.min()),
                "q1": float(s.quantile(0.25)),
                "median": float(s.median()),
                "q3": float(s.quantile(0.75)),
                "max": float(s.max())
            }
        except Exception:
            return None
    
    if group_col and group_col in df.columns:
        groups = df[group_col].dropna().unique()[:8]
        result = {}
        for g in groups:
            st = compute_violin_stats(df[df[group_col] == g][x])
            if st:
                result[str(g)] = st
        return {"type": "violin", "groups": result, "grouped": True}
    else:
        st = compute_violin_stats(df[x])
        return {"type": "violin", "groups": {"all": st} if st else {}, "grouped": False}


def _density_data(s: pd.Series) -> dict:
    """Generate data for density plot - smoothed distribution."""
    clean = s.dropna()
    if len(clean) < 3:
        return {"type": "density", "labels": [], "values": []}
    
    try:
        import numpy as np
        from scipy import stats
        
        # Create kernel density estimate
        kde = stats.gaussian_kde(clean)
        x_range = np.linspace(clean.min(), clean.max(), 100)
        density = kde(x_range)
        
        return {
            "type": "density",
            "labels": [float(x) for x in x_range],
            "values": [float(d) for d in density]
        }
    except ImportError:
        # Fallback to histogram if scipy not available
        return {"type": "density", "labels": [], "values": [], "fallback": "histogram"}


def _detect_coordinate_columns(df: pd.DataFrame) -> dict:
    """Detect latitude and longitude columns in the dataframe."""
    lat_candidates = []
    lon_candidates = []

    LAT_NAMES = {'lat', 'latitude', 'latitud', 'breite', 'lat_deg', 'y_coord', 'ylat'}
    LON_NAMES = {'lon', 'lng', 'longitude', 'longitud', 'laenge', 'lon_deg', 'x_coord', 'xlon', 'long'}

    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in LAT_NAMES:
            lat_candidates.append(col)
        elif col_lower in LON_NAMES:
            lon_candidates.append(col)
        # Only auto-detect from value ranges for columns with geo-sounding names
        # Avoid grabbing generic 'x', 'y' or any numeric column by value alone
        elif pd.api.types.is_numeric_dtype(df[col]) and any(hint in col_lower for hint in ('lat', 'lon', 'lng', 'coord', 'geo')):
            values = df[col].dropna()
            if len(values) > 0:
                if values.min() >= -90 and values.max() <= 90:
                    lat_candidates.append(col)
                elif values.min() >= -180 and values.max() <= 180:
                    lon_candidates.append(col)

    return {
        "lat": lat_candidates[0] if lat_candidates else None,
        "lon": lon_candidates[0] if lon_candidates else None
    }


def _scatter_map_data(df: pd.DataFrame, lat_col: str, lon_col: str,
                      size_col: str = None, color_col: str = None,
                      label_col: str = None, max_points: int = 2000) -> dict:
    """Generate data for scatter map — includes popup labels for all non-geo cols."""
    geo_cols = {lat_col, lon_col}
    needed = [lat_col, lon_col]
    if size_col and size_col not in geo_cols:
        needed.append(size_col)
    if color_col and color_col not in geo_cols:
        needed.append(color_col)

    # Pick a label column: prefer explicit, then a text/name column, then nothing
    if not label_col:
        for c in df.columns:
            if c in geo_cols:
                continue
            if df[c].dtype == object or str(df[c].dtype) == 'string':
                label_col = c
                break

    # All extra columns to include in the popup (up to 6)
    popup_cols = [c for c in df.columns if c not in geo_cols][:6]
    for c in popup_cols:
        if c not in needed:
            needed.append(c)

    sub = df[needed].dropna(subset=[lat_col, lon_col])
    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=42)

    # Build color scale if color_col is numeric
    color_map = {}
    if color_col and color_col in sub.columns:
        if pd.api.types.is_numeric_dtype(sub[color_col]):
            mn, mx = sub[color_col].min(), sub[color_col].max()
            rng = mx - mn if mx != mn else 1
            # map 0→1 to a cyan→orange gradient
            def _hex(v):
                t = (v - mn) / rng
                r = int(255 * t)
                g = int(180 * (1 - t) + 212 * t)
                b = int(214 * (1 - t) + 0 * t)
                return f"#{r:02x}{g:02x}{b:02x}"
            color_map = {i: _hex(row[color_col]) for i, row in sub.iterrows()}

    points = []
    for i, r in sub.iterrows():
        pt = {"lat": float(r[lat_col]), "lon": float(r[lon_col])}
        if size_col and size_col in r.index and pd.notna(r[size_col]):
            pt["size"] = float(r[size_col])
        if color_col and color_col in r.index:
            if color_map:
                pt["color"] = color_map.get(i, "#5fd4d6")
            else:
                pt["color"] = str(r[color_col])
        # Popup: all extra columns as key→value pairs
        popup = {}
        for c in popup_cols:
            if c in r.index and pd.notna(r[c]):
                popup[c] = str(r[c]) if not isinstance(r[c], (int, float)) else round(float(r[c]), 4)
        if popup:
            pt["popup"] = popup
        points.append(pt)

    return {
        "type": "scatter_map",
        "points": points,
        "lat_col": lat_col,
        "lon_col": lon_col,
        "popup_cols": popup_cols,
    }


def suggest_choropleth_columns(df: pd.DataFrame) -> list[dict]:
    """For a GeoJSON-backed dataframe, suggest numeric columns to choropleth by.
    Returns list of {col, title, reason} dicts — one per good numeric column."""
    # Skip internal/geo columns
    skip = {'_geometry_type', '_feature_index'}
    numeric_cols = [c for c in df.select_dtypes(include='number').columns if c not in skip]
    # Prefer columns that look like meaningful measures
    priority_hints = ('pop', 'gdp', 'income', 'area', 'density', 'count',
                      'rate', 'index', 'score', 'total', 'value', 'hdi', 'pct', 'percent')
    priority = [c for c in numeric_cols if any(h in c.lower() for h in priority_hints)]
    rest = [c for c in numeric_cols if c not in priority]
    ordered = priority + rest

    suggestions = []
    for col in ordered[:6]:            # at most 6 suggestions
        non_null = df[col].dropna()
        if len(non_null) < 2:
            continue
        suggestions.append({
            "col": col,
            "title": f"Choropleth — {col}",
            "reason": f"Colour each region by {col} (range: {non_null.min():.3g} – {non_null.max():.3g}).",
        })
    return suggestions
    """Generate data for choropleth map (colored regions)."""
    return {
        "type": "choropleth",
        "geojson": geojson,
        "region_id_col": region_id_col,
        "value_col": value_col,
        "data": df[[region_id_col, value_col]].dropna().to_dict('records')
    }


# ── public API ──────────────────────────────────────────────────────────────

def chart_data(df: pd.DataFrame, x: str, chart_type: str,
               y: str = None, group: str = None,
               x_min: float = None, x_max: float = None,
               bar_limit: int = None, bin_width: float = None) -> dict:
    if x not in df.columns:
        raise ValueError(f"Column '{x}' not found.")
    if x_min is not None and x_max is not None and x_min > x_max:
        raise ValueError("X-axis minimum must be less than or equal to the maximum.")
    if bin_width is not None and bin_width <= 0:
        raise ValueError("Bin width must be greater than zero.")
    s = df[x]

    if chart_type == "histogram":
        if not pd.api.types.is_numeric_dtype(s):
            raise ValueError(f"'{x}' is not numeric — try a bar chart instead.")
        hist = _histogram_data(s, x_min=x_min, x_max=x_max, bin_width=bin_width)
        n_shown = int(sum(hist["values"])) if hist["values"] else 0
        return {
            "type": "histogram",
            "x": x,
            "x_min": x_min,
            "x_max": x_max,
            "bin_width": bin_width,
            "x_label": x,
            "y_label": "Count",
            "caption": (
                f"Distribution of '{x}' across {n_shown:,} values, split into "
                f"{len(hist['labels'])} bars. Bar height = how many rows fall in that range."
            ),
            **hist,
        }

    if chart_type == "bar":
        # Legend/series split (Power BI style): one set of bars per group_col
        # value, e.g. compare 'male' vs 'female' Value across 'Year'.
        if group and y:
            if group not in df.columns:
                raise ValueError(f"Column '{group}' not found.")
            if y not in df.columns:
                raise ValueError(f"Column '{y}' not found.")
            if not pd.api.types.is_numeric_dtype(df[y]):
                raise ValueError(f"'{y}' must be numeric to compare on a bar chart.")
            multi = _grouped_series_xy(df, x, y, group, max_points=200)
            caption = f"Sum of '{y}' for each '{x}', split by '{group}'."
            if multi["truncated_groups"]:
                caption += f" Showing the {len(multi['series'])} largest '{group}' groups."
            return {"type": "bar", "x": x, "y": y, "group": group,
                    "x_label": x, "y_label": y, "caption": caption, **multi}

        # Wide-format compare: two or more separate numeric Y columns
        # plotted as grouped bars against the same X.
        ys = _split_cols(y)
        if ys:
            for col in ys:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found.")
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValueError(f"'{col}' must be numeric to compare on a bar chart.")
            multi = _multi_series_xy(df, x, ys, max_points=200)
            caption = f"Average of {', '.join(ys)} for each '{x}'."
            if multi["sampled"]:
                caption += f" (Grouped into {multi['shown_points']:,} '{x}' buckets from {multi['total_points']:,} distinct values.)"
            return {"type": "bar", "x": x, "y_columns": ys,
                    "x_label": x, "y_label": "Average value", "caption": caption, **multi}

        top_n = min(max(int(bar_limit), 1), 1000) if bar_limit is not None else None
        bar = _bar_data(s, top_n=top_n)
        total_categories = s.nunique(dropna=True)
        caption = f"How often each value of '{x}' occurs, most frequent first."
        if top_n is not None and total_categories > len(bar["labels"]):
            caption += f" Showing top {len(bar['labels'])} of {total_categories:,} distinct values."
        return {
            "type": "bar", "x": x, "bar_limit": top_n,
            "x_label": x, "y_label": "Count", "caption": caption,
            **bar,
        }

    if chart_type == "pie":
        pie = _pie_data(s)
        return {
            "type": "pie", "x": x,
            "caption": f"Share of each category in '{x}' — slice size = proportion of rows.",
            **pie,
        }

    if chart_type == "scatter":
        if not y or y not in df.columns:
            raise ValueError("Scatter needs a Y column.")
        if not pd.api.types.is_numeric_dtype(s) or not pd.api.types.is_numeric_dtype(df[y]):
            raise ValueError("Scatter needs two numeric columns.")
        scatter = _scatter_data(df, x, y)
        total_rows = len(df[[x, y]].dropna())
        caption = f"Each dot is one row — '{x}' on the X-axis vs '{y}' on the Y-axis."
        if total_rows > len(scatter["points"]):
            caption += f" Showing a random sample of {len(scatter['points']):,} of {total_rows:,} rows."
        return {"type": "scatter", "x": x, "y": y, "x_label": x, "y_label": y, "caption": caption, **scatter}

    if chart_type == "bubble":
        if not y or y not in df.columns:
            raise ValueError("Bubble chart needs X, Y and a Size column.")
        if not group or group not in df.columns:
            raise ValueError("Bubble chart needs a Size column (use the 'group/size' field).")
        for col in [x, y, group]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"'{col}' must be numeric for a bubble chart.")
        # normalise size to 3–30 range
        sub = df[[x, y, group]].dropna()
        total_rows = len(sub)
        if len(sub) > 400:
            sub = sub.sample(400, random_state=42)
        sz = sub[group]
        sz_norm = 3 + 27 * (sz - sz.min()) / (sz.max() - sz.min() + 1e-9)
        pts = [{"x": float(r[x]), "y": float(r[y]), "r": float(sz_norm.iloc[i])}
               for i, (_, r) in enumerate(sub.iterrows())]
        caption = f"'{x}' vs '{y}', with bubble size showing '{group}'."
        if total_rows > len(pts):
            caption += f" Showing a random sample of {len(pts):,} of {total_rows:,} rows."
        return {
            "type": "bubble", "x": x, "y": y, "size": group,
            "x_label": x, "y_label": y, "caption": caption, "points": pts,
        }

    if chart_type == "boxplot":
        if not pd.api.types.is_numeric_dtype(s):
            raise ValueError(f"'{x}' must be numeric for a box plot.")
        caption = f"Spread of '{x}': box = middle 50% of values, line = median, dots = outliers."
        if group:
            caption += f" Grouped by '{group}'."
        return {
            "type": "boxplot", "x": x, "group": group,
            "x_label": x, "y_label": x, "caption": caption,
            **_boxplot_data(df, x, group),
        }

    if chart_type == "line":
        # Legend/series split (Power BI style): one line per group_col value,
        # e.g. 'Indicator_name' = male vs female, X-axis 'Year', Y-axis 'Value'.
        if group and y:
            if group not in df.columns:
                raise ValueError(f"Column '{group}' not found.")
            if y not in df.columns:
                raise ValueError(f"Column '{y}' not found.")
            if not pd.api.types.is_numeric_dtype(df[y]):
                raise ValueError(f"'{y}' must be numeric for a line chart.")
            multi = _grouped_series_xy(df, x, y, group)
            caption = f"'{y}' over '{x}', split by '{group}'."
            if multi["truncated_groups"]:
                caption += f" Showing the {len(multi['series'])} largest '{group}' groups."
            if multi["sampled"]:
                caption += f" (Showing {multi['shown_points']:,} points aggregated from {multi['total_points']:,} rows.)"
            return {"type": "line", "x": x, "y": y, "group": group,
                    "x_label": x, "y_label": y, "caption": caption, **multi}

        # Wide-format compare: two or more separate numeric Y columns (e.g.
        # 'male' and 'female' columns) plotted as separate lines over X.
        ys = _split_cols(y)
        if not ys:
            raise ValueError("Line chart needs a Y column.")
        for col in ys:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found.")
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"'{col}' must be numeric for a line chart.")

        if len(ys) == 1:
            line = _line_data(df, x, ys[0])
            if line["sampled"]:
                caption = (
                    f"'{ys[0]}' over '{x}' — showing {line['shown_points']:,} points aggregated "
                    f"(averaged) from {line['total_points']:,} rows to keep the chart readable and fast."
                )
            else:
                caption = f"'{ys[0]}' plotted over '{x}' for all {line['total_points']:,} rows."
            return {"type": "line", "x": x, "y": ys[0], "x_label": x, "y_label": ys[0], "caption": caption, **line}

        multi = _multi_series_xy(df, x, ys)
        caption = f"Comparing {', '.join(ys)} over '{x}'."
        if multi["sampled"]:
            caption += f" (Showing {multi['shown_points']:,} points aggregated from {multi['total_points']:,} rows.)"
        return {"type": "line", "x": x, "y_columns": ys, "x_label": x, "y_label": "Value", "caption": caption, **multi}


    if chart_type == "word_frequency":
        top = nlp.word_frequency(s, top_n=25)
        if not top:
            raise ValueError(f"'{x}' has no usable text to build a word frequency chart from.")
        return {"type": "word_frequency", "x": x,
                "labels": [w["word"] for w in top], "values": [w["count"] for w in top]}

    if chart_type == "wordcloud":
        top = nlp.word_frequency(s, top_n=60)
        if not top:
            raise ValueError(f"'{x}' has no usable text to build a word cloud from.")
        return {"type": "wordcloud", "x": x, "words": top}

    if chart_type == "heatmap":
        if not y or y not in df.columns:
            raise ValueError("Heatmap needs a Y column (correlation matrix).")
        # Create correlation matrix
        corr = df[[x, y]].corr()
        matrix = corr.values.tolist()
        labels = [x, y]
        return _heatmap_data(matrix, labels)

    if chart_type == "treemap":
        return _treemap_data(s)

    if chart_type == "radar":
        ys = _split_cols(y)
        if not ys:
            raise ValueError("Radar chart needs numeric columns.")
        cols = [x] + ys
        seen = set()
        deduped = []
        for c in cols:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and c not in seen:
                seen.add(c)
                deduped.append(c)
        if not deduped:
            raise ValueError("Radar chart needs at least one numeric column.")
        return _radar_data(df, deduped, group)

    if chart_type == "area":
        # Legend/series split (Power BI style): one filled series per
        # group_col value, e.g. male vs female Value over Year.
        if group and y:
            if group not in df.columns:
                raise ValueError(f"Column '{group}' not found.")
            if y not in df.columns:
                raise ValueError(f"Column '{y}' not found.")
            if not pd.api.types.is_numeric_dtype(df[y]):
                raise ValueError(f"'{y}' must be numeric for an area chart.")
            multi = _grouped_series_xy(df, x, y, group)
            caption = f"'{y}' over '{x}', split by '{group}' (area)."
            if multi["truncated_groups"]:
                caption += f" Showing the {len(multi['series'])} largest '{group}' groups."
            if multi["sampled"]:
                caption += f" (Showing {multi['shown_points']:,} points aggregated from {multi['total_points']:,} rows.)"
            return {"type": "area", "x": x, "y": y, "group": group,
                    "x_label": x, "y_label": y, "caption": caption, **multi}

        ys = _split_cols(y)
        if not ys:
            raise ValueError("Area chart needs a Y column.")
        for col in ys:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found.")
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"'{col}' must be numeric for an area chart.")

        if len(ys) == 1:
            area = _area_data(df, x, ys[0])
            if area["sampled"]:
                caption = (
                    f"'{ys[0]}' over '{x}' — showing {area['shown_points']:,} points aggregated "
                    f"(averaged) from {area['total_points']:,} rows to keep the chart readable and fast."
                )
            else:
                caption = f"'{ys[0]}' plotted over '{x}' for all {area['total_points']:,} rows."
            return {"x": x, "y": ys[0], "x_label": x, "y_label": ys[0], "caption": caption, **area}

        multi = _multi_series_xy(df, x, ys)
        caption = f"Comparing {', '.join(ys)} over '{x}' (area)."
        if multi["sampled"]:
            caption += f" (Showing {multi['shown_points']:,} points aggregated from {multi['total_points']:,} rows.)"
        return {"type": "area", "x": x, "y_columns": ys, "x_label": x, "y_label": "Value", "caption": caption, **multi}

    if chart_type == "violin":
        if not pd.api.types.is_numeric_dtype(s):
            raise ValueError(f"'{x}' must be numeric for a violin plot.")
        return _violin_data(df, x, group)

    if chart_type == "density":
        if not pd.api.types.is_numeric_dtype(s):
            raise ValueError(f"'{x}' must be numeric for a density plot.")
        return _density_data(s)

    if chart_type == "scatter_map":
        if not y or y not in df.columns:
            raise ValueError("Scatter map needs latitude and longitude columns.")
        return _scatter_map_data(df, x, y, group)

    if chart_type == "choropleth":
        raise ValueError("Choropleth maps require GeoJSON data. Use the GeoJSON upload option.")

    raise ValueError(f"Unknown chart type '{chart_type}'.")


def suggest_visuals(df: pd.DataFrame, max_suggestions: int = 3, has_geojson: bool = False) -> list:
    suggestions = []

    # ── Choropleth suggestions (highest priority when GeoJSON is present) ──────
    if has_geojson:
        # Find the best name column (longest non-numeric string column)
        skip_cols = {'_geometry_type', '_feature_index'}
        # Detect text columns regardless of whether they're object, string, or category
        str_cols = [
            c for c in df.columns
            if c not in skip_cols and
            str(df[c].dtype) not in ('int8','int16','int32','int64',
                                     'Int8','Int16','Int32','Int64',
                                     'float32','float64','bool','boolean') and
            not str(df[c].dtype).startswith('float') and
            not str(df[c].dtype).startswith('int')
        ]
        PRIORITY_HINTS = ('admin', 'name', 'country', 'region', 'county',
                          'province', 'state', 'district', 'city', 'sovereignt')

        def _name_score(c):
            hint_score = 3 if any(h in c.lower() for h in PRIORITY_HINTS) else 0
            try:
                non_null = df[c].dropna()
                # For category/string dtype, also filter out empty strings
                if hasattr(non_null, 'astype'):
                    non_null = non_null.astype(str)
                    non_null = non_null[non_null.str.strip() != '']
                    non_null = non_null[non_null != 'nan']
                fill_score = len(non_null) / max(len(df), 1)
            except Exception:
                fill_score = 0
            return hint_score + fill_score

        name_col = max(str_cols, key=_name_score) if str_cols else None

        # Rows to send with each choropleth spec (feature_index + value + name)
        keep_cols = [c for c in ['_feature_index', name_col] if c] if name_col else ['_feature_index']

        for spec in suggest_choropleth_columns(df):
            row_cols = keep_cols + [spec["col"]]
            rows = df[row_cols].dropna(subset=[spec["col"]]).to_dict('records')
            suggestions.append({
                "title":    spec["title"],
                "reason":   spec["reason"],
                "type":     "choropleth",
                "value_col": spec["col"],
                "name_col": name_col,
                "rows":     rows,
            })
        if len(suggestions) >= max_suggestions:
            return suggestions[:max_suggestions]
        max_suggestions -= len(suggestions)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    remaining_cols = [c for c in df.columns if c not in numeric_cols]
    text_cols = [c for c in remaining_cols if nlp.is_text_column(df[c])]
    categorical_cols = [c for c in remaining_cols if c not in text_cols]

    # 1. Strongest correlated pair → scatter
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True).abs()
        corr_matrix = corr.to_numpy(copy=True)
        np.fill_diagonal(corr_matrix, 0)
        if corr_matrix.max() > 0:
            idx = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
            x, y = numeric_cols[idx[0]], numeric_cols[idx[1]]
            try:
                suggestions.append({
                    "title": f"{x} vs {y} (scatter)",
                    "reason": "The two most strongly correlated numeric columns — a scatter plot reveals the relationship.",
                    **chart_data(df, x, "scatter", y),
                })
            except ValueError:
                pass

    # 2. Text columns → word frequency (most informative view for free-form text)
    for c in text_cols[:2]:
        try:
            suggestions.append({
                "title": f"Most frequent words in {c}",
                "reason": "This looks like free-form text — a word frequency chart shows what's talked about most.",
                **chart_data(df, c, "word_frequency"),
            })
        except ValueError:
            pass

    # 3. Categorical columns → pie for low cardinality, bar for higher
    cat_candidates = [c for c in categorical_cols if 1 < df[c].nunique(dropna=True) <= 20]
    for c in cat_candidates[:2]:
        n_unique = df[c].nunique(dropna=True)
        ctype = "pie" if n_unique <= 6 else "bar"
        suggestions.append({
            "title": f"Distribution of {c}",
            "reason": f"{'Pie chart' if ctype == 'pie' else 'Bar chart'} showing how '{c}' values are distributed — useful for spotting class imbalance.",
            **chart_data(df, c, ctype),
        })

    # 4. Most spread numeric → box plot (best for outlier detection)
    if numeric_cols:
        spreads = {c: df[c].dropna().std() for c in numeric_cols if df[c].dropna().std() > 0}
        for c in sorted(spreads, key=lambda k: -spreads[k])[:1]:
            suggestions.append({
                "title": f"Box plot of {c}",
                "reason": "Normal box plot for the most spread-out numeric column.",
                "_boxplotMode": "full",
                **chart_data(df, c, "boxplot"),
            })

    # 5. Histogram for second most spread numeric
    ranked = sorted(spreads, key=lambda k: -spreads[k])
    for c in ranked[1:2]:
        suggestions.append({
            "title": f"Distribution of {c}",
            "reason": "Histogram showing how values are spread — check for skew or unusual clusters.",
            **chart_data(df, c, "histogram"),
        })

    # 6. Density plot for most spread numeric
    if numeric_cols and spreads:
        most_spread = max(spreads, key=lambda k: spreads[k])
        try:
            suggestions.append({
                "title": f"Density of {most_spread}",
                "reason": "Smoothed distribution curve — useful for seeing the shape of the data distribution.",
                **chart_data(df, most_spread, "density"),
            })
        except Exception:
            pass

    # 7. Scatter map / heatmap if lat/lon columns detected — insert at front so
    #    the cap doesn't cut them off when there are many other chart types.
    coords = _detect_coordinate_columns(df)
    if coords["lat"] and coords["lon"]:
        try:
            num_cols_for_map = [c for c in numeric_cols if c not in {coords["lat"], coords["lon"]}]
            size_col  = num_cols_for_map[0] if num_cols_for_map else None
            color_col = categorical_cols[0] if categorical_cols else (num_cols_for_map[1] if len(num_cols_for_map) > 1 else size_col)

            map_data = _scatter_map_data(df, coords["lat"], coords["lon"],
                                         size_col=size_col, color_col=color_col)
            geo_suggestions = [{
                "title": f"Map — {coords['lat']} / {coords['lon']}",
                "reason": (
                    f"Your data has geographic coordinates — plotted as an interactive map. "
                    + (f"Markers sized by {size_col}. " if size_col else "")
                    + "Click any marker to see all column values."
                ),
                **map_data,
            }]

            if len(df) > 200:
                heatmap_data = _scatter_map_data(df, coords["lat"], coords["lon"], max_points=5000)
                heatmap_data["type"] = "heatmap_map"
                geo_suggestions.append({
                    "title": f"Density heatmap — {coords['lat']} / {coords['lon']}",
                    "reason": "Shows where data points are densest — useful for spotting hotspots.",
                    **heatmap_data,
                })

            # Put geo charts first so they're never cut off by the cap
            suggestions = geo_suggestions + suggestions
        except Exception:
            pass

    return suggestions[:max_suggestions]