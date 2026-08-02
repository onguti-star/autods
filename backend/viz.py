"""
Visualization data layer — produces Chart.js-ready data for all chart types.
"""
import numpy as np
import pandas as pd

from . import nlp


# ── helpers ────────────────────────────────────────────────────────────────

def _histogram_data(s: pd.Series, bins: int = 12):
    clean = s.dropna()
    if len(clean) == 0:
        return {"labels": [], "values": []}
    n_bins = min(bins, max(1, clean.nunique()))
    counts, edges = np.histogram(clean, bins=n_bins)
    labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(edges) - 1)]
    return {"labels": labels, "values": [int(c) for c in counts]}


def _bar_data(s: pd.Series, top_n: int = 15):
    counts = s.value_counts(dropna=True).head(top_n)
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
    if pd.api.types.is_numeric_dtype(sub[x]):
        xs = [float(v) for v in sub[x]]
    else:
        xs = [str(v) for v in sub[x]]
    ys = [float(v) for v in sub[y]]
    return {"labels": xs, "values": ys}


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
    if pd.api.types.is_numeric_dtype(sub[x]):
        xs = [float(v) for v in sub[x]]
    else:
        xs = [str(v) for v in sub[x]]
    ys = [float(v) for v in sub[y]]
    return {"type": "area", "labels": xs, "values": ys}


def _violin_data(df: pd.DataFrame, x: str, group_col: str = None) -> dict:
    """Generate data for violin plot - distribution with density."""
    def compute_violin_stats(s):
        s = s.dropna()
        if len(s) == 0:
            return None
        
        # Create bins for violin shape
        try:
            import numpy as np
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
    
    for col in df.columns:
        col_lower = col.lower()
        # Check column names
        if col_lower in ['lat', 'latitude', 'y', 'latitud', 'breite']:
            lat_candidates.append(col)
        elif col_lower in ['lon', 'lng', 'longitude', 'x', 'longitud', 'laenge']:
            lon_candidates.append(col)
        # Check if column contains coordinate-like values
        elif pd.api.types.is_numeric_dtype(df[col]):
            values = df[col].dropna()
            if len(values) > 0:
                # Latitude ranges from -90 to 90
                if values.min() >= -90 and values.max() <= 90:
                    lat_candidates.append(col)
                # Longitude ranges from -180 to 180
                elif values.min() >= -180 and values.max() <= 180:
                    lon_candidates.append(col)
    
    return {
        "lat": lat_candidates[0] if lat_candidates else None,
        "lon": lon_candidates[0] if lon_candidates else None
    }


def _scatter_map_data(df: pd.DataFrame, lat_col: str, lon_col: str, size_col: str = None, color_col: str = None, max_points: int = 1000) -> dict:
    """Generate data for scatter map visualization."""
    cols = [lat_col, lon_col]
    if size_col:
        cols.append(size_col)
    if color_col and color_col not in cols:
        cols.append(color_col)
    
    sub = df[cols].dropna()
    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=42)
    
    points = []
    for _, r in sub.iterrows():
        point = {
            "lat": float(r[lat_col]),
            "lon": float(r[lon_col])
        }
        if size_col and pd.api.types.is_numeric_dtype(df[size_col]):
            point["size"] = float(r[size_col])
        if color_col:
            point["color"] = str(r[color_col])
        points.append(point)
    
    return {
        "type": "scatter_map",
        "points": points,
        "lat_col": lat_col,
        "lon_col": lon_col
    }


def _choropleth_data(geojson: dict, df: pd.DataFrame, region_id_col: str, value_col: str) -> dict:
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
               y: str = None, group: str = None) -> dict:
    if x not in df.columns:
        raise ValueError(f"Column '{x}' not found.")
    s = df[x]

    if chart_type == "histogram":
        if not pd.api.types.is_numeric_dtype(s):
            raise ValueError(f"'{x}' is not numeric — try a bar chart instead.")
        return {"type": "histogram", "x": x, **_histogram_data(s)}

    if chart_type == "bar":
        return {"type": "bar", "x": x, **_bar_data(s)}

    if chart_type == "pie":
        return {"type": "pie", "x": x, **_pie_data(s)}

    if chart_type == "scatter":
        if not y or y not in df.columns:
            raise ValueError("Scatter needs a Y column.")
        if not pd.api.types.is_numeric_dtype(s) or not pd.api.types.is_numeric_dtype(df[y]):
            raise ValueError("Scatter needs two numeric columns.")
        return {"type": "scatter", "x": x, "y": y, **_scatter_data(df, x, y)}

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
        if len(sub) > 400:
            sub = sub.sample(400, random_state=42)
        sz = sub[group]
        sz_norm = 3 + 27 * (sz - sz.min()) / (sz.max() - sz.min() + 1e-9)
        pts = [{"x": float(r[x]), "y": float(r[y]), "r": float(sz_norm.iloc[i])}
               for i, (_, r) in enumerate(sub.iterrows())]
        return {"type": "bubble", "x": x, "y": y, "size": group, "points": pts}

    if chart_type == "boxplot":
        if not pd.api.types.is_numeric_dtype(s):
            raise ValueError(f"'{x}' must be numeric for a box plot.")
        return {"type": "boxplot", "x": x, "group": group, **_boxplot_data(df, x, group)}

    if chart_type == "line":
        if not y or y not in df.columns:
            raise ValueError("Line chart needs a Y column.")
        if not pd.api.types.is_numeric_dtype(df[y]):
            raise ValueError(f"'{y}' must be numeric for a line chart.")
        return {"type": "line", "x": x, "y": y, **_line_data(df, x, y)}

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
        if not y:
            raise ValueError("Radar chart needs numeric columns.")
        cols = [x] + ([y] if y and y != x else [])
        cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            raise ValueError("Radar chart needs at least one numeric column.")
        return _radar_data(df, cols, group)

    if chart_type == "area":
        if not y or y not in df.columns:
            raise ValueError("Area chart needs a Y column.")
        return _area_data(df, x, y)

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


def suggest_visuals(df: pd.DataFrame, max_suggestions: int = 6) -> list:
    suggestions = []
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

    # 6. Correlation heatmap for multiple numeric columns
    if len(numeric_cols) >= 3:
        try:
            corr = df[numeric_cols].corr()
            matrix = corr.values.tolist()
            labels = numeric_cols
            suggestions.append({
                "title": "Correlation heatmap",
                "reason": "Visualize relationships between all numeric columns — red = positive correlation, blue = negative.",
                **chart_data(df, numeric_cols[0], "heatmap", numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]),
            })
        except Exception:
            pass

    # 7. Radar chart for comparing numeric columns (if we have a categorical column to group by)
    if len(numeric_cols) >= 3 and categorical_cols:
        try:
            suggestions.append({
                "title": f"Radar chart: {', '.join(numeric_cols[:4])}",
                "reason": f"Compare multiple numeric columns across groups in {categorical_cols[0]}.",
                **chart_data(df, numeric_cols[0], "radar", numeric_cols[1] if len(numeric_cols) > 1 else None, categorical_cols[0]),
            })
        except Exception:
            pass

    # 8. Density plot for most spread numeric
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

    # 9. Scatter map if lat/lon columns detected
    coords = _detect_coordinate_columns(df)
    if coords["lat"] and coords["lon"]:
        try:
            suggestions.append({
                "title": f"Geographic scatter map ({coords['lat']} vs {coords['lon']})",
                "reason": "Interactive map showing the geographic distribution of your data points.",
                **chart_data(df, coords["lat"], "scatter_map", coords["lon"]),
            })
        except Exception:
            pass

    return suggestions[:max_suggestions]
