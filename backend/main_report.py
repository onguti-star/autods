import io
import html
import json
from datetime import datetime

import pandas as pd

from . import eda
from . import narrate


def _fmt_report_value(value) -> str:
    if value is pd.NA:
        return "missing"
    if pd.isna(value):
        return "missing"
    if isinstance(value, float):
        return f"{value:,.4g}"
    return str(value).replace("\n", " ").replace("\r", " ")


def _fmt_prediction_inputs(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "No input values saved."
    row = rows[0]
    if not row:
        return "No input values saved."
    return "; ".join(f"{key}: {_fmt_report_value(value)}" for key, value in row.items())


def _compact_unsupervised_result(result: dict) -> dict:
    """Keep report-worthy unsupervised details without storing huge label arrays twice."""
    compact = dict(result)
    labels = compact.pop("labels", [])
    compact["n_rows_scored"] = len(labels)
    points = compact.pop("points", [])
    if points and not compact["n_rows_scored"]:
        compact["n_rows_scored"] = len(points)
    viz = compact.get("visualization")
    if isinstance(viz, dict) and "points" in viz:
        compact["visualization"] = {
            k: v for k, v in viz.items() if k != "points"
        }
        compact["n_visualized_points"] = len(viz.get("points") or [])
    if "anomaly_indices" in compact:
        compact["anomaly_indices_preview"] = compact["anomaly_indices"][:20]
        compact.pop("anomaly_indices", None)
    if "anomaly_scores" in compact:
        compact["anomaly_scores_preview"] = compact["anomaly_scores"][:10]
        compact.pop("anomaly_scores", None)
    return compact


def _has_cleaning_history(session) -> bool:
    return bool(session.cleaning_log or session.chat_clean_log)


def _build_html_report(session, extra_charts=None) -> str:
    """Build a styled HTML presentation report."""
    profile = eda.profile_dataframe(session.df)
    narrative = narrate.narrate_eda(profile)
    charts = list(extra_charts or [])
    chart_data_json = json.dumps(charts)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    preview_title = "Cleaned Data Preview (First 10 Rows)" if _has_cleaning_history(session) else "Data Preview (First 10 Rows)"
    
    # HTML structure with Bootstrap 5 CDN + inline styling
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoDS Presentation Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            padding: 0;
            line-height: 1.6;
            margin: 0;
        }
        .container {
            max-width: 95vw;
            width: 1400px;
            background-color: white;
            padding: 50px;
            min-height: 100vh;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        @media (max-width: 1600px) {
            .container {
                width: 95vw;
                max-width: none;
                padding: 40px;
            }
        }
        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }
        }
        .report-header {
            border-bottom: 4px solid #0d6efd;
            padding-bottom: 20px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 20px;
        }
        .report-title {
            font-size: 2.5em;
            font-weight: 700;
            color: #0d6efd;
            margin-bottom: 10px;
            flex: 1;
        }
        .report-meta {
            color: #6c757d;
            font-size: 0.95em;
        }
        .action-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .btn-action {
            padding: 8px 16px;
            border-radius: 6px;
            border: none;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 600;
            transition: all 0.2s;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .btn-primary-action {
            background-color: #0d6efd;
            color: white;
        }
        .btn-primary-action:hover {
            background-color: #0a58ca;
            color: white;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(13,110,253,0.3);
        }
        .btn-secondary-action {
            background-color: #6c757d;
            color: white;
        }
        .btn-secondary-action:hover {
            background-color: #5a6268;
            color: white;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(108,117,125,0.3);
        }
        .section-title {
            font-size: 1.8em;
            font-weight: 600;
            color: #0d6efd;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e9ecef;
            page-break-after: avoid;
        }
        .subsection-title {
            font-size: 1.3em;
            font-weight: 600;
            color: #495057;
            margin-top: 25px;
            margin-bottom: 15px;
        }
        .stat-box {
            background-color: #f8f9fa;
            border-left: 4px solid #0d6efd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .stat-value {
            font-size: 1.4em;
            font-weight: 700;
            color: #0d6efd;
        }
        .stat-label {
            color: #6c757d;
            font-size: 0.9em;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        table thead {
            background-color: #0d6efd;
            color: white;
        }
        table th {
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        table td {
            padding: 10px 12px;
            border-bottom: 1px solid #e9ecef;
        }
        table tbody tr:hover {
            background-color: #f8f9fa;
        }
        .toc {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 4px;
            margin-bottom: 30px;
            border-left: 4px solid #0d6efd;
        }
        .toc h3 {
            color: #0d6efd;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        .toc ul {
            list-style: none;
            padding-left: 0;
        }
        .toc li {
            margin: 8px 0;
        }
        .toc a {
            color: #0d6efd;
            text-decoration: none;
            transition: color 0.2s;
        }
        .toc a:hover {
            color: #0a58ca;
            text-decoration: underline;
        }
        .correlation-item {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            background-color: #f8f9fa;
            margin: 8px 0;
            border-radius: 4px;
            border-left: 3px solid #0d6efd;
        }
        .correlation-value {
            font-weight: 700;
            color: #0d6efd;
            font-family: 'Courier New', monospace;
        }
        .cleaning-item {
            padding: 10px;
            margin: 8px 0;
            background-color: #d4edda;
            border-left: 3px solid #198754;
            border-radius: 4px;
        }
        .prediction-item {
            padding: 14px 16px;
            margin: 12px 0;
            background-color: #fff8e1;
            border-left: 4px solid #ffc107;
            border-radius: 4px;
        }
        .prediction-output {
            font-size: 1.1em;
            font-weight: 700;
            color: #198754;
            margin-bottom: 6px;
        }
        .prediction-meta,
        .prediction-inputs {
            color: #6c757d;
            font-size: 0.9em;
        }
        .prediction-inputs {
            margin-top: 6px;
            overflow-wrap: anywhere;
        }
        .analysis-item {
            padding: 14px 16px;
            margin: 12px 0;
            background-color: #eef8ff;
            border-left: 4px solid #0dcaf0;
            border-radius: 4px;
        }
        .analysis-meta {
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 4px;
        }
        .narrative-text {
            color: #495057;
            line-height: 1.8;
            margin: 15px 0;
        }
        .data-preview {
            font-size: 0.85em;
            overflow-x: auto;
        }
        .chart-row {
            display: flex;
            flex-direction: column;
            gap: 40px;
            margin-bottom: 50px;
        }
        .chart-card {
            background-color: #f8f9fa;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            border: 1px solid #e9ecef;
            position: relative;
        }
        .chart-canvas-wrap {
            position: relative;
            width: 100%;
            height: 800px;
            min-height: 600px;
        }
        .chart-actions {
            position: absolute;
            top: 10px;
            right: 10px;
            display: flex;
            gap: 8px;
        }
        .chart-btn {
            padding: 6px 12px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
            background-color: white;
            cursor: pointer;
            font-size: 0.85em;
            font-weight: 600;
            transition: all 0.2s;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        }
        .chart-btn:hover {
            background-color: #0d6efd;
            color: white;
            border-color: #0d6efd;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(13,110,253,0.3);
        }
        .chart-canvas-wrap canvas {
            width: 100% !important;
            height: 100% !important;
        }
        .chart-title {
            margin-bottom: 12px;
            color: #0d6efd;
            font-size: 1.15em;
            font-weight: 700;
        }
        .chart-notice {
            color: #6c757d;
            font-size: 0.9em;
            margin-bottom: 12px;
        }
        @media (max-width: 992px) {
            .chart-row {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }
            .report-title {
                font-size: 1.8em;
            }
            .section-title {
                font-size: 1.4em;
            }
        }
        @media print {
            body {
                background-color: white;
                padding: 0;
                margin: 0;
            }
            .container {
                box-shadow: none;
                padding: 20px;
                max-width: 100%;
                width: 100%;
            }
            .action-buttons {
                display: none !important;
            }
            .section-title {
                page-break-after: avoid;
            }
            table {
                page-break-inside: avoid;
            }
            .chart-card {
                page-break-inside: avoid;
            }
        }
    </style>
</head>
<body>
    <div class="container">""")
    
    # Header
    filename = session.filename.rsplit(".", 1)[0]
    html_parts.append(f"""        <div class="report-header">
            <div>
                <h1 class="report-title">📊 AutoDS Analysis Report</h1>
                <p class="report-meta"><strong>Dataset:</strong> {filename}</p>
                <p class="report-meta"><strong>Generated:</strong> {generated_at}</p>
            </div>
            <div class="action-buttons">
                <button class="btn-action btn-primary-action" onclick="window.print()">
                    🖨️ Print / Save as PDF
                </button>
            </div>
        </div>""")
    
    # Table of Contents
    html_parts.append("""        <div class="toc">
            <h3>📋 Table of Contents</h3>
            <ul>
                <li><a href="#dataset">Dataset Overview</a></li>
                <li><a href="#summary">EDA Summary</a></li>
                <li><a href="#describe">Data Describe Summary</a></li>
                <li><a href="#columns">Column Analysis</a></li>
                <li><a href="#correlations">Correlations</a></li>""")
    if charts:
        html_parts.append('                <li><a href="#visualizations">Visualizations</a></li>')
    
    if _has_cleaning_history(session):
        html_parts.append('                <li><a href="#cleaning">Data Cleaning Log</a></li>')
    if session.leaderboard:
        html_parts.append('                <li><a href="#training">Model Training Results</a></li>')
    if session.saved_predictions:
        html_parts.append('                <li><a href="#predictions">Saved Predictions</a></li>')
    if session.unsupervised_results:
        html_parts.append('                <li><a href="#unsupervised">Unsupervised Learning</a></li>')
    
    html_parts.append(f"""                <li><a href="#preview">{html.escape(preview_title)}</a></li>
            </ul>
        </div>""")
    
    # Dataset Overview Section
    html_parts.append(f"""        <h2 class="section-title" id="dataset">📈 Dataset Overview</h2>
        <div class="stat-box">
            <div class="stat-label">Total Rows</div>
            <div class="stat-value">{profile['shape']['rows']:,}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Total Columns</div>
            <div class="stat-value">{profile['shape']['columns']:,}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Duplicate Rows</div>
            <div class="stat-value">{profile['duplicate_rows']:,}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Missing Cells</div>
            <div class="stat-value">{profile['total_missing_cells']:,}</div>
            <span style="color: #6c757d; font-size: 0.9em;">of {profile['total_cells']:,} total</span>
        </div>""")
    
    # EDA Summary Section
    html_parts.append("""        <h2 class="section-title" id="summary">📝 EDA Summary</h2>
        <div class="narrative-text">""")
    
    for note in narrative:
        html_parts.append(f"            <p>• {note}</p>\n")
    
    html_parts.append("        </div>")

    describe = profile.get("describe") or {}
    if describe.get("columns") and describe.get("index"):
        html_parts.append("""        <h2 class="section-title" id="describe">📐 Data Describe Summary</h2>
        <div class="data-preview">
            <table class="table">
                <thead><tr><th>Statistic</th>""")
        for col_name in describe.get("columns", []):
            html_parts.append(f"<th>{html.escape(str(col_name))}</th>")
        html_parts.append("</tr></thead><tbody>")
        for row_idx, stat_name in enumerate(describe.get("index", [])):
            values = describe.get("data", [[]])[row_idx] if row_idx < len(describe.get("data", [])) else []
            html_parts.append(f"""                <tr><td><strong>{html.escape(str(stat_name))}</strong></td>""")
            for col_idx, _ in enumerate(describe.get("columns", [])):
                value = values[col_idx] if col_idx < len(values) else None
                html_parts.append(f"<td>{html.escape(_fmt_report_value(value)) if value is not None else ''}</td>")
            html_parts.append("</tr>")
        html_parts.append("""                </tbody>
            </table>
        </div>""")
    
    # Columns Section
    html_parts.append("""        <h2 class="section-title" id="columns">🔍 Column Analysis</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>Column Name</th>
                    <th>Type</th>
                    <th>Unique</th>
                    <th>Missing</th>
                    <th>Sample Stats</th>
                </tr>
            </thead>
            <tbody>""")
    
    for col in profile["columns"]:
        stats_str = ""
        stats = col.get("stats")
        if stats:
            stats_str = f"μ={_fmt_report_value(stats.get('mean'))}, min={_fmt_report_value(stats.get('min'))}, max={_fmt_report_value(stats.get('max'))}"
        
        top_values_str = ""
        top_values = col.get("top_values")
        if top_values:
            top = ", ".join(f"{v['value']} ({v['count']:,})" for v in top_values[:3])
            top_values_str = f"<br><small style='color:#6c757d;'>Top: {top}</small>"
        
        html_parts.append(f"""                <tr>
                    <td><strong>{col['name']}</strong></td>
                    <td>{col['type']}</td>
                    <td>{col['unique']:,}</td>
                    <td>{col['missing']:,} ({col['missing_pct']}%)</td>
                    <td>{stats_str}{top_values_str}</td>
                </tr>""")
    
    html_parts.append("""            </tbody>
        </table>""")
    
    # Correlations Section
    corr = profile.get("correlation", {})
    if corr.get("matrix") and len(corr.get("columns", [])) >= 2:
        pairs = []
        corr_cols = corr["columns"]
        matrix = corr["matrix"]
        for i, a in enumerate(corr_cols):
            for j in range(i + 1, len(corr_cols)):
                pairs.append((a, corr_cols[j], matrix[i][j]))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        html_parts.append("""        <h2 class="section-title" id="correlations">🔗 Strongest Correlations</h2>""")
        
        for a, b, val in pairs[:10]:
            color = "#0d6efd" if val > 0 else "#dc3545"
            html_parts.append(f"""        <div class="correlation-item">
            <span><strong>{a}</strong> ↔ <strong>{b}</strong></span>
            <span class="correlation-value" style="color: {color};">{val:+.3f}</span>
        </div>""")
    
    if charts:
        html_parts.append("""        <h2 class="section-title" id="visualizations">📊 Visualizations Created</h2>
        <div class="chart-row">""")
        for index, chart in enumerate(charts):
            canvas_id = f"chart_{index}"
            chart_title = html.escape(str(chart.get("title") or "Visualization"))
            chart_note = html.escape(str(chart.get("reason") or ""))
            html_parts.append(f"""            <div class="chart-card">
                <div class="chart-title">{chart_title}</div>
                <div class="chart-notice">{chart_note}</div>
                <div class="chart-actions">
                    <button class="chart-btn" onclick="copyChart({index})">📋 Copy Image</button>
                </div>
                <div class="chart-canvas-wrap"><canvas id="{canvas_id}"></canvas></div>
            </div>""")
        html_parts.append("        </div>")
    
    # Cleaning Log Section
    if _has_cleaning_history(session):
        html_parts.append("""        <h2 class="section-title" id="cleaning">✨ Data Cleaning Log</h2>""")
        for entry in session.cleaning_log:
            html_parts.append(f"""        <div class="cleaning-item">
            ✓ {html.escape(str(entry))}
        </div>""")
        for entry in session.chat_clean_log:
            command = html.escape(str(entry.get("command", "")))
            message = html.escape(str(entry.get("message", "")))
            html_parts.append(f"""        <div class="cleaning-item">
            ✓ <strong>{command}</strong><br>
            <span>{message}</span>
        </div>""")
    
    # Model Training Section
    if session.leaderboard:
        html_parts.append(f"""        <h2 class="section-title" id="training">🤖 Model Training Results</h2>
        <p><strong>Target Column:</strong> {session.target}</p>
        <p><strong>Problem Type:</strong> {session.problem_type.title()}</p>
        <p><strong>Best Model:</strong> {session.best_model_name}</p>
        <table class="table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Metrics</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>""")
        
        for row in session.leaderboard:
            status = "✓ Success" if not row.get("error") else "✗ Failed"
            if row.get("error"):
                metrics = f"<span style='color:#dc3545;'>{row['error']}</span>"
            else:
                metrics = ", ".join(f"<strong>{k}:</strong> {_fmt_report_value(v)}" for k, v in row.get("metrics", {}).items())
            
            html_parts.append(f"""                <tr>
                    <td><strong>{row.get('model', 'model')}</strong></td>
                    <td>{metrics}</td>
                    <td>{status}</td>
                </tr>""")
        
        html_parts.append("""            </tbody>
        </table>""")

    if session.saved_predictions:
        html_parts.append("""        <h2 class="section-title" id="predictions">🎯 Saved Predictions</h2>""")
        for prediction in session.saved_predictions.values():
            target = html.escape(str(prediction.get("target") or "Prediction"))
            outputs = ", ".join(_fmt_report_value(v) for v in prediction.get("predictions", []))
            output = html.escape(outputs or "missing")
            source = html.escape(str(prediction.get("source_name") or "Current training"))
            model = html.escape(str(prediction.get("model_name") or "model"))
            created_at = html.escape(str(prediction.get("created_at") or ""))
            inputs = html.escape(_fmt_prediction_inputs(prediction.get("inputs", [])))
            narrative_text = html.escape(str(prediction.get("narrative") or ""))
            html_parts.append(f"""        <div class="prediction-item">
            <div class="prediction-output">{target}: {output}</div>
            <div class="prediction-meta">{source} · {model} · {created_at}</div>
            <div class="prediction-inputs"><strong>Inputs:</strong> {inputs}</div>""")
            if narrative_text:
                html_parts.append(f"""            <div class="prediction-inputs"><strong>Note:</strong> {narrative_text}</div>""")
            html_parts.append("        </div>")

    if session.unsupervised_results:
        html_parts.append("""        <h2 class="section-title" id="unsupervised">🧭 Unsupervised Learning</h2>""")
        suggestions = session.unsupervised_results.get("suggestions") or {}
        preprocessing = suggestions.get("preprocessing") or {}
        if preprocessing:
            features = preprocessing.get("features_used") or []
            feature_text = ", ".join(html.escape(str(c)) for c in features[:12])
            if len(features) > 12:
                feature_text += ", ..."
            html_parts.append(f"""        <div class="analysis-item">
            <strong>Preprocessing for Unsupervised Learning</strong>
            <div class="analysis-meta"><strong>Scaling:</strong> {html.escape(str(preprocessing.get("scaling", "StandardScaler")))}</div>
            <div class="analysis-meta">{html.escape(str(preprocessing.get("scaling_description", "")))}</div>
            <div class="analysis-meta"><strong>Numeric features used:</strong> {len(features):,}{f" ({feature_text})" if feature_text else ""}</div>
        </div>""")
        cluster_analysis = session.unsupervised_results.get("cluster_analysis")
        if cluster_analysis:
            html_parts.append(f"""        <div class="analysis-item">
                <strong>Cluster Number Analysis</strong>
            <div class="analysis-meta">Best silhouette K: {html.escape(str(cluster_analysis.get("best_silhouette_k", "")))}; best Davies-Bouldin K: {html.escape(str(cluster_analysis.get("best_db_k", "")))}; elbow K: {html.escape(str(cluster_analysis.get("elbow_k", "")))}; best Calinski-Harabasz K: {html.escape(str(cluster_analysis.get("best_ch_k", "")))}</div>""")
            rows = cluster_analysis.get("k_analysis") or []
            if rows:
                html_parts.append("""            <table class="table"><thead><tr><th>K</th><th>Inertia</th><th>Silhouette</th><th>Davies-Bouldin</th><th>Calinski-Harabasz</th><th>Votes</th></tr></thead><tbody>""")
                for row in rows:
                    html_parts.append(f"""                <tr>
                    <td>{html.escape(str(row.get("k", "")))}</td>
                    <td>{_fmt_report_value(row.get("inertia"))}</td>
                    <td>{_fmt_report_value(row.get("silhouette"))}</td>
                    <td>{_fmt_report_value(row.get("davies_bouldin"))}</td>
                    <td>{_fmt_report_value(row.get("calinski_harabasz"))}</td>
                    <td>{html.escape(str(row.get("votes", "")))}</td>
                </tr>""")
                html_parts.append("            </tbody></table>")
            html_parts.append("        </div>")
        clustering = session.unsupervised_results.get("clustering")
        if clustering:
            method = html.escape(str(clustering.get("method", "Clustering")))
            selected = html.escape(str(clustering.get("selected_method") or method))
            reason = html.escape(str(clustering.get("selection_reason") or ""))
            rows_scored = clustering.get("n_rows_scored")
            html_parts.append(f"""        <div class="analysis-item">
            <strong>{method}</strong>
            <div class="analysis-meta">Selected method: {selected}</div>""")
            if rows_scored:
                html_parts.append(f"""            <div class="analysis-meta">Rows scored: {int(rows_scored):,}</div>""")
            if reason:
                html_parts.append(f"""            <div class="analysis-meta">{reason}</div>""")
            sizes = clustering.get("cluster_sizes") or {}
            if sizes:
                html_parts.append("            <table class='table'><thead><tr><th>Cluster</th><th>Rows</th></tr></thead><tbody>")
                for name, count in sizes.items():
                    html_parts.append(f"""                <tr><td>{html.escape(str(name))}</td><td>{int(count):,}</td></tr>""")
                html_parts.append("            </tbody></table>")
            metrics = clustering.get("metrics") or {}
            if metrics:
                metrics_text = ", ".join(f"<strong>{html.escape(str(k))}:</strong> {_fmt_report_value(v)}" for k, v in metrics.items())
                html_parts.append(f"""            <div class="analysis-meta">{metrics_text}</div>""")
            html_parts.append("        </div>")
        anomaly = session.unsupervised_results.get("anomaly")
        if anomaly:
            html_parts.append(f"""        <div class="analysis-item">
            <strong>{html.escape(str(anomaly.get("method", "Anomaly Detection")))}</strong>
            <div class="analysis-meta">Anomalies found: {int(anomaly.get("n_outliers", 0)):,} ({_fmt_report_value(anomaly.get("outlier_percentage", 0))}% of data)</div>
            <div class="analysis-meta">Normal rows: {int(anomaly.get("n_normal", 0)):,}</div>
        </div>""")
        reduction = session.unsupervised_results.get("reduction")
        if reduction:
            html_parts.append(f"""        <div class="analysis-item">
            <strong>{html.escape(str(reduction.get("method", "Dimensionality Reduction")))}</strong>
            <div class="analysis-meta">Components: {html.escape(str(reduction.get("n_components", "")))}</div>
            <div class="analysis-meta">Points generated: {int(reduction.get("n_rows_scored", 0)):,}</div>""")
            if reduction.get("explained_variance"):
                variance = ", ".join(f"PC{i + 1}: {_fmt_report_value(v)}%" for i, v in enumerate(reduction.get("explained_variance", [])))
                html_parts.append(f"""            <div class="analysis-meta">Explained variance: {variance}</div>""")
            html_parts.append("        </div>")
        association = session.unsupervised_results.get("association")
        if association:
            html_parts.append(f"""        <div class="analysis-item">
            <strong>{html.escape(str(association.get("method", "Association Rules")))}</strong>
            <div class="analysis-meta">Rules found: {int(association.get("n_rules", 0)):,}</div>
            <div class="analysis-meta">Minimum support: {_fmt_report_value(association.get("min_support"))}; minimum confidence: {_fmt_report_value(association.get("min_confidence"))}</div>""")
            rules = association.get("rules") or []
            if rules:
                html_parts.append("""            <table class="table"><thead><tr><th>Antecedents</th><th>Consequents</th><th>Support</th><th>Confidence</th><th>Lift</th></tr></thead><tbody>""")
                for rule in rules[:10]:
                    html_parts.append(f"""                <tr>
                    <td>{html.escape(", ".join(map(str, rule.get("antecedents", []))))}</td>
                    <td>{html.escape(", ".join(map(str, rule.get("consequents", []))))}</td>
                    <td>{_fmt_report_value(rule.get("support"))}</td>
                    <td>{_fmt_report_value(rule.get("confidence"))}</td>
                    <td>{_fmt_report_value(rule.get("lift"))}</td>
                </tr>""")
                html_parts.append("            </tbody></table>")
            html_parts.append("        </div>")
    
    # Data Preview Section
    html_parts.append(f"""        <h2 class="section-title" id="preview">👀 {html.escape(preview_title)}</h2>
        <div class="data-preview">
            <table class="table">""")
    
    preview = eda.safe_preview(session.df, 10)
    if preview:
        headers = list(preview[0].keys())
        html_parts.append("                <thead><tr>")
        for h in headers:
            html_parts.append(f"<th>{html.escape(str(h))}</th>")
        html_parts.append("                </tr></thead>")
        html_parts.append("                <tbody>")
        
        for row in preview:
            html_parts.append("                    <tr>")
            for h in headers:
                val = _fmt_report_value(row.get(h))
                html_parts.append(f"<td>{html.escape(val)}</td>")
            html_parts.append("                    </tr>")
        
        html_parts.append("                </tbody>")
    else:
        html_parts.append("        <p>No preview rows available.</p>")
    
    html_parts.append("""            </table>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        function copyReportLink() {
            const link = window.location.href;
            navigator.clipboard.writeText(link).then(() => {
                showNotification('Report link copied to clipboard!');
            }).catch(() => {
                prompt('Copy this link:', link);
            });
        }

        function downloadHTML() {
            const htmlContent = document.documentElement.outerHTML;
            const blob = new Blob([htmlContent], { type: 'text/html' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'autods_report.html';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            showNotification('HTML report downloaded!');
        }

        function copyChart(index) {
            const canvas = document.getElementById(`chart_${index}`);
            if (!canvas) {
                showNotification('❌ Chart not found');
                return;
            }
            
            // Find the chart card container
            const chartCard = canvas.closest('.chart-card');
            if (!chartCard) {
                showNotification('❌ Chart container not found');
                return;
            }
            
            // Method 1: Try to copy the entire chart card as an image using html2canvas approach
            // Since we can't use external libraries, we'll create a downloadable link
            try {
                const dataUrl = canvas.toDataURL('image/png');
                
                // Create a temporary download link
                const link = document.createElement('a');
                link.download = `chart_${index + 1}.png`;
                link.href = dataUrl;
                
                // Trigger download
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                
                showNotification('✓ Chart downloaded! Check your downloads folder.');
                
                // Also try to copy to clipboard
                copyToClipboard(dataUrl);
                
            } catch (err) {
                console.error('Download failed:', err);
                showNotification('❌ Failed to download. Please right-click the chart and select "Save image as..."');
            }
        }

        function copyToClipboard(dataUrl) {
            // Convert data URL to blob
            fetch(dataUrl)
                .then(res => res.blob())
                .then(blob => {
                    if (navigator.clipboard && window.ClipboardItem) {
                        try {
                            const item = new ClipboardItem({ 'image/png': blob });
                            return navigator.clipboard.write([item]);
                        } catch (err) {
                            console.log('Clipboard API not supported');
                            return Promise.resolve();
                        }
                    }
                    return Promise.resolve();
                })
                .then(() => {
                    if (navigator.clipboard && window.ClipboardItem) {
                        showNotification('✓ Chart also copied to clipboard! You can paste it now.');
                    }
                })
                .catch(err => {
                    console.log('Clipboard copy failed:', err);
                });
        }

        function copyReportContent() {
            // Create a simplified text version of the report
            let textContent = 'AUTODS ANALYSIS REPORT\n';
            textContent += '='.repeat(50) + '\n\n';
            
            // Extract title and metadata
            const title = document.querySelector('.report-title')?.textContent || 'AutoDS Analysis Report';
            const dataset = document.querySelector('.report-meta strong')?.parentElement?.textContent || '';
            textContent += `${title}\n${dataset}\n\n`;
            
            // Extract all section titles and content
            const sections = document.querySelectorAll('.section-title, .narrative-text p, .stat-box, .correlation-item, .cleaning-item, .prediction-item, .analysis-item');
            sections.forEach(section => {
                const text = section.textContent.trim();
                if (text) {
                    textContent += text + '\n\n';
                }
            });
            
            // Copy to clipboard
            navigator.clipboard.writeText(textContent).then(() => {
                showNotification('Report text copied to clipboard! You can paste it into Word, PowerPoint, or any document.');
            }).catch(() => {
                // Fallback: select all text
                const range = document.createRange();
                range.selectNode(document.body);
                const selection = window.getSelection();
                selection.removeAllRanges();
                selection.addRange(range);
                showNotification('Text selected! Press Ctrl+C to copy.');
            });
        }

        function showNotification(message) {
            // Create notification element
            const notification = document.createElement('div');
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background-color: #198754;
                color: white;
                padding: 15px 20px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 14px;
                font-weight: 600;
                z-index: 10000;
                animation: slideIn 0.3s ease-out;
            `;
            notification.textContent = message;
            document.body.appendChild(notification);
            
            // Remove after 3 seconds
            setTimeout(() => {
                notification.style.animation = 'slideOut 0.3s ease-out';
                setTimeout(() => {
                    document.body.removeChild(notification);
                }, 300);
            }, 3000);
        }

        // Add animation styles
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(400px); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(400px); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    </script>
    <script>
        const reportCharts = """ + chart_data_json + """;

        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'bottom' },
                tooltip: { mode: 'index', intersect: false }
            }
        };
        const pieColors = ['#5fd4d6', '#ffb627', '#5fd98c', '#ef6f6f', '#a78bfa', '#fb923c', '#34d399', '#f472b6', '#60a5fa', '#facc15', '#94a3b8', '#f87171'];

        function fmtShort(value) {
            const n = Number(value);
            if (Number.isFinite(n)) {
                return Math.abs(n) >= 1000
                    ? n.toLocaleString(undefined, { maximumFractionDigits: 1 })
                    : n.toFixed(Math.abs(n) < 10 ? 2 : 1).replace(/\\.0$/, '');
            }
            return String(value);
        }

        function boxplotTooltipOptions(chart) {
            return {
                displayColors: false,
                callbacks: {
                    title: (items) => {
                        const label = (items && items[0] && items[0].label) || 'all';
                        return chart.group ? `${chart.group}: ${label}` : (chart.x || 'Box plot');
                    },
                    label: (ctx) => {
                        const label = ctx.label || 'all';
                        const stats = chart.groups && chart.groups[label];
                        if (!stats) return `${ctx.dataset.label || chart.x || 'Value'}: ${fmtShort(ctx.parsed.y)}`;
                        const current = ctx.dataset.label || 'Value';
                        const iqr = Number(stats.q3) - Number(stats.q1);
                        return [
                            `${current}: ${fmtShort(ctx.parsed.y)}`,
                            `Min: ${fmtShort(stats.min)}`,
                            `Q1: ${fmtShort(stats.q1)}`,
                            `Median: ${fmtShort(stats.median)}`,
                            `Q3: ${fmtShort(stats.q3)}`,
                            `Max: ${fmtShort(stats.max)}`,
                            `IQR: ${fmtShort(iqr)}`,
                            `Outliers: ${(stats.outliers || []).length}`,
                        ];
                    }
                }
            };
        }

        function renderHeatmap(canvas, chart) {
            const matrix = chart.matrix || [];
            const labels = chart.labels || [];
            const n = matrix.length;
            if (!n) return false;

            const wrap = canvas.parentElement;
            const rect = wrap.getBoundingClientRect();
            const width = Math.max(640, Math.floor(rect.width || 640));
            const height = Math.max(520, Math.floor(rect.height || 520));
            canvas.width = width;
            canvas.height = height;

            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, width, height);

            const labelSpace = labels.length ? 120 : 36;
            const topSpace = 30;
            const rightSpace = 24;
            const bottomSpace = labels.length ? 130 : 36;
            const plotW = width - labelSpace - rightSpace;
            const plotH = height - topSpace - bottomSpace;
            const size = Math.max(80, Math.min(plotW, plotH));
            const cellSize = size / n;
            const offsetX = labelSpace + Math.max(0, (plotW - size) / 2);
            const offsetY = topSpace + Math.max(0, (plotH - size) / 2);

            let maxAbs = 0;
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < (matrix[i] || []).length; j++) {
                    const v = Number(matrix[i][j]);
                    if (Number.isFinite(v)) maxAbs = Math.max(maxAbs, Math.abs(v));
                }
            }
            if (!maxAbs) maxAbs = 1;

            function cellColor(value) {
                const intensity = Math.min(1, Math.abs(value) / maxAbs);
                if (value >= 0) {
                    const r = Math.round(255 - (255 - 220) * intensity);
                    const g = Math.round(255 - (255 - 53) * intensity);
                    const b = Math.round(255 - (255 - 69) * intensity);
                    return `rgb(${r},${g},${b})`;
                }
                const r = Math.round(255 - (255 - 49) * intensity);
                const g = Math.round(255 - (255 - 130) * intensity);
                const b = Math.round(255 - (255 - 206) * intensity);
                return `rgb(${r},${g},${b})`;
            }

            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    const v = Number((matrix[i] || [])[j] || 0);
                    const x = offsetX + j * cellSize;
                    const y = offsetY + i * cellSize;
                    ctx.fillStyle = cellColor(v);
                    ctx.fillRect(x, y, Math.max(1, cellSize - 1), Math.max(1, cellSize - 1));
                    if (cellSize >= 34) {
                        ctx.fillStyle = Math.abs(v) > maxAbs * 0.55 ? '#fff' : '#212529';
                        ctx.font = `${Math.min(12, cellSize / 3)}px Arial, sans-serif`;
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillText(v.toFixed(2), x + cellSize / 2, y + cellSize / 2);
                    }
                }
            }

            ctx.fillStyle = '#495057';
            ctx.font = '11px Arial, sans-serif';
            for (let i = 0; i < n; i++) {
                const label = String(labels[i] || '');
                ctx.save();
                ctx.translate(offsetX + i * cellSize + cellSize / 2, offsetY + size + 10);
                ctx.rotate(-Math.PI / 4);
                ctx.textAlign = 'right';
                ctx.textBaseline = 'middle';
                ctx.fillText(label, 0, 0);
                ctx.restore();

                ctx.textAlign = 'right';
                ctx.textBaseline = 'middle';
                ctx.fillText(label, offsetX - 8, offsetY + i * cellSize + cellSize / 2);
            }
            return true;
        }

        function buildChartConfig(chart) {
            switch (chart.type) {
                case 'pie':
                    return {
                        type: 'doughnut',
                        data: {
                            labels: chart.labels,
                            datasets: [{
                                data: chart.values,
                                backgroundColor: pieColors,
                                borderColor: '#ffffff',
                                borderWidth: 2
                            }]
                        },
                        options: {
                            ...chartOptions,
                            cutout: '52%',
                            plugins: {
                                ...chartOptions.plugins,
                                legend: {
                                    position: 'right',
                                    labels: { color: '#495057', boxWidth: 12, padding: 12 }
                                },
                                tooltip: {
                                    callbacks: {
                                        label: (ctx) => {
                                            const total = ctx.dataset.data.reduce((a, b) => a + Number(b || 0), 0);
                                            const value = Number(ctx.parsed || 0);
                                            const pct = total ? ` (${(value / total * 100).toFixed(1)}%)` : '';
                                            return `${ctx.label}: ${value}${pct}`;
                                        }
                                    }
                                }
                            }
                        }
                    };
                case 'bar':
                case 'histogram':
                    return {
                        type: 'bar',
                        data: {
                            labels: chart.labels,
                            datasets: [{
                                label: chart.x || 'Values',
                                data: chart.values,
                                backgroundColor: 'rgba(13,110,253,0.75)',
                                borderColor: 'rgba(13,110,253,1)',
                                borderWidth: 1
                            }]
                        },
                        options: {
                            ...chartOptions,
                            scales: { y: { beginAtZero: true } }
                        }
                    };
                case 'line':
                    return {
                        type: 'line',
                        data: {
                            labels: chart.labels,
                            datasets: [{
                                label: chart.y || chart.x || 'Series',
                                data: chart.values,
                                borderColor: 'rgba(13,110,253,0.85)',
                                backgroundColor: 'rgba(13,110,253,0.3)',
                                fill: true,
                                tension: 0.25
                            }]
                        },
                        options: chartOptions
                    };
                case 'scatter':
                    return {
                        type: 'scatter',
                        data: {
                            datasets: [{
                                label: chart.title || `${chart.x} vs ${chart.y}`,
                                data: chart.points,
                                pointRadius: 5,
                                pointBackgroundColor: 'rgba(13,110,253,0.85)',
                                borderColor: 'rgba(13,110,253,1)'
                            }]
                        },
                        options: {
                            ...chartOptions,
                            scales: {
                                x: { type: 'linear', title: { display: true, text: chart.x } },
                                y: { title: { display: true, text: chart.y } }
                            }
                        }
                    };
                case 'boxplot': {
                    if (!chart.groups) return null;
                    const groupEntries = Object.entries(chart.groups);
                    if (!groupEntries.length) return null;
                    const isSimple = chart._boxplotMode === 'simple';
                    const bpLabels = groupEntries.map(([g]) => g);
                    const bpDatasets = isSimple
                        ? [{ label: chart.x || 'Median', data: groupEntries.map(([, s]) => (s && s.median) || 0), backgroundColor: 'rgba(95,212,214,0.73)' }]
                        : [
                            { label: 'Q1',     data: groupEntries.map(([, s]) => (s && s.q1)     || 0), backgroundColor: 'rgba(95,212,214,0.45)' },
                            { label: 'Median', data: groupEntries.map(([, s]) => (s && s.median) || 0), backgroundColor: 'rgba(255,182,39,0.75)'  },
                            { label: 'Q3',     data: groupEntries.map(([, s]) => (s && s.q3)     || 0), backgroundColor: 'rgba(95,217,140,0.45)'  }
                          ];
                    return {
                        type: 'bar',
                        data: { labels: bpLabels, datasets: bpDatasets },
 	                        options: {
 	                            ...chartOptions,
 	                            interaction: { mode: 'nearest', intersect: true },
 	                            plugins: {
 	                                ...chartOptions.plugins,
 	                                legend: { display: !isSimple, labels: { color: '#495057' } },
 	                                tooltip: boxplotTooltipOptions(chart)
 	                            },
 	                            scales: { y: { beginAtZero: true } }
 	                        }
                    };
                }
                case 'wordcloud':
                case 'word_frequency': {
                    if (!chart.words || !chart.words.length) return null;
                    const cloudContainer = document.createElement('div');
                    cloudContainer.className = 'wordcloud-wrap';
                    const counts = chart.words.map(w => w.count);
                    const min = Math.min(...counts), max = Math.max(...counts);
                    const scale = c => min === max ? 24 : 13 + ((c - min) / (max - min)) * 46;
                    cloudContainer.innerHTML = chart.words.map((w, i) =>
                        `<span style="font-size:${scale(w.count).toFixed(0)}px;color:${pieColors[i % pieColors.length]};" title="${w.word}: ${w.count} occurrence(s)">${w.word}</span>`
                    ).join('');
                    canvas.parentElement.insertBefore(cloudContainer, canvas);
                    canvas.remove();
                    return null;
                }
                case 'density': {
                    if (!chart.labels || !chart.labels.length) return null;
                    return {
                        type: 'line',
                        data: {
                            labels: chart.labels,
                            datasets: [{
                                label: chart.x || 'Density',
                                data: chart.values,
                                borderColor: 'rgba(255,182,39,0.85)',
                                backgroundColor: 'rgba(255,182,39,0.2)',
                                fill: true,
                                pointRadius: 0,
                                tension: 0.4
                            }]
                        },
                        options: {
                            ...chartOptions,
                            scales: {
                                x: { ticks: { maxTicksLimit: 10 } },
                                y: { beginAtZero: true }
                            }
                        }
                    };
                }
                case 'violin': {
                    if (!chart.groups || !Object.keys(chart.groups).length) return null;
                    const groupEntries = Object.entries(chart.groups).filter(([, s]) => s);
                    if (!groupEntries.length) return null;
                    
                    const violinLabels = groupEntries.map(([g]) => g);
                    const violinDatasets = [
                        { label: 'Q1', data: groupEntries.map(([, s]) => (s && s.q1) || 0), backgroundColor: 'rgba(95,212,214,0.45)' },
                        { label: 'Median', data: groupEntries.map(([, s]) => (s && s.median) || 0), backgroundColor: 'rgba(255,182,39,0.75)' },
                        { label: 'Q3', data: groupEntries.map(([, s]) => (s && s.q3) || 0), backgroundColor: 'rgba(95,217,140,0.45)' }
                    ];
                    
                    return {
                        type: 'bar',
                        data: { labels: violinLabels, datasets: violinDatasets },
                        options: {
                            ...chartOptions,
                            interaction: { mode: 'nearest', intersect: true },
                            plugins: {
                                ...chartOptions.plugins,
                                legend: { display: true, labels: { color: '#495057' } }
                            },
                            scales: { y: { beginAtZero: true } }
                        }
                    };
                }
                case 'treemap': {
                    if (!chart.labels || !chart.labels.length) return null;
                    const treemapContainer = document.createElement('div');
                    treemapContainer.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px;width:100%;';
                    const total = chart.values.reduce((a, b) => a + b, 0);
                    treemapContainer.innerHTML = chart.labels.map((label, i) => {
                        const share = total ? (chart.values[i] / total * 100).toFixed(1) : '0.0';
                        return `<div style="background:${pieColors[i % pieColors.length]}cc;color:#fff;padding:12px;border-radius:4px;text-align:center;font-family:Arial,sans-serif;">
                            <div style="font-weight:bold;font-size:12px;margin-bottom:4px;">${label.length > 15 ? label.slice(0, 12) + '…' : label}</div>
                            <div style="font-size:10px;opacity:0.9;">${chart.values[i]} (${share}%)</div>
                        </div>`;
                    }).join('');
                    canvas.parentElement.insertBefore(treemapContainer, canvas);
                    canvas.remove();
                    return null;
                }
                case 'radar': {
                    if (!chart.labels || !chart.datasets || !chart.datasets.length) return null;
                    return {
                        type: 'radar',
                        data: {
                            labels: chart.labels,
                            datasets: chart.datasets.map((ds, i) => ({
                                label: ds.label || `Series ${i + 1}`,
                                data: ds.data,
                                borderColor: pieColors[i % pieColors.length],
                                backgroundColor: pieColors[i % pieColors.length] + '33',
                                pointBackgroundColor: pieColors[i % pieColors.length],
                                borderWidth: 2
                            }))
                        },
                        options: {
                            ...chartOptions,
                            scales: {
                                r: {
                                    beginAtZero: true,
                                    ticks: { color: '#495057', backdropColor: 'transparent' },
                                    grid: { color: '#dee2e6' },
                                    pointLabels: { color: '#495057', font: { size: 11 } }
                                }
                            }
                        }
                    };
                }
                case 'scatter_map': {
                    const notice = document.createElement('div');
                    notice.className = 'chart-notice';
                    notice.textContent = 'Scatter maps require an interactive map and cannot be rendered in the static HTML report.';
                    notice.style.marginTop = '20px';
                    canvas.parentElement.insertBefore(notice, canvas);
                    canvas.remove();
                    return null;
                }
                default:
                    return null;
            }
        }

        reportCharts.forEach((chart, index) => {
            const canvas = document.getElementById(`chart_${index}`);
            if (!canvas) return;
            if (chart.type === 'heatmap') {
                if (renderHeatmap(canvas, chart)) return;
            }
            const config = buildChartConfig(chart);
            if (!config) {
                const notice = document.createElement('div');
                notice.className = 'chart-notice';
                notice.textContent = 'This visualization is not rendered as an interactive chart in the downloaded report.';
                canvas.parentElement.insertBefore(notice, canvas);
                canvas.remove();
                return;
            }
            new Chart(canvas, config);
        });
    </script>
</body>
</html>""")
    
    return "\n".join(html_parts)


def _build_work_report(session) -> str:
    profile = eda.profile_dataframe(session.df)
    narrative = narrate.narrate_eda(profile)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Determine how far the user has progressed
    has_cleaning = _has_cleaning_history(session)
    has_training  = bool(session.leaderboard)
    has_unsupervised = bool(session.unsupervised_results)
    has_predictions  = bool(session.saved_predictions)

    if has_training or has_predictions:
        stage = "Model Training & Prediction"
    elif has_unsupervised:
        stage = "Unsupervised Learning"
    elif has_cleaning:
        stage = "Data Cleaning"
    else:
        stage = "Exploratory Data Analysis"

    preview_title = "Cleaned Data — First 15 Rows" if has_cleaning else "Original Data — First 15 Rows"

    lines = [
        f"# AutoDS Work Report: {session.filename}",
        "",
        f"Generated: {generated_at}",
        f"Current stage: **{stage}**",
        "",
        "## Dataset",
        "",
        f"- Rows: {profile['shape']['rows']:,}",
        f"- Columns: {profile['shape']['columns']:,}",
        f"- Duplicate rows: {profile['duplicate_rows']:,}",
        f"- Missing cells: {profile['total_missing_cells']:,} of {profile['total_cells']:,}",
        "",
        "## Summary",
        "",
    ]

    lines.extend(f"- {note}" for note in narrative)
    lines.extend(["", "## Columns", ""])

    for col in profile["columns"]:
        line = (
            f"- `{col['name']}`: {col['type']} ({col['dtype']}), "
            f"{col['unique']:,} unique, {col['missing']:,} missing ({col['missing_pct']}%)"
        )
        stats = col.get("stats")
        if stats:
            line += (
                f", mean {_fmt_report_value(stats.get('mean'))}, "
                f"min {_fmt_report_value(stats.get('min'))}, "
                f"max {_fmt_report_value(stats.get('max'))}"
            )
        top_values = col.get("top_values")
        if top_values:
            top = "; ".join(f"{v['value']}: {v['count']:,}" for v in top_values[:5])
            line += f", top values: {top}"
        lines.append(line)

    corr = profile.get("correlation", {})
    if corr.get("matrix") and len(corr.get("columns", [])) >= 2:
        pairs = []
        corr_cols = corr["columns"]
        matrix = corr["matrix"]
        for i, a in enumerate(corr_cols):
            for j in range(i + 1, len(corr_cols)):
                pairs.append((a, corr_cols[j], matrix[i][j]))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        lines.extend(["", "## Strongest Correlations", ""])
        for a, b, val in pairs[:10]:
            lines.append(f"- `{a}` vs `{b}`: {val:+.3f}")

    if has_cleaning:
        lines.extend(["", "## Cleaning Log", ""])
        lines.extend(f"- {entry}" for entry in session.cleaning_log)
        lines.extend(f"- {entry.get('command', '')}: {entry.get('message', '')}" for entry in session.chat_clean_log)

    if session.leaderboard:
        lines.extend(["", "## Model Training", ""])
        lines.append(f"- Target column: `{session.target}`")
        lines.append(f"- Problem type: {session.problem_type.title() if session.problem_type else 'Unknown'}")
        lines.append(f"- Best model: **{session.best_model_name}**")
        lines.append(f"- Features used: {', '.join(f'`{c}`' for c in session.feature_columns) if session.feature_columns else 'all columns'}")
        lines.append("")
        lines.append("| Rank | Model | Metrics |")
        lines.append("| --- | --- | --- |")
        for i, row in enumerate(session.leaderboard, 1):
            if row.get("error"):
                metrics = f"failed: {row['error']}"
            else:
                metrics = ", ".join(f"{k}: {_fmt_report_value(v)}" for k, v in row.get("metrics", {}).items())
            marker = " ⭐" if row.get("model") == session.best_model_name else ""
            lines.append(f"| {i} | {row.get('model', 'model')}{marker} | {metrics} |")

    if session.saved_predictions:
        lines.extend(["", "## Predictions", ""])
        for prediction in session.saved_predictions.values():
            outputs = ", ".join(_fmt_report_value(v) for v in prediction.get("predictions", [])) or "missing"
            lines.append(f"- **Target:** `{prediction.get('target', 'Prediction')}` → {outputs}")
            lines.append(
                f"  - Model: {prediction.get('source_name', 'Current training')} "
                f"({prediction.get('model_name', 'model')})"
            )
            if prediction.get("created_at"):
                lines.append(f"  - Saved: {prediction['created_at']}")
            lines.append(f"  - Inputs: {_fmt_prediction_inputs(prediction.get('inputs', []))}")
            if prediction.get("narrative"):
                lines.append(f"  - Note: {prediction['narrative']}")

    if session.unsupervised_results:
        lines.extend(["", "## Unsupervised Learning", ""])
        u = session.unsupervised_results

        preprocessing = (u.get("suggestions") or {}).get("preprocessing") or {}
        if preprocessing:
            features = preprocessing.get("features_used") or []
            lines.append(f"**Preprocessing:** {preprocessing.get('scaling', 'StandardScaler')} scaling on {len(features):,} numeric feature(s)")
            if features:
                lines.append(f"- Features: {', '.join(f'`{c}`' for c in features[:15])}{'...' if len(features) > 15 else ''}")
            lines.append("")

        cluster_analysis = u.get("cluster_analysis")
        if cluster_analysis:
            lines.append("**Cluster Number Analysis**")
            lines.append(
                f"- Best silhouette K: {cluster_analysis.get('best_silhouette_k', '?')}; "
                f"best Davies-Bouldin K: {cluster_analysis.get('best_db_k', '?')}; "
                f"elbow K: {cluster_analysis.get('elbow_k', '?')}; "
                f"best Calinski-Harabasz K: {cluster_analysis.get('best_ch_k', '?')}"
            )
            rows_k = cluster_analysis.get("k_analysis") or []
            if rows_k:
                lines.append("")
                lines.append("| K | Inertia | Silhouette | Davies-Bouldin | Calinski-Harabasz | Votes |")
                lines.append("| --- | --- | --- | --- | --- | --- |")
                for row in rows_k:
                    lines.append(
                        f"| {row.get('k')} | {_fmt_report_value(row.get('inertia'))} "
                        f"| {_fmt_report_value(row.get('silhouette'))} "
                        f"| {_fmt_report_value(row.get('davies_bouldin'))} "
                        f"| {_fmt_report_value(row.get('calinski_harabasz'))} "
                        f"| {row.get('votes', '')} |"
                    )
            lines.append("")

        clustering = u.get("clustering")
        if clustering:
            method = clustering.get("selected_method") or clustering.get("method", "Clustering")
            lines.append(f"**{method}** (chosen method)")
            reason = clustering.get("selection_reason", "")
            if reason:
                lines.append(f"- {reason}")
            if clustering.get("n_rows_scored"):
                lines.append(f"- Rows scored: {int(clustering['n_rows_scored']):,}")
            sizes = clustering.get("cluster_sizes") or {}
            if sizes:
                lines.append("")
                lines.append("| Cluster | Rows |")
                lines.append("| --- | --- |")
                for cname, count in sizes.items():
                    lines.append(f"| {cname} | {int(count):,} |")
            metrics = clustering.get("metrics") or {}
            if metrics:
                lines.append("")
                lines.append("Metrics: " + ", ".join(f"{k}: {_fmt_report_value(v)}" for k, v in metrics.items()))
            lines.append("")

        anomaly = u.get("anomaly")
        if anomaly:
            lines.append(f"**{anomaly.get('method', 'Anomaly Detection')}**")
            lines.append(f"- Anomalies found: {int(anomaly.get('n_outliers', 0)):,} ({_fmt_report_value(anomaly.get('outlier_percentage', 0))}% of data)")
            lines.append(f"- Normal rows: {int(anomaly.get('n_normal', 0)):,}")
            lines.append("")

        reduction = u.get("reduction")
        if reduction:
            lines.append(f"**{reduction.get('method', 'Dimensionality Reduction')}**")
            lines.append(f"- Components: {reduction.get('n_components', '?')}")
            lines.append(f"- Points scored: {int(reduction.get('n_rows_scored', 0)):,}")
            ev = reduction.get("explained_variance") or []
            if ev:
                lines.append("- Explained variance: " + ", ".join(f"PC{i+1}: {_fmt_report_value(v)}%" for i, v in enumerate(ev)))
            lines.append("")

        association = u.get("association")
        if association:
            lines.append(f"**{association.get('method', 'Association Rules')}**")
            lines.append(f"- Rules found: {int(association.get('n_rules', 0)):,}")
            lines.append(f"- Min support: {_fmt_report_value(association.get('min_support'))} | Min confidence: {_fmt_report_value(association.get('min_confidence'))}")
            rules = association.get("rules") or []
            if rules:
                lines.append("")
                lines.append("| Antecedents | Consequents | Support | Confidence | Lift |")
                lines.append("| --- | --- | --- | --- | --- |")
                for rule in rules[:10]:
                    ant = ", ".join(str(x) for x in (rule.get("antecedents") or []))
                    con = ", ".join(str(x) for x in (rule.get("consequents") or []))
                    lines.append(f"| {ant} | {con} | {_fmt_report_value(rule.get('support'))} | {_fmt_report_value(rule.get('confidence'))} | {_fmt_report_value(rule.get('lift'))} |")
            lines.append("")

    lines.extend(["", f"## {preview_title}", ""])
    preview = eda.safe_preview(session.df, 15)
    if preview:
        headers = list(preview[0].keys())
        lines.append("| " + " | ".join(map(str, headers)) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in preview:
            vals = [_fmt_report_value(row.get(h)).replace("|", "\\|") for h in headers]
            lines.append("| " + " | ".join(vals) + " |")
    else:
        lines.append("No preview rows available.")

    lines.append("")
    return "\n".join(lines)