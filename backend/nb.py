"""
Generate a compact Jupyter Notebook (.ipynb) for the work completed in a
session. The export intentionally avoids dumping every AutoDS helper function;
it includes only the successful user-facing results recorded on the session.
"""
from __future__ import annotations  # makes list|None work on Python < 3.10 (fixes VS Code squiggles)
import base64
import io
import json
from datetime import datetime
from typing import Any

# Country name -> ISO-3 code, matching the exact canonical names used by
# backend/geo_data/world_countries.geojson (see geo.py) -- lets exported
# notebooks build a Plotly choropleth without needing that boundary file.
_NAME_TO_ISO3 = {
    "Zimbabwe": "ZWE",
    "Zambia": "ZMB",
    "Yemen": "YEM",
    "Vietnam": "VNM",
    "Venezuela": "VEN",
    "Vatican": "VAT",
    "Vanuatu": "VUT",
    "Uzbekistan": "UZB",
    "Uruguay": "URY",
    "Federated States of Micronesia": "FSM",
    "Marshall Islands": "MHL",
    "Northern Mariana Islands": "MNP",
    "United States Virgin Islands": "VIR",
    "Guam": "GUM",
    "American Samoa": "ASM",
    "Puerto Rico": "PRI",
    "United States of America": "USA",
    "South Georgia and the Islands": "SGS",
    "British Indian Ocean Territory": "IOT",
    "Saint Helena": "SHN",
    "Pitcairn Islands": "PCN",
    "Anguilla": "AIA",
    "Falkland Islands": "FLK",
    "Cayman Islands": "CYM",
    "Bermuda": "BMU",
    "British Virgin Islands": "VGB",
    "Turks and Caicos Islands": "TCA",
    "Montserrat": "MSR",
    "Jersey": "JEY",
    "Guernsey": "GGY",
    "Isle of Man": "IMN",
    "United Kingdom": "GBR",
    "United Arab Emirates": "ARE",
    "Ukraine": "UKR",
    "Uganda": "UGA",
    "Turkmenistan": "TKM",
    "Turkey": "TUR",
    "Tunisia": "TUN",
    "Trinidad and Tobago": "TTO",
    "Tonga": "TON",
    "Togo": "TGO",
    "East Timor": "TLS",
    "Thailand": "THA",
    "United Republic of Tanzania": "TZA",
    "Tajikistan": "TJK",
    "Taiwan": "TWN",
    "Syria": "SYR",
    "Switzerland": "CHE",
    "Sweden": "SWE",
    "Swaziland": "SWZ",
    "Suriname": "SUR",
    "South Sudan": "SSD",
    "Sudan": "SDN",
    "Sri Lanka": "LKA",
    "Spain": "ESP",
    "South Korea": "KOR",
    "South Africa": "ZAF",
    "Somalia": "SOM",
    "Somaliland": "SOL",
    "Solomon Islands": "SLB",
    "Slovakia": "SVK",
    "Slovenia": "SVN",
    "Singapore": "SGP",
    "Sierra Leone": "SLE",
    "Seychelles": "SYC",
    "Republic of Serbia": "SRB",
    "Senegal": "SEN",
    "Saudi Arabia": "SAU",
    "São Tomé and Principe": "STP",
    "San Marino": "SMR",
    "Samoa": "WSM",
    "Saint Vincent and the Grenadines": "VCT",
    "Saint Lucia": "LCA",
    "Saint Kitts and Nevis": "KNA",
    "Rwanda": "RWA",
    "Russia": "RUS",
    "Romania": "ROU",
    "Qatar": "QAT",
    "Portugal": "PRT",
    "Poland": "POL",
    "Philippines": "PHL",
    "Peru": "PER",
    "Paraguay": "PRY",
    "Papua New Guinea": "PNG",
    "Panama": "PAN",
    "Palau": "PLW",
    "Pakistan": "PAK",
    "Oman": "OMN",
    "Norway": "NOR",
    "North Korea": "PRK",
    "Nigeria": "NGA",
    "Niger": "NER",
    "Nicaragua": "NIC",
    "New Zealand": "NZL",
    "Niue": "NIU",
    "Cook Islands": "COK",
    "Netherlands": "NLD",
    "Aruba": "ABW",
    "Curaçao": "CUW",
    "Nepal": "NPL",
    "Nauru": "NRU",
    "Namibia": "NAM",
    "Mozambique": "MOZ",
    "Morocco": "MAR",
    "Western Sahara": "ESH",
    "Montenegro": "MNE",
    "Mongolia": "MNG",
    "Moldova": "MDA",
    "Monaco": "MCO",
    "Mexico": "MEX",
    "Mauritius": "MUS",
    "Mauritania": "MRT",
    "Malta": "MLT",
    "Mali": "MLI",
    "Maldives": "MDV",
    "Malaysia": "MYS",
    "Malawi": "MWI",
    "Madagascar": "MDG",
    "Macedonia": "MKD",
    "Luxembourg": "LUX",
    "Lithuania": "LTU",
    "Liechtenstein": "LIE",
    "Libya": "LBY",
    "Liberia": "LBR",
    "Lesotho": "LSO",
    "Lebanon": "LBN",
    "Latvia": "LVA",
    "Laos": "LAO",
    "Kyrgyzstan": "KGZ",
    "Kuwait": "KWT",
    "Kosovo": "KOS",
    "Kiribati": "KIR",
    "Kenya": "KEN",
    "Kazakhstan": "KAZ",
    "Jordan": "JOR",
    "Japan": "JPN",
    "Jamaica": "JAM",
    "Italy": "ITA",
    "Israel": "ISR",
    "Palestine": "PSE",
    "Ireland": "IRL",
    "Iraq": "IRQ",
    "Iran": "IRN",
    "Indonesia": "IDN",
    "India": "IND",
    "Iceland": "ISL",
    "Hungary": "HUN",
    "Honduras": "HND",
    "Haiti": "HTI",
    "Guyana": "GUY",
    "Guinea-Bissau": "GNB",
    "Guinea": "GIN",
    "Guatemala": "GTM",
    "Grenada": "GRD",
    "Greece": "GRC",
    "Ghana": "GHA",
    "Germany": "DEU",
    "Georgia": "GEO",
    "Gambia": "GMB",
    "Gabon": "GAB",
    "France": "FRA",
    "Saint Pierre and Miquelon": "SPM",
    "Wallis and Futuna": "WLF",
    "Saint Martin": "MAF",
    "Saint Barthelemy": "BLM",
    "French Polynesia": "PYF",
    "New Caledonia": "NCL",
    "French Southern and Antarctic Lands": "ATF",
    "Aland": "ALA",
    "Finland": "FIN",
    "Fiji": "FJI",
    "Ethiopia": "ETH",
    "Estonia": "EST",
    "Eritrea": "ERI",
    "Equatorial Guinea": "GNQ",
    "El Salvador": "SLV",
    "Egypt": "EGY",
    "Ecuador": "ECU",
    "Dominican Republic": "DOM",
    "Dominica": "DMA",
    "Djibouti": "DJI",
    "Greenland": "GRL",
    "Faroe Islands": "FRO",
    "Denmark": "DNK",
    "Czech Republic": "CZE",
    "Northern Cyprus": "CYN",
    "Cyprus": "CYP",
    "Cuba": "CUB",
    "Croatia": "HRV",
    "Ivory Coast": "CIV",
    "Costa Rica": "CRI",
    "Democratic Republic of the Congo": "COD",
    "Republic of the Congo": "COG",
    "Comoros": "COM",
    "Colombia": "COL",
    "China": "CHN",
    "Macao S.A.R": "MAC",
    "Hong Kong S.A.R.": "HKG",
    "Chile": "CHL",
    "Chad": "TCD",
    "Central African Republic": "CAF",
    "Cape Verde": "CPV",
    "Canada": "CAN",
    "Cameroon": "CMR",
    "Cambodia": "KHM",
    "Myanmar": "MMR",
    "Burundi": "BDI",
    "Burkina Faso": "BFA",
    "Bulgaria": "BGR",
    "Brunei": "BRN",
    "Brazil": "BRA",
    "Botswana": "BWA",
    "Bosnia and Herzegovina": "BIH",
    "Bolivia": "BOL",
    "Bhutan": "BTN",
    "Benin": "BEN",
    "Belize": "BLZ",
    "Belgium": "BEL",
    "Belarus": "BLR",
    "Barbados": "BRB",
    "Bangladesh": "BGD",
    "Bahrain": "BHR",
    "The Bahamas": "BHS",
    "Azerbaijan": "AZE",
    "Austria": "AUT",
    "Australia": "AUS",
    "Indian Ocean Territories": "AUS",
    "Heard Island and McDonald Islands": "HMD",
    "Norfolk Island": "NFK",
    "Ashmore and Cartier Islands": "AUS",
    "Armenia": "ARM",
    "Argentina": "ARG",
    "Antigua and Barbuda": "ATG",
    "Angola": "AGO",
    "Andorra": "AND",
    "Algeria": "DZA",
    "Albania": "ALB",
    "Afghanistan": "AFG",
    "Siachen Glacier": "KAS",
    "Antarctica": "ATA",
    "Sint Maarten": "SXM",
    "Tuvalu": "TUV"
}

# Reverse: ISO-3 code -> full lowercase country name.
# Used when a dataset column already contains codes like "VEN", "USA", "GBR"
# rather than spelled-out names -- the choropleth still renders, and hover
# labels show the readable name instead of the raw code.
_ISO3_TO_NAME: dict[str, str] = {code: name.lower() for name, code in _NAME_TO_ISO3.items()}

# Full US state name -> USPS 2-letter code, for Plotly's 'USA-states' locationmode.
_STATE_TO_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC"
}


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _json_literal(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=_json_default)


def _py_embed(value: Any) -> str:
    """
    Embed a Python value into generated notebook *source code* so it comes
    back out as the same value when the cell runs.

    _json_literal's output looks like Python (dicts/lists/strings all look
    the same in both), but it isn't: JSON's `null`/`true`/`false` are not
    valid Python — they're `None`/`True`/`False`. Any chart or result
    containing a None (e.g. a histogram with no x_min/x_max set) got
    embedded as literal `null` in the .ipynb source, which crashed with
    `NameError: name 'null' is not defined` the moment that cell ran.

    This instead serializes to a JSON *string*, double-encodes it into a
    valid Python string literal, and has the notebook parse it back with
    `json.loads(...)` at run time — so the round trip is exact regardless
    of None/True/False/nested values, and it's still just as inspectable
    since the underlying JSON is unaffected.
    """
    json_str = json.dumps(value, ensure_ascii=False, default=_json_default)
    return f"json.loads({json.dumps(json_str)})"


def _comment_lines(lines: list[str]) -> str:
    return "\n".join(f"# {line}" if line else "#" for line in lines)


def _training_runs(session) -> list[dict]:
    runs = []
    for run_id, run in (getattr(session, "saved_runs", None) or {}).items():
        runs.append({
            "run_id": run_id,
            "name": run.get("name") or f"Saved run: {run.get('target', '')}",
            "created_at": run.get("created_at"),
            "target": run.get("target"),
            "problem_type": run.get("problem_type"),
            "best_model_name": run.get("best_model_name"),
            "feature_columns": run.get("feature_columns") or [],
            "leaderboard": run.get("leaderboard") or [],
            "is_current": False,
        })

    if getattr(session, "leaderboard", None):
        runs.append({
            "run_id": "current",
            "name": f"Current training: {session.target}",
            "created_at": None,
            "target": session.target,
            "problem_type": session.problem_type,
            "best_model_name": session.best_model_name,
            "feature_columns": session.feature_columns or [],
            "leaderboard": session.leaderboard or [],
            "is_current": True,
        })
    return runs


def _has_completed_work(session, charts: list | None = None) -> bool:
    return any([
        getattr(session, "notes", None) and session.notes.strip(),
        getattr(session, "cleaning_log", None),
        getattr(session, "chat_clean_log", None),
        getattr(session, "leaderboard", None),
        getattr(session, "saved_runs", None),
        getattr(session, "saved_predictions", None),
        getattr(session, "unsupervised_results", None),
        charts,
        getattr(session, "last_visualization", None),
    ])


def _add_notes_cell(cells: list[dict], session):
    notes = (getattr(session, "notes", "") or "").strip()
    if not notes:
        return
    cells.append(_markdown_cell(f"## Notes\n\n{notes}"))


def _add_data_cell(cells: list[dict], session):
    csv_buf = io.StringIO()
    session.df.to_csv(csv_buf, index=False)
    csv_data_b64 = base64.b64encode(csv_buf.getvalue().encode("utf-8")).decode("ascii")
    cells.append(_code_cell(
        "import base64\n"
        "import io\n"
        "import json\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        f"CSV_DATA_B64 = {json.dumps(csv_data_b64)}\n\n"
        "df = pd.read_csv(io.BytesIO(base64.b64decode(CSV_DATA_B64)))\n"
        "print(f'Loaded final AutoDS data: {len(df):,} rows x {len(df.columns)} columns')\n"
        "df.head()"
    ))


def _add_cleaning_cells(cells: list[dict], session):
    cleaning_log = getattr(session, "cleaning_log", None) or []
    chat_log = getattr(session, "chat_clean_log", None) or []
    if not cleaning_log and not chat_log:
        return

    lines = ["Successful cleaning/actions recorded in AutoDS:"]
    lines.extend(f"- {entry}" for entry in cleaning_log)
    lines.extend(
        f"- Chat command: {entry.get('command', '')} -> {entry.get('message', '')}"
        for entry in chat_log
    )
    cells.append(_markdown_cell("## Cleaning Done"))
    cells.append(_code_cell(
        _comment_lines(lines) + "\n\n"
        "# The exported dataframe above is already the cleaned final result.\n"
        "cleaned_df = df.copy()\n"
        "cleaned_df.to_csv('autods_cleaned_data.csv', index=False)\n"
        "print(cleaned_df.shape)\n"
    ))


def _add_visualization_cells(cells: list[dict], session, charts: list | None = None):
    # Prefer the full chart history sent from the frontend (customChartSpecs —
    # every visualization the user built for this dataset). Fall back to just
    # the most recent one if no history was passed in (e.g. old GET calls).
    all_charts = charts if charts else ([getattr(session, "last_visualization", None)] if getattr(session, "last_visualization", None) else [])
    all_charts = [c for c in all_charts if c]
    if not all_charts:
        return

    cells.append(_markdown_cell(f"## Visualizations Done ({len(all_charts)})"))
    for i, chart in enumerate(all_charts, start=1):
        cells.append(_code_cell(
            "import json\n"
            "import matplotlib.pyplot as plt\n\n"
            f"chart = {_py_embed(chart)}\n"
            "chart_type = chart.get('type')\n"
            "x = chart.get('x')\n"
            "y = chart.get('y')\n"
            "_plotted = False  # set True below once a chart type renders via a non-matplotlib path\n\n"
            "if chart_type in {'scatter', 'line'} and x in df.columns and y in df.columns:\n"
            "    plot_df = df[[x, y]].dropna()\n"
            "    if chart_type == 'line':\n"
            "        plot_df = plot_df.sort_values(x)\n"
            "        plt.plot(plot_df[x], plot_df[y], marker='o')\n"
            "    else:\n"
            "        plt.scatter(plot_df[x], plot_df[y], alpha=0.7)\n"
            "    plt.xlabel(x)\n"
            "    plt.ylabel(y)\n"
            "elif chart_type in {'histogram', 'boxplot'} and x in df.columns:\n"
            "    if chart_type == 'boxplot':\n"
            "        df[x].dropna().plot(kind='box')\n"
            "    else:\n"
            "        df[x].dropna().hist(bins=12)\n"
            "    plt.xlabel(x)\n"
            "elif chart_type in {'bar', 'word_frequency'} and chart.get('labels'):\n"
            "    labels = chart.get('labels', [])\n"
            "    values = chart.get('values', [])\n"
            "    plt.barh(labels[::-1], values[::-1])\n"
            "    plt.xlabel('Count')\n"
            "elif chart_type == 'wordcloud' and chart.get('words'):\n"
            "    words = chart.get('words', [])\n"
            "    labels = [w.get('word') for w in words][:25][::-1]\n"
            "    values = [w.get('count') for w in words][:25][::-1]\n"
            "    plt.barh(labels, values)\n"
            "    plt.xlabel('Count')\n"
            "elif chart_type == 'choropleth' and chart.get('rows'):\n"
            "    # Renders as an actual interactive map via Plotly Express, which ships\n"
            "    # its own world/US-state boundaries -- no geopandas or shapefile needed.\n"
            "    # Falls back to a static ranked bar chart if plotly isn't installed, or\n"
            "    # if this map used a custom-uploaded .geojson whose shapes Plotly has no\n"
            "    # built-in knowledge of.\n"
            "    value_col = chart.get('value_col')\n"
            "    name_col = chart.get('name_col')\n"
            "    level = chart.get('level')\n"
            "    rows = [r for r in chart['rows'] if r.get(value_col) is not None and r.get(name_col)]\n"
            "    _NAME_TO_ISO3 = " + _py_embed(_NAME_TO_ISO3) + "\n"
            "    _ISO3_TO_NAME = " + _py_embed(_ISO3_TO_NAME) + "\n"
            "    _STATE_TO_ABBR = " + _py_embed(_STATE_TO_ABBR) + "\n"
            "    try:\n"
            "        import plotly.express as px\n"
            "        if level == 'world':\n"
            "            # Detect whether the column holds ISO-3 codes (e.g. 'VEN', 'USA')\n"
            "            # or full/partial country names -- handle both transparently.\n"
            "            sample_vals = [str(r.get(name_col, '')).strip().upper() for r in rows[:10]]\n"
            "            col_has_iso3 = sum(1 for v in sample_vals if v in _ISO3_TO_NAME) >= len(sample_vals) // 2\n"
            "            for r in rows:\n"
            "                raw = str(r.get(name_col, '')).strip()\n"
            "                if col_has_iso3:\n"
            "                    # Column already has ISO-3 codes -- use directly for Plotly;\n"
            "                    # also set a display-friendly full lowercase name for hover.\n"
            "                    iso3 = raw.upper()\n"
            "                    r['_loc'] = iso3 if iso3 in _ISO3_TO_NAME else None\n"
            "                    r['_display_name'] = _ISO3_TO_NAME.get(iso3, raw.lower())\n"
            "                else:\n"
            "                    # Column has country names -- look up the ISO-3 code.\n"
            "                    r['_loc'] = _NAME_TO_ISO3.get(raw)\n"
            "                    r['_display_name'] = raw.lower()\n"
            "            hover_col = '_display_name'\n"
            "            locationmode, scope = 'ISO-3', 'world'\n"
            "        elif level == 'us_states':\n"
            "            for r in rows:\n"
            "                raw = str(r.get(name_col)).strip()\n"
            "                r['_loc'] = _STATE_TO_ABBR.get(raw, raw.upper() if len(raw) == 2 else None)\n"
            "                r['_display_name'] = raw.lower()\n"
            "            hover_col = '_display_name'\n"
            "            locationmode, scope = 'USA-states', 'usa'\n"
            "        else:\n"
            "            raise ValueError('this map used a custom .geojson -- Plotly has no built-in boundaries for it')\n"
            "        mapped_rows = [r for r in rows if r.get('_loc')]\n"
            "        if not mapped_rows:\n"
            "            raise ValueError(\"none of the region names matched Plotly's built-in boundaries\")\n"
            "        fig = px.choropleth(\n"
            "            mapped_rows, locations='_loc', locationmode=locationmode, color=value_col,\n"
            "            hover_name=hover_col, color_continuous_scale='Viridis', scope=scope,\n"
            f"            title=chart.get('title') or 'AutoDS chart {i}',\n"
            "        )\n"
            "        skipped = len(rows) - len(mapped_rows)\n"
            "        if skipped:\n"
            "            print(f'{skipped} region(s) had no matching Plotly boundary and were left blank on the map.')\n"
            "        fig.show()\n"
            "        _plotted = True\n"
            "    except ImportError:\n"
            "        print('plotly is not installed (pip install plotly) -- showing a ranked bar chart instead.')\n"
            "    except Exception as _map_err:\n"
            "        print(f'Could not build an interactive map ({_map_err}) -- showing a ranked bar chart instead.')\n"
            "    if not _plotted:\n"
            "        rows.sort(key=lambda r: r[value_col], reverse=True)\n"
            "        labels = [str(r.get(name_col, '?')) for r in rows][:20][::-1]\n"
            "        values = [r[value_col] for r in rows][:20][::-1]\n"
            "        plt.barh(labels, values)\n"
            "        plt.xlabel(value_col)\n"
            "elif x in df.columns:\n"
            "    df[x].value_counts(dropna=True).head(15).sort_values().plot(kind='barh')\n"
            "    plt.xlabel('Count')\n"
            "else:\n"
            "    print('The recorded chart columns are not present in the exported data.')\n\n"
            "if not _plotted:\n"
            f"    plt.title(chart.get('title') or 'AutoDS chart {i}')\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
        ))


def _add_training_cells(cells: list[dict], session):
    runs = _training_runs(session)
    if not runs:
        return

    cells.append(_markdown_cell("## Model Training Done"))
    cells.append(_code_cell(
        "import json\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n\n"
        "def train_replay(df, target, problem_type):\n"
        "    data = df.dropna(subset=[target]).copy()\n"
        "    X = data.drop(columns=[target])\n"
        "    y = data[target]\n"
        "    numeric_cols = X.select_dtypes(include='number').columns.tolist()\n"
        "    categorical_cols = [c for c in X.columns if c not in numeric_cols]\n"
        "    preprocessor = ColumnTransformer([\n"
        "        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric_cols),\n"
        "        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols),\n"
        "    ])\n"
        "    if problem_type == 'classification':\n"
        "        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced')\n"
        "        stratify = y if y.value_counts().min() >= 2 else None\n"
        "    else:\n"
        "        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)\n"
        "        stratify = None\n"
        "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)\n"
        "    pipe = Pipeline([('prep', preprocessor), ('model', model)])\n"
        "    pipe.fit(X_train, y_train)\n"
        "    preds = pipe.predict(X_test)\n"
        "    if problem_type == 'classification':\n"
        "        print('accuracy:', accuracy_score(y_test, preds))\n"
        "        print('f1_weighted:', f1_score(y_test, preds, average='weighted', zero_division=0))\n"
        "    else:\n"
        "        print('mae:', mean_absolute_error(y_test, preds))\n"
        "        print('r2:', r2_score(y_test, preds))\n"
        "    return pipe\n"
    ))

    for run in runs:
        cells.append(_code_cell(
            f"run = {_py_embed(run)}\n\n"
            "print(run['name'])\n"
            "print('target:', run['target'])\n"
            "print('problem_type:', run['problem_type'])\n"
            "print('best_model_recorded_by_autods:', run['best_model_name'])\n"
            "leaderboard = pd.DataFrame(run['leaderboard'])\n"
            "display(leaderboard)\n\n"
            "# Runnable replay using the exported final data and the same target.\n"
            "# AutoDS originally tested several models; the recorded leaderboard above is the source of truth.\n"
            "model = train_replay(df, run['target'], run['problem_type'])\n"
        ))


def _add_prediction_cells(cells: list[dict], session):
    predictions = getattr(session, "saved_predictions", None) or {}
    if not predictions:
        return

    cells.append(_markdown_cell("## Predictions Done"))
    cells.append(_code_cell(
        "import json\n\n"
        f"saved_predictions = {_py_embed(list(predictions.values()))}\n\n"
        "for prediction in saved_predictions:\n"
        "    print('target:', prediction.get('target'))\n"
        "    print('model:', prediction.get('model_name'))\n"
        "    print('inputs:', prediction.get('inputs'))\n"
        "    print('predictions:', prediction.get('predictions'))\n"
        "    print()\n"
    ))


def _add_unsupervised_cells(cells: list[dict], session):
    results = getattr(session, "unsupervised_results", None) or {}
    meaningful = {k: v for k, v in results.items() if k != "suggestions" and v}
    if not meaningful:
        return

    cells.append(_markdown_cell("## Unsupervised Analysis Done"))
    cells.append(_code_cell(
        "import json\n\n"
        f"unsupervised_results = {_py_embed(meaningful)}\n\n"
        "for name, result in unsupervised_results.items():\n"
        "    print(f'[{name}]')\n"
        "    for key, value in result.items():\n"
        "        if key in {'points', 'rules', 'rows'}:\n"
        "            print(f'{key}: {len(value) if hasattr(value, \"__len__\") else value}')\n"
        "        else:\n"
        "            print(f'{key}: {value}')\n"
        "    print()\n"
    ))


def build_notebook(session, charts: list | None = None) -> str:
    """Build a compact .ipynb JSON string for completed session work.
    `charts` is the frontend's full chart history for this dataset (same list
    used by the HTML report) so every visualization gets its own cell, not
    just the most recently created one."""
    name = session.filename.rsplit(".", 1)[0]
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = "AutoDS - Completed Work Notebook" if _has_completed_work(session, charts) else "AutoDS - Data Notebook"

    cells = [
        _markdown_cell(
            f"# {title}\n\n"
            f"**Dataset:** {session.filename}  \n"
            f"**Generated:** {generated_at}  \n"
            f"**Rows:** {len(session.df):,}  **Columns:** {len(session.df.columns)}\n\n"
            "This notebook keeps the export focused on the Python code and results from this session."
        )
    ]

    _add_data_cell(cells, session)
    _add_notes_cell(cells, session)
    _add_cleaning_cells(cells, session)
    _add_visualization_cells(cells, session, charts)
    _add_training_cells(cells, session)
    _add_prediction_cells(cells, session)
    _add_unsupervised_cells(cells, session)

    if len(cells) == 2:
        cells.append(_code_cell(
            "# No cleaning, training, prediction, visualization, or unsupervised actions were recorded yet.\n"
            "# The cell above contains your current dataset, ready for your own Python analysis.\n"
            "df.describe(include='all')"
        ))

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": cells,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False)