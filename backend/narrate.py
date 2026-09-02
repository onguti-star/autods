"""
Plain-English narration. Turns the numeric profile / leaderboard dicts
into short, human-readable explanations — no external API call, just
rules over the stats we've already computed, so it's instant and free
to run on every step.
"""
from typing import List


def _pct(n, d):
    return round((n / d) * 100, 1) if d else 0


# Keyword sets used to guess what a dataset is "about" from its column names.
DOMAIN_KEYWORDS = {
    "hotel / hospitality booking": [
        "hotel", "room", "checkin", "check_in", "checkout", "check_out", "guest",
        "reservation", "booking", "nights", "adr", "lead_time", "stay",
    ],
    "retail / e-commerce sales": [
        "product", "sku", "order", "cart", "discount", "quantity", "price",
        "customer_id", "purchase", "category", "revenue", "sales",
    ],
    "banking / finance": [
        "loan", "credit", "balance", "transaction", "account", "interest",
        "default", "income", "debt", "mortgage",
    ],
    "healthcare / patient": [
        "patient", "diagnosis", "treatment", "symptom", "blood", "hospital",
        "disease", "doctor", "medication", "bmi",
    ],
    "human resources / employee": [
        "employee", "salary", "department", "tenure", "attrition", "manager",
        "performance", "hire_date", "promotion",
    ],
    "real estate / property": [
        "bedroom", "bathroom", "sqft", "square_feet", "property", "zipcode",
        "lot_size", "listing", "rent", "house",
    ],
    "marketing / campaigns": [
        "campaign", "click", "impression", "ctr", "conversion", "channel",
        "spend", "leads", "engagement",
    ],
    "education / students": [
        "student", "grade", "course", "exam", "school", "gpa", "enrollment",
        "teacher", "attendance",
    ],
    "travel / airline": [
        "flight", "airline", "departure", "arrival", "fare", "passenger",
        "airport", "ticket",
    ],
}


def guess_domain(columns: list) -> str | None:
    """Best-effort guess at what the dataset is about, from column names."""
    col_text = " ".join(c.lower() for c in columns)
    best_domain, best_score = None, 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in col_text)
        if score > best_score:
            best_domain, best_score = domain, score
    if best_score >= 2:  # require at least 2 matching keywords to avoid false positives
        return best_domain
    return None


def narrate_eda(profile: dict) -> List[str]:
    """Returns a list of short paragraphs explaining what's in the dataset."""
    notes = []
    rows, cols = profile["shape"]["rows"], profile["shape"]["columns"]
    columns = profile["columns"]
    col_names = [c["name"] for c in columns]

    # 0. Domain guess — the "this looks like hotel data..." opener
    domain = guess_domain(col_names)
    if domain:
        notes.append(
            f"Based on the column names, this looks like {domain} data — "
            f"{rows:,} rows and {cols} columns."
        )
    else:
        notes.append(
            f"This dataset has {rows:,} rows and {cols} columns."
        )

    # 2. Duplicates
    if profile["duplicate_rows"] > 0:
        notes.append(
            f"{profile['duplicate_rows']} rows ({_pct(profile['duplicate_rows'], rows)}%) "
            f"are exact duplicates of another row — worth checking whether that's expected "
            f"or you accidentally loaded the same records twice."
        )

    # 3. Missingness — call out the worst offenders
    missing_cols = sorted(
        [c for c in columns if c["missing"] > 0],
        key=lambda c: -c["missing_pct"]
    )
    if missing_cols:
        worst = missing_cols[:3]
        listing = ", ".join(f"{c['name']} ({c['missing_pct']}%)" for c in worst)
        notes.append(
            f"{len(missing_cols)} column(s) have missing values. The most affected: {listing}. "
            f"During training, missing numeric values get filled with the median and missing "
            f"categories with the most common value, so you don't need to clean this by hand."
        )
    else:
        notes.append("No missing values anywhere — this dataset is complete.")

    # 4. Column type mix
    numeric_cols = [c for c in columns if c["type"] == "numeric"]
    categorical_cols = [c for c in columns if c["type"] == "categorical"]
    text_cols = [c for c in columns if c["type"] == "text"]
    type_sentence = f"{len(numeric_cols)} column(s) are numeric and {len(categorical_cols)} are categorical (text/labels)."
    if text_cols:
        type_sentence += f" {len(text_cols)} column(s) look like free-form text (reviews, comments, descriptions)."
    notes.append(type_sentence)

    # 4b. Free-text column details
    for c in text_cols:
        stats = c.get("text_stats", {})
        top_words = stats.get("top_words", [])[:5]
        words_note = ", ".join(w["word"] for w in top_words) if top_words else "no clear top words"
        notes.append(
            f"'{c['name']}' averages {stats.get('avg_words', 0)} words per entry across "
            f"{stats.get('documents', 0):,} non-empty entries, with a vocabulary of "
            f"{stats.get('vocab_size', 0):,} unique words. Most frequent words: {words_note}."
        )

    # 5. High-cardinality categorical warning (e.g. IDs, free text)
    high_card = [c for c in categorical_cols if c["unique"] > 0.8 * rows and rows > 20]
    if high_card:
        names = ", ".join(c["name"] for c in high_card[:3])
        notes.append(
            f"{names} look like unique identifiers or free text (almost every value is "
            f"different) — they likely won't help a model predict anything and are good "
            f"candidates to exclude as a target or feature."
        )

    # 6. Imbalanced categorical columns
    imbalanced = []
    for c in categorical_cols:
        if c["top_values"]:
            top_share = _pct(c["top_values"][0]["count"], rows)
            if top_share > 80:
                imbalanced.append((c["name"], c["top_values"][0]["value"], top_share))
    if imbalanced:
        name, val, share = imbalanced[0]
        notes.append(
            f"'{name}' is heavily skewed toward one value ('{val}' makes up {share}% of rows) "
            f"— if you use this as a prediction target, accuracy alone can be misleading "
            f"since a model could score well just by always guessing the majority class."
        )

    # 7. Strong correlations
    corr = profile.get("correlation", {})
    if corr.get("matrix"):
        strong_pairs = []
        cols_c = corr["columns"]
        m = corr["matrix"]
        for i in range(len(cols_c)):
            for j in range(i + 1, len(cols_c)):
                v = m[i][j]
                if abs(v) >= 0.7:
                    strong_pairs.append((cols_c[i], cols_c[j], v))
        if strong_pairs:
            strong_pairs.sort(key=lambda x: -abs(x[2]))
            top = strong_pairs[0]
            direction = "move together" if top[2] > 0 else "move in opposite directions"
            notes.append(
                f"'{top[0]}' and '{top[1]}' are strongly correlated ({top[2]:+.2f}) — they "
                f"{direction} closely, which is useful to know since including both as "
                f"features rarely adds much beyond using one."
            )

    return notes


def narrate_training(
    problem_type: str,
    target: str,
    leaderboard: list,
    best_model: str,
    feature_importance: list,
) -> List[str]:
    notes = []

    kind = "classification (predicting a category)" if problem_type == "classification" else \
           "regression (predicting a number)"
    notes.append(
        f"'{target}' was detected as a {kind} problem, so AutoDS trained and compared "
        f"{len([r for r in leaderboard if 'metrics' in r])} different model types against it."
    )

    scored = [r for r in leaderboard if "metrics" in r]
    if not scored:
        notes.append("None of the candidate models trained successfully — check the error messages above.")
        return notes

    best = scored[0]
    if problem_type == "classification":
        acc = best["metrics"]["accuracy"]
        f1 = best["metrics"]["f1_weighted"]
        notes.append(
            f"The best performer was {best_model}, correctly classifying {acc*100:.1f}% of "
            f"held-out rows it hadn't seen during training (F1 score of {f1:.2f}, which "
            f"balances catching positives against avoiding false alarms)."
        )
        if acc < 0.6:
            notes.append(
                "That accuracy is fairly low — it suggests the available columns don't "
                "strongly predict the target, or the target itself may be close to random "
                "with respect to these features. More/different features would likely help "
                "more than trying additional model types."
            )
    else:
        r2 = best["metrics"]["r2"]
        rmse = best["metrics"]["rmse"]
        notes.append(
            f"The best performer was {best_model}, explaining {r2*100:.1f}% of the variation "
            f"in '{target}' (R² = {r2:.2f}). On average, its predictions are off by about "
            f"{rmse:,.2f} units (RMSE)."
        )
        if r2 < 0.3:
            notes.append(
                "That R² is fairly low, meaning the model isn't capturing most of what "
                "drives '{}'. ".format(target) +
                "That points to missing important features rather than a modeling problem."
            )

    if len(scored) > 1:
        spread = scored[0]["primary_score"] - scored[-1]["primary_score"]
        worst = scored[-1]
        notes.append(
            f"For comparison, the weakest model ({worst['model']}) scored "
            f"{worst['primary_score']:.2f} vs. the best's {scored[0]['primary_score']:.2f} — "
            f"a gap of {spread:.2f}, which is {'a meaningful difference' if spread > 0.1 else 'fairly small, so several models are roughly tied'}."
        )

    if feature_importance:
        top = feature_importance[:3]
        names = ", ".join(f"'{f['feature'].split('__', 1)[-1]}'" for f in top)
        notes.append(
            f"The features the model relied on most were {names} — these had the biggest "
            f"influence on its predictions."
        )

    return notes


def narrate_prediction(problem_type: str, target: str, prediction, model_name: str, input_row: dict) -> str:
    given = ", ".join(f"{k}={v}" for k, v in input_row.items() if v is not None)
    if problem_type == "classification":
        return (
            f"Given {given}, {model_name} predicts '{target}' = {prediction}."
        )
    return (
        f"Given {given}, {model_name} predicts '{target}' ≈ {prediction:,.2f}."
        if isinstance(prediction, (int, float)) else
        f"Given {given}, {model_name} predicts '{target}' = {prediction}."
    )


def explain_chart(chart_type: str, x: str, df, y: str | None = None, group: str | None = None) -> str | None:
    """
    Data-driven, one-to-two sentence explanation of what a specific chart
    actually shows — separate from the chart's static "caption" (which just
    describes chart mechanics, e.g. "box = middle 50%"). This looks at the
    real values behind the chart and reports the finding: which category
    dominates, how strong a correlation is, which direction a trend moved,
    whether a distribution is skewed or has outliers, etc.

    Best-effort only: returns None (never raises) if the chart type isn't
    covered or the data can't support a finding — callers should treat a
    missing explanation as "nothing extra to add", not an error.
    """
    import pandas as pd  # local import keeps narrate.py import-light for callers that don't need pandas

    try:
        if x not in df.columns:
            return None
        s = df[x]

        if chart_type in ("histogram", "density", "kde"):
            clean = pd.to_numeric(s, errors="coerce").dropna()
            if len(clean) < 3:
                return None
            mean, median, std = clean.mean(), clean.median(), clean.std()
            skew = clean.skew()
            if pd.isna(skew):
                skew = 0
            if skew > 0.5:
                shape = "right-skewed — most values are on the lower end with a tail of higher outliers"
            elif skew < -0.5:
                shape = "left-skewed — most values are on the higher end with a tail of lower outliers"
            else:
                shape = "roughly symmetric around the average"
            q1, q3 = clean.quantile([0.25, 0.75])
            iqr = q3 - q1
            n_out = int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum())
            parts = [
                f"Average '{x}' is {mean:,.2f} (median {median:,.2f}); the distribution is {shape}."
            ]
            if n_out:
                parts.append(
                    f"{n_out} value(s) ({n_out / len(clean) * 100:.1f}%) sit far enough from the "
                    f"bulk of the data to count as statistical outliers."
                )
            return " ".join(parts)

        if chart_type in ("bar", "pie", "treemap"):
            counts = s.value_counts(dropna=True)
            if counts.empty:
                return None
            top_val, top_count = counts.index[0], int(counts.iloc[0])
            total = int(counts.sum())
            share = top_count / total * 100
            txt = (
                f"'{top_val}' is the most common value in '{x}', accounting for {share:.1f}% "
                f"of {total:,} rows across {len(counts)} distinct value(s)."
            )
            if share > 80:
                txt += (
                    " That's a heavy skew toward one value — if you plan to predict this column, "
                    "a model could score misleadingly well just by always guessing it."
                )
            return txt

        if chart_type in ("scatter", "bubble"):
            if not y or y not in df.columns:
                return None
            sub = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 3:
                return None
            corr = sub[x].corr(sub[y])
            if pd.isna(corr):
                return None
            strength = "strong" if abs(corr) >= 0.7 else "moderate" if abs(corr) >= 0.4 else "weak"
            direction = "positive" if corr > 0 else "negative"
            txt = (
                f"'{x}' and '{y}' have a {strength} {direction} relationship "
                f"(correlation {corr:+.2f})."
            )
            if abs(corr) >= 0.7:
                txt += " They move closely together, so using both as model features rarely adds much beyond using one."
            return txt

        if chart_type in ("line", "area"):
            if not y or y not in df.columns:
                return None
            sub = df[[x, y]].copy()
            sub[y] = pd.to_numeric(sub[y], errors="coerce")
            sub = sub.dropna()
            if len(sub) < 2:
                return None
            try:
                sub = sub.sort_values(x)
            except TypeError:
                pass
            first, last = float(sub[y].iloc[0]), float(sub[y].iloc[-1])
            change = last - first
            direction = "increased" if change > 0 else "decreased" if change < 0 else "stayed flat"
            txt = f"'{y}' {direction} from {first:,.2f} to {last:,.2f} across the range shown"
            if first != 0:
                txt += f" ({change / abs(first) * 100:+.1f}%)"
            peak_pos = sub[y].idxmax()
            peak_val = float(sub[y].max())
            if abs(peak_val - max(first, last)) > 1e-9:
                txt += f", peaking at {peak_val:,.2f} around {x}={sub.loc[peak_pos, x]}."
            else:
                txt += "."
            return txt

        if chart_type in ("boxplot", "violin"):
            clean = pd.to_numeric(s, errors="coerce").dropna()
            if len(clean) < 3:
                return None
            q1, med, q3 = clean.quantile([0.25, 0.5, 0.75])
            iqr = q3 - q1
            n_out = int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum())
            txt = (
                f"Median '{x}' is {med:,.2f}, with the middle 50% of values between "
                f"{q1:,.2f} and {q3:,.2f}."
            )
            if n_out:
                txt += f" {n_out} outlier(s) fall well outside that range."
            return txt

        if chart_type == "choropleth":
            clean = pd.to_numeric(s, errors="coerce").dropna()
            if clean.empty:
                return None
            return (
                f"'{x}' ranges from {clean.min():,.2f} to {clean.max():,.2f} across the mapped "
                f"regions, averaging {clean.mean():,.2f}."
            )

        return None
    except Exception:
        # Explanations are a bonus, never a reason to break a chart request.
        return None