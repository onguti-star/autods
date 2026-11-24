"""Hypothesis tests exposed by the AutoDS statistical-analysis panel."""

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats


class HypothesisTestRequest(BaseModel):
    column_a: str = Field(min_length=1, max_length=200)
    column_b: str = Field(min_length=1, max_length=200)
    test: Literal["auto", "correlation", "two_group", "anova", "chi_square"] = "auto"
    alpha: float = Field(default=0.05, gt=0, lt=1)
    correlation_method: Literal["pearson", "spearman"] = "pearson"


def _clean_numeric_pair(df: pd.DataFrame, a: str, b: str) -> tuple[pd.Series, pd.Series]:
    data = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 3:
        raise ValueError("The selected columns need at least 3 complete numeric rows.")
    return data[a].astype(float), data[b].astype(float)


def _base(result: dict, *, test_name: str, null: str, alternative: str,
          statistic_name: str, statistic: float, p_value: float, alpha: float,
          n: int, method_note: str = "") -> dict:
    p_value = float(p_value)
    return {
        "test": test_name,
        "null_hypothesis": null,
        "alternative_hypothesis": alternative,
        "statistic_name": statistic_name,
        "statistic": float(statistic),
        "p_value": p_value,
        "alpha": alpha,
        "significant": bool(p_value < alpha),
        "decision": "Reject the null hypothesis" if p_value < alpha else "Fail to reject the null hypothesis",
        "n": int(n),
        "method_note": method_note,
        **result,
    }


def _two_group(df: pd.DataFrame, value_col: str, group_col: str, alpha: float) -> dict:
    data = df[[value_col, group_col]].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna()
    groups = list(data[group_col].value_counts().index)
    if len(groups) != 2:
        raise ValueError("The group column must contain exactly two non-missing groups for a two-group test.")
    x = data.loc[data[group_col] == groups[0], value_col].astype(float)
    y = data.loc[data[group_col] == groups[1], value_col].astype(float)
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Each group needs at least 2 numeric observations.")
    if x.nunique() < 2 or y.nunique() < 2:
        raise ValueError("Both groups need variation in the numeric outcome.")
    t_stat, p_value = stats.ttest_ind(x, y, equal_var=False)
    difference = float(x.mean() - y.mean())
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    df_welch = (x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)) ** 2 / (
        (x.var(ddof=1) / len(x)) ** 2 / (len(x) - 1) +
        (y.var(ddof=1) / len(y)) ** 2 / (len(y) - 1)
    )
    margin = stats.t.ppf(1 - alpha / 2, df_welch) * se
    pooled_sd = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    cohen_d = difference / pooled_sd if pooled_sd else 0.0
    return _base({
        "column_a": value_col, "column_b": group_col,
        "groups": [str(groups[0]), str(groups[1])],
        "group_means": {str(groups[0]): float(x.mean()), str(groups[1]): float(y.mean())},
        "difference": difference,
        "confidence_interval": [float(difference - margin), float(difference + margin)],
        "effect_size_name": "Cohen's d", "effect_size": float(cohen_d),
    }, test_name="Welch's two-sample t-test", null=f"The mean of '{value_col}' is equal across the two groups.",
       alternative=f"The mean of '{value_col}' differs between the two groups.", statistic_name="t", statistic=t_stat,
       p_value=p_value, alpha=alpha, n=len(data),
       method_note="Welch's test does not assume equal group variances; the confidence interval is for group 1 minus group 2.")


def _anova(df: pd.DataFrame, value_col: str, group_col: str, alpha: float) -> dict:
    data = df[[value_col, group_col]].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna()
    grouped = [g[value_col].astype(float).to_numpy() for _, g in data.groupby(group_col, sort=False) if len(g) >= 2]
    labels = [str(k) for k, g in data.groupby(group_col, sort=False) if len(g) >= 2]
    if len(grouped) < 3:
        raise ValueError("ANOVA needs at least 3 groups with 2 or more observations each.")
    f_stat, p_value = stats.f_oneway(*grouped)
    grand = data[value_col].mean()
    between = sum(len(g) * (g.mean() - grand) ** 2 for g in grouped)
    total = sum(((g - grand) ** 2).sum() for g in [data[value_col].astype(float)])
    eta = between / total if total else 0.0
    return _base({"column_a": value_col, "column_b": group_col, "groups": labels,
                  "group_means": {label: float(g.mean()) for label, g in zip(labels, grouped)},
                  "effect_size_name": "Eta squared", "effect_size": float(eta),
                  "confidence_interval": None},
                 test_name="One-way ANOVA", null=f"All groups have the same mean '{value_col}'.",
                 alternative=f"At least one group mean for '{value_col}' differs.", statistic_name="F", statistic=f_stat,
                 p_value=p_value, alpha=alpha, n=len(data),
                 method_note="ANOVA identifies whether any group differs; it does not identify which pairs differ.")


def _correlation(df: pd.DataFrame, a: str, b: str, alpha: float, method: str) -> dict:
    x, y = _clean_numeric_pair(df, a, b)
    if x.nunique() < 2 or y.nunique() < 2:
        raise ValueError("Both numeric columns need variation for a correlation test.")
    if method == "spearman":
        coefficient, p_value = stats.spearmanr(x, y)
        name = "Spearman rank-correlation test"
    else:
        coefficient, p_value = stats.pearsonr(x, y)
        name = "Pearson correlation test"
    z = np.arctanh(np.clip(coefficient, -0.999999, 0.999999))
    margin = stats.norm.ppf(1 - alpha / 2) / np.sqrt(len(x) - 3) if len(x) > 3 else None
    ci = [float(np.tanh(z - margin)), float(np.tanh(z + margin))] if margin is not None else None
    return _base({"column_a": a, "column_b": b, "effect_size_name": "Correlation coefficient",
                  "effect_size": float(coefficient), "confidence_interval": ci},
                 test_name=name, null=f"There is no {method} association between '{a}' and '{b}'.",
                 alternative=f"There is a {method} association between '{a}' and '{b}'.",
                 statistic_name="r" if method == "pearson" else "rho", statistic=coefficient,
                 p_value=p_value, alpha=alpha, n=len(x),
                 method_note="The confidence interval uses Fisher's z approximation and describes association, not causation.")


def _chi_square(df: pd.DataFrame, a: str, b: str, alpha: float) -> dict:
    data = df[[a, b]].dropna()
    table = pd.crosstab(data[a], data[b])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Both categorical columns need at least two observed categories.")
    chi, p_value, dof, expected = stats.chi2_contingency(table)
    n = int(table.to_numpy().sum())
    v = np.sqrt(chi / (n * min(table.shape[0] - 1, table.shape[1] - 1))) if n else 0.0
    return _base({"column_a": a, "column_b": b, "effect_size_name": "Cramer's V", "effect_size": float(v),
                  "confidence_interval": None, "degrees_of_freedom": int(dof),
                  "contingency_table": {str(i): {str(j): int(table.loc[i, j]) for j in table.columns} for i in table.index}},
                 test_name="Chi-square test of independence", null=f"'{a}' and '{b}' are independent.",
                 alternative=f"'{a}' and '{b}' are associated.", statistic_name="chi-square", statistic=chi,
                 p_value=p_value, alpha=alpha, n=n,
                 method_note="Cramer's V measures association strength; it does not imply causation.")


def run_hypothesis_test(df: pd.DataFrame, req: HypothesisTestRequest) -> dict:
    if req.column_a not in df.columns or req.column_b not in df.columns:
        raise ValueError("Both selected columns must exist in the current dataset.")
    if req.column_a == req.column_b:
        raise ValueError("Choose two different columns.")
    a_num = pd.api.types.is_numeric_dtype(df[req.column_a])
    b_num = pd.api.types.is_numeric_dtype(df[req.column_b])
    if req.test == "correlation" or (req.test == "auto" and a_num and b_num):
        return _correlation(df, req.column_a, req.column_b, req.alpha, req.correlation_method)
    if req.test == "chi_square" or (req.test == "auto" and not a_num and not b_num):
        return _chi_square(df, req.column_a, req.column_b, req.alpha)
    value_col, group_col = (req.column_a, req.column_b) if a_num else (req.column_b, req.column_a)
    group_count = df[group_col].dropna().nunique()
    if req.test == "anova" or (req.test == "auto" and group_count > 2):
        return _anova(df, value_col, group_col, req.alpha)
    return _two_group(df, value_col, group_col, req.alpha)
