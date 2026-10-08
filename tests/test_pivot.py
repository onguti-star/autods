import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.pivot import PivotRequest, build_pivot


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 80
    return pd.DataFrame({
        "region": rng.choice(["North", "South", "East"], n),
        "product": rng.choice(["A", "B"], n),
        "channel": rng.choice(["web", "store"], n),
        "revenue": rng.integers(10, 100, n),
        "qty": rng.integers(1, 9, n),
        "order_date": pd.date_range("2025-01-01", periods=n, freq="9D").strftime("%d-%m-%Y"),
    })


def _cells(result):
    return {tuple(r["labels"]): r["cells"] for r in result["rows"]}


def test_sum_by_one_field_matches_groupby_and_has_total(df):
    res = build_pivot(df, PivotRequest(rows=["region"], values=["revenue"]))
    expected = df.groupby("region")["revenue"].sum()
    cells = _cells(res)
    for region, total in expected.items():
        assert cells[(region,)] == [int(total)]
    assert cells[("Total",)] == [int(df["revenue"].sum())]
    assert res["rows"][-1]["is_total"] is True


def test_rows_by_columns_matches_pandas_crosstab(df):
    res = build_pivot(df, PivotRequest(rows=["region"], columns=["product"], values=["revenue"]))
    expected = df.pivot_table(index="region", columns="product", values="revenue", aggfunc="sum")
    assert res["headers"] == ["A", "B", "Total"]
    cells = _cells(res)
    for region in expected.index:
        assert cells[(region,)][:2] == [int(expected.loc[region, "A"]), int(expected.loc[region, "B"])]
        assert cells[(region,)][2] == int(expected.loc[region].sum())


def test_no_values_counts_rows(df):
    res = build_pivot(df, PivotRequest(rows=["region"]))
    cells = _cells(res)
    assert cells[("Total",)] == [len(df)]
    assert sum(cells[(r,)][0] for r in df["region"].unique()) == len(df)


def test_average_and_value_order_follow_the_request(df):
    res = build_pivot(df, PivotRequest(rows=["region"], values=["revenue", "qty"], aggregate="mean"))
    assert res["headers"] == ["mean of revenue", "mean of qty"]
    east = df[df["region"] == "East"]
    assert _cells(res)[("East",)] == [round(east["revenue"].mean(), 2), round(east["qty"].mean(), 2)]


def test_two_row_fields_nest(df):
    res = build_pivot(df, PivotRequest(rows=["region", "product"], values=["revenue"]))
    expected = df.groupby(["region", "product"])["revenue"].sum()
    cells = _cells(res)
    for (region, product), total in expected.items():
        assert cells[(region, product)] == [int(total)]
    assert res["row_fields"] == ["region", "product"]


@pytest.mark.parametrize("mode", ["of_total", "of_row", "of_col"])
def test_percent_modes_add_up_to_100(df, mode):
    res = build_pivot(df, PivotRequest(rows=["region"], columns=["product"], values=["revenue"], percent=mode))
    cells = _cells(res)
    assert res["is_percent"] is True
    if mode == "of_total":
        assert cells[("Total",)][-1] == 100
        assert sum(cells[(r,)][-1] for r in df["region"].unique()) == pytest.approx(100, abs=0.05)
    elif mode == "of_row":
        for r in df["region"].unique():
            assert cells[(r,)][-1] == 100
            assert sum(cells[(r,)][:2]) == pytest.approx(100, abs=0.05)
    else:
        assert cells[("Total",)][:2] == [100, 100]
        for col in range(2):
            assert sum(cells[(r,)][col] for r in df["region"].unique()) == pytest.approx(100, abs=0.05)


def test_percent_without_totals_still_correct(df):
    res = build_pivot(df, PivotRequest(rows=["region"], columns=["product"], values=["revenue"],
                                       percent="of_row", totals=False))
    assert res["headers"] == ["A", "B"]
    for row in res["rows"]:
        assert sum(row["cells"]) == pytest.approx(100, abs=0.05)


def test_group_dates_by_month(df):
    res = build_pivot(df, PivotRequest(rows=["order_date"], values=["revenue"], date_by="month"))
    labels = [r["labels"][0] for r in res["rows"] if not r["is_total"]]
    assert labels == sorted(labels)
    assert all(len(label) == 7 and label[4] == "-" for label in labels)
    assert any("grouped by month" in n for n in res["notes"])


def test_blank_groups_are_kept_not_dropped():
    data = pd.DataFrame({"g": ["a", None, "a", None], "v": [1, 2, 3, 4]})
    res = build_pivot(data, PivotRequest(rows=["g"], values=["v"]))
    cells = _cells(res)
    assert cells[("(blank)",)] == [6]
    assert cells[("a",)] == [4]


def test_sort_by_value_descending_keeps_total_last(df):
    res = build_pivot(df, PivotRequest(rows=["region"], values=["revenue"], sort_by="value", descending=True))
    body = [r["cells"][0] for r in res["rows"] if not r["is_total"]]
    assert body == sorted(body, reverse=True)
    assert res["rows"][-1]["is_total"] is True


@pytest.mark.parametrize("kwargs, message", [
    (dict(), "at least one field"),
    (dict(rows=["nope"], values=["revenue"]), "not found"),
    (dict(rows=["region"], columns=["region"], values=["revenue"]), "in both Rows and Columns"),
    (dict(rows=["region"], values=["product"]), "isn't numeric"),
    (dict(rows=["region"], values=["region"]), "both a value"),
    (dict(rows=["region"], values=["revenue", "qty"], percent="of_total"), "one value"),
    (dict(rows=["region"], values=["revenue"], percent="of_row"), "need a field in Columns"),
    (dict(rows=["region"], values=["revenue"], aggregate="mean", percent="of_total"), "Sum or Count"),
])
def test_helpful_errors(df, kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_pivot(df, PivotRequest(**kwargs))


def test_count_works_on_text_columns(df):
    res = build_pivot(df, PivotRequest(rows=["region"], values=["product"], aggregate="count"))
    assert _cells(res)[("Total",)] == [len(df)]


def test_pivot_endpoint_end_to_end(df):
    from backend.main import app

    client = TestClient(app)
    resp = client.post("/api/upload", files={"file": ("t.csv", df.to_csv(index=False).encode(), "text/csv")})
    sid = resp.json()["session_id"]

    ok = client.post(f"/api/pivot/{sid}", json={"rows": ["region"], "columns": ["product"], "values": ["revenue"]})
    assert ok.status_code == 200
    assert ok.json()["headers"] == ["A", "B", "Total"]

    bad = client.post(f"/api/pivot/{sid}", json={"rows": ["region"], "values": ["product"]})
    assert bad.status_code == 400
    assert "isn't numeric" in bad.json()["detail"]

    missing = client.post("/api/pivot/not-a-session", json={"rows": ["region"]})
    assert missing.status_code == 404
