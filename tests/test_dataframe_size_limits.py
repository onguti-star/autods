import pytest
from fastapi import HTTPException

from backend import main


class ShapeOnlyFrame:
    def __init__(self, rows, cols):
        self.shape = (rows, cols)


def test_dataframe_size_allows_dataset_just_over_previous_limit():
    main._validate_dataframe_size(ShapeOnlyFrame(5_000_019, 10))


def test_dataframe_size_rejects_rows_over_current_limit():
    with pytest.raises(HTTPException) as exc:
        main._validate_dataframe_size(ShapeOnlyFrame(main.MAX_DATAFRAME_ROWS + 1, 10))

    assert exc.value.status_code == 400
    assert f"{main.MAX_DATAFRAME_ROWS:,} rows" in exc.value.detail

