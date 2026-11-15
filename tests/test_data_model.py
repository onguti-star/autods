import pandas as pd

from backend import main, store


def test_data_model_finds_conventional_foreign_key():
    products = store.create_session(
        pd.DataFrame({"id": [1, 2], "name": ["A", "B"]}),
        "products.csv",
    )
    orders = store.create_session(
        pd.DataFrame({"order_id": [10, 11], "product_id": [1, 2]}),
        "orders.csv",
    )

    try:
        result = main.data_model(main.DataModelRequest(session_ids=[products.id, orders.id]))
        assert {table["name"] for table in result["tables"]} == {"products", "orders"}
        assert any(
            relationship["from_table"] == "orders"
            and relationship["from_column"] == "product_id"
            and relationship["to_table"] == "products"
            and relationship["to_column"] == "id"
            for relationship in result["relationships"]
        )
    finally:
        store.delete_session(products.id)
        store.delete_session(orders.id)


def test_restored_session_keeps_filename_for_data_model():
    session = store.create_session(pd.DataFrame({"id": [1]}), "products.csv")
    try:
        store.SESSIONS.pop(session.id)
        restored = store.get_session(session.id)
        assert restored.filename == "products.csv"
    finally:
        store.delete_session(session.id)
