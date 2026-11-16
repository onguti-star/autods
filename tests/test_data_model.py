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


def test_data_model_rejects_value_overlap_without_key_family_match():
    products = store.create_session(
        pd.DataFrame({"product_id": [1, 2, 3, 4]}),
        "products.csv",
    )
    refunds = store.create_session(
        pd.DataFrame({"order_item_refund_id": [1, 2, 3, 4]}),
        "order_item_refunds.csv",
    )

    try:
        result = main.data_model(main.DataModelRequest(session_ids=[products.id, refunds.id]))
        assert result["relationships"] == []
    finally:
        store.delete_session(products.id)
        store.delete_session(refunds.id)


def test_data_model_keeps_maven_fuzzy_factory_relationships_distinct():
    sessions = store.create_session(
        pd.DataFrame({"website_session_id": [1, 2], "user_id": [10, 11]}),
        "website_sessions.csv",
    )
    orders = store.create_session(
        pd.DataFrame({"order_id": [20, 21], "website_session_id": [1, 2], "primary_product_id": [1, 1]}),
        "orders_mav.csv",
    )
    items = store.create_session(
        pd.DataFrame({"order_item_id": [30, 31], "order_id": [20, 21], "product_id": [1, 1]}),
        "order_items.csv",
    )
    products = store.create_session(
        pd.DataFrame({"product_id": [1, 2], "product_name": ["A", "B"]}),
        "products.csv",
    )
    refunds = store.create_session(
        pd.DataFrame({"order_item_refund_id": [40], "order_item_id": [30]}),
        "order_item_refunds.csv",
    )

    try:
        result = main.data_model(main.DataModelRequest(
            session_ids=[sessions.id, orders.id, items.id, products.id, refunds.id]
        ))
        pairs = {
            (r["from_table"], r["from_column"], r["to_table"], r["to_column"])
            for r in result["relationships"]
        }
        assert ("orders_mav", "website_session_id", "website_sessions", "website_session_id") in pairs
        assert ("orders_mav", "primary_product_id", "products", "product_id") in pairs
        assert ("order_items", "product_id", "products", "product_id") in pairs
        assert ("order_items", "order_id", "orders_mav", "order_id") in pairs
        assert ("order_item_refunds", "order_item_id", "order_items", "order_item_id") in pairs
        assert not any(
            r["from_table"] == "order_items" and r["to_table"] == "website_sessions"
            for r in result["relationships"]
        )
    finally:
        for session in (sessions, orders, items, products, refunds):
            store.delete_session(session.id)
