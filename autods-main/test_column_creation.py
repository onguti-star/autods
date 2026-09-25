"""Test the new column creation feature with mathematical expressions."""
import pandas as pd
from backend.clean_chat import run_command


def test_create_column_multiply():
    """Test creating a column by multiplying two existing columns."""
    df = pd.DataFrame({
        'price': [10, 20, 30],
        'quantity': [2, 3, 4]
    })
    
    result_df, message = run_command(
        df, 
        "create column total as price * quantity"
    )
    
    print("Test 1: Create column with multiplication")
    print(f"Message: {message}")
    print(f"Result:\n{result_df}")
    assert 'total' in result_df.columns, "Column 'total' should be created"
    assert result_df['total'].tolist() == [20, 60, 120], "Values should be price * quantity"
    print("✓ Test passed\n")


def test_create_column_add():
    """Test creating a column by adding to an existing column."""
    df = pd.DataFrame({
        'price': [10, 20, 30]
    })
    
    result_df, message = run_command(
        df,
        "add new column tax as price * 0.15"
    )
    
    print("Test 2: Create column with percentage")
    print(f"Message: {message}")
    print(f"Result:\n{result_df}")
    assert 'tax' in result_df.columns, "Column 'tax' should be created"
    assert abs(result_df['tax'].iloc[0] - 1.5) < 0.01, "Tax should be 15% of price"
    print("✓ Test passed\n")


def test_create_column_subtract():
    """Test creating a column by subtracting columns."""
    df = pd.DataFrame({
        'revenue': [100, 200, 300],
        'cost': [60, 80, 100]
    })
    
    result_df, message = run_command(
        df,
        "new column profit = revenue - cost"
    )
    
    print("Test 3: Create column with subtraction")
    print(f"Message: {message}")
    print(f"Result:\n{result_df}")
    assert 'profit' in result_df.columns, "Column 'profit' should be created"
    assert result_df['profit'].tolist() == [40, 120, 200], "Values should be revenue - cost"
    print("✓ Test passed\n")


def test_create_column_complex():
    """Test creating a column with a complex expression."""
    df = pd.DataFrame({
        'price': [100, 200, 300],
        'cost': [60, 80, 100]
    })
    
    result_df, message = run_command(
        df,
        "create column margin as (price - cost) / price * 100"
    )
    
    print("Test 4: Create column with complex expression")
    print(f"Message: {message}")
    print(f"Result:\n{result_df}")
    assert 'margin' in result_df.columns, "Column 'margin' should be created"
    expected = [40.0, 60.0, 66.66666666666667]
    for i, exp in enumerate(expected):
        assert abs(result_df['margin'].iloc[i] - exp) < 0.01, f"Margin at index {i} should be ~{exp}"
    print("✓ Test passed\n")


def test_create_column_power():
    """Test creating a column with power operator."""
    df = pd.DataFrame({
        'value': [2, 3, 4]
    })
    
    result_df, message = run_command(
        df,
        "create column squared as value ** 2"
    )
    
    print("Test 5: Create column with power operator")
    print(f"Message: {message}")
    print(f"Result:\n{result_df}")
    assert 'squared' in result_df.columns, "Column 'squared' should be created"
    assert result_df['squared'].tolist() == [4, 9, 16], "Values should be value squared"
    print("✓ Test passed\n")


def test_create_column_with_missing_values():
    """Test creating a column when some values are missing."""
    df = pd.DataFrame({
        'price': [10, None, 30],
        'quantity': [2, 3, 4]
    })
    
    result_df, message = run_command(
        df,
        "create column total as price * quantity"
    )
    
    print("Test 6: Create column with missing values")
    print(f"Message: {message}")
    print(f"Result:\n{result_df}")
    assert 'total' in result_df.columns, "Column 'total' should be created"
    assert result_df['total'].iloc[0] == 20, "First value should be 20"
    assert pd.isna(result_df['total'].iloc[1]), "Second value should be NaN"
    assert result_df['total'].iloc[2] == 120, "Third value should be 120"
    print("✓ Test passed\n")


def test_create_column_where_it_is():
    """Test creating a column using 'where it is' format."""
    df = pd.DataFrame({
        'SLOPE': [2, 3, 4],
        'RAINFALL': [10, 20, 30]
    })
    
    result_df, message = run_command(
        df,
        "create a new column called nun where it is SLOPE * RAINFALL"
    )
    
    print("Test 7: Create column with 'where it is' format")
    print(f"Message: {message}")
    print(f"Result:\n{result_df}")
    assert 'nun' in result_df.columns, "Column 'nun' should be created"
    assert result_df['nun'].tolist() == [20, 60, 120], "Values should be SLOPE * RAINFALL"
    print("✓ Test passed\n")


def test_create_column_sum_all():
    """Test creating a column by summing all numeric columns."""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    
    result_df, message = run_command(
        df,
        "create column total as sum of all columns"
    )
    
    print("Test 8: Create column as sum of all columns")
    print(f"Message: {message}")
    print(f"Result:\n{result_df}")
    assert 'total' in result_df.columns, "Column 'total' should be created"
    assert result_df['total'].tolist() == [12, 15, 18], "Values should be sum of all columns"
    print("✓ Test passed\n")


def test_create_column_mean_all():
    """Test creating a column by taking mean of all numeric columns."""
    df = pd.DataFrame({
        'A': [10, 20, 30],
        'B': [20, 30, 40],
        'C': [30, 40, 50]
    })
    
    result_df, message = run_command(
        df,
        "create column avg as mean of all columns"
    )
    
    print("Test 9: Create column as mean of all columns")
    print(f"Message: {message}")
    print(f"Result:\n{result_df}")
    assert 'avg' in result_df.columns, "Column 'avg' should be created"
    assert result_df['avg'].tolist() == [20.0, 30.0, 40.0], "Values should be mean of all columns"
    print("✓ Test passed\n")


def test_create_column_multiply_all():
    """Test creating a column by multiplying all numeric columns."""
    df = pd.DataFrame({
        'A': [2, 3, 4],
        'B': [3, 4, 5]
    })
    
    result_df, message = run_command(
        df,
        "create column product as multiply all columns"
    )
    
    print("Test 10: Create column as product of all columns")
    print(f"Message: {message}")
    print(f"Result:\n{result_df}")
    assert 'product' in result_df.columns, "Column 'product' should be created"
    assert result_df['product'].tolist() == [6, 12, 20], "Values should be product of all columns"
    print("✓ Test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Column Creation with Mathematical Expressions")
    print("=" * 60 + "\n")
    
    test_create_column_multiply()
    test_create_column_add()
    test_create_column_subtract()
    test_create_column_complex()
    test_create_column_power()
    test_create_column_with_missing_values()
    test_create_column_where_it_is()
    test_create_column_sum_all()
    test_create_column_mean_all()
    test_create_column_multiply_all()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
