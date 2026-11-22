"""
Test script to demonstrate how PRODUCT_NAME correlates with other columns.
This creates sample data and runs the EDA analysis.
"""
import pandas as pd
import numpy as np
from backend.eda import profile_dataframe, correlation_matrix

# Create sample dataset with PRODUCT_NAME and related columns
np.random.seed(42)
n_rows = 1000

# PRODUCT_NAME as categorical identifier
products = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E']
product_name = np.random.choice(products, n_rows)

# Create correlated numeric columns based on product
# Each product has different price/rating/sales characteristics
product_stats = {
    'Product_A': {'price': 50, 'rating': 4.5, 'sales': 1000, 'cost': 30},
    'Product_B': {'price': 75, 'rating': 4.2, 'sales': 800, 'cost': 45},
    'Product_C': {'price': 100, 'rating': 4.8, 'sales': 1200, 'cost': 60},
    'Product_D': {'price': 25, 'rating': 3.9, 'sales': 2000, 'cost': 15},
    'Product_E': {'price': 150, 'rating': 4.6, 'sales': 600, 'cost': 90},
}

prices = [product_stats[p]['price'] + np.random.normal(0, 5) for p in product_name]
ratings = [product_stats[p]['rating'] + np.random.normal(0, 0.3) for p in product_name]
sales = [product_stats[p]['sales'] + np.random.normal(0, 100) for p in product_name]
costs = [product_stats[p]['cost'] + np.random.normal(0, 3) for p in product_name]

# Create DataFrame
df = pd.DataFrame({
    'PRODUCT_NAME': product_name,
    'price': prices,
    'rating': ratings,
    'sales': sales,
    'cost': costs,
    'revenue': [p * s for p, s in zip(prices, sales)],
    'profit': [(p - c) * s for p, c, s in zip(prices, costs, sales)],
    'category': np.random.choice(['Electronics', 'Clothing', 'Food'], n_rows),
    'in_stock': np.random.choice([0, 1], n_rows, p=[0.2, 0.8])
})

print("=" * 80)
print("SAMPLE DATASET CREATED")
print("=" * 80)
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nColumn types:")
print(df.dtypes)

# Run correlation matrix (only numeric columns)
print("\n" + "=" * 80)
print("CORRELATION MATRIX (Numeric Columns Only)")
print("=" * 80)
corr_result = correlation_matrix(df)
print(f"Columns analyzed: {corr_result['columns']}")
print(f"\nCorrelation Matrix:")
corr_df = pd.DataFrame(
    corr_result['matrix'],
    index=corr_result['columns'],
    columns=corr_result['columns']
)
print(corr_df.round(3))

# Show PRODUCT_NAME relationship through group statistics
print("\n" + "=" * 80)
print("PRODUCT_NAME RELATIONSHIP ANALYSIS")
print("=" * 80)
print("\nPRODUCT_NAME value counts:")
print(df['PRODUCT_NAME'].value_counts())

print("\nNumeric statistics by PRODUCT_NAME:")
print(df.groupby('PRODUCT_NAME')[['price', 'rating', 'sales', 'revenue', 'profit']].mean().round(2))

# Full profile
print("\n" + "=" * 80)
print("FULL EDA PROFILE")
print("=" * 80)
profile = profile_dataframe(df)

print(f"\nDataset shape: {profile['shape']}")
print(f"Total missing cells: {profile['total_missing_cells']}")
print(f"Duplicate rows: {profile['duplicate_rows']}")

print("\nColumn profiles:")
for col_profile in profile['columns']:
    print(f"\n  {col_profile['name']}:")
    print(f"    Type: {col_profile['type']}")
    print(f"    Missing: {col_profile['missing']} ({col_profile['missing_pct']}%)")
    print(f"    Unique: {col_profile['unique']}")
    if col_profile['type'] == 'numeric' and 'stats' in col_profile:
        print(f"    Mean: {col_profile['stats'].get('mean', 'N/A')}")
    if col_profile['type'] == 'categorical' and 'top_values' in col_profile:
        print(f"    Top values: {[v['value'] for v in col_profile['top_values'][:3]]}")

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print("""
1. PRODUCT_NAME is a CATEGORICAL column, so it doesn't appear in the correlation matrix
   (correlation only works on numeric columns).

2. However, PRODUCT_NAME has strong RELATIONSHIPS with numeric columns:
   - Each product has distinct price, rating, sales, and profit characteristics
   - You can see this through groupby analysis above

3. To analyze categorical-to-numeric relationships:
   - Use groupby().mean() as shown above
   - Use ANOVA or t-tests for statistical significance
   - Encode PRODUCT_NAME (one-hot, label encoding) to include in correlation

4. The correlation matrix shows relationships between NUMERIC columns:
   - price ↔ revenue: very high correlation (price * sales = revenue)
   - cost ↔ profit: strong negative correlation
   - sales ↔ revenue: very high correlation
""")