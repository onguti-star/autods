"""
Test the new correlation table feature.
This simulates asking "how does UNITS_SOLD correlate to other columns"
"""
import pandas as pd
import numpy as np
from backend.assistant import answer_question_table, _corr_strength

# Create sample dataset
np.random.seed(42)
n_rows = 1000

df = pd.DataFrame({
    'UNITS_SOLD': np.random.randint(1, 100, n_rows),
    'price': np.random.uniform(10, 200, n_rows),
    'revenue': np.random.uniform(100, 5000, n_rows),
    'customer_rating': np.random.uniform(1, 5, n_rows),
    'cost': np.random.uniform(5, 150, n_rows),
    'profit': np.random.uniform(10, 1000, n_rows),
})

# Make some realistic correlations
df['revenue'] = df['UNITS_SOLD'] * df['price'] + np.random.normal(0, 100, n_rows)
df['profit'] = df['revenue'] - df['cost'] * df['UNITS_SOLD']

print("Testing correlation table feature...")
print("=" * 80)

# Test the question
question = "how does UNITS_SOLD correlate to other columns"
result = answer_question_table(df, question)

if result and 'columns' in result and 'rows' in result:
    print(f"\nQuestion: {question}")
    print(f"\nCorrelation Table:")
    print(f"Columns: {result['columns']}")
    print(f"\nRows:")
    for row in result['rows']:
        print(row)
else:
    print(f"\nNo table result returned. Result: {result}")
    print("\nTrying alternative questions...")
    
    # Try other phrasings
    for q in ["correlation of UNITS_SOLD", "UNITS_SOLD correlations", "show correlations for UNITS_SOLD"]:
        result = answer_question_table(df, q)
        print(f"\nQ: {q}")
        print(f"Result: {result}")