import sqlite3
import pandas as pd

# Create a sample database
conn = sqlite3.connect('sample_data.db')

# Sample data: Sales data
data = {
    'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
    'Product': ['A', 'B', 'C'] * 33 + ['A'],
    'Sales': [100, 150, 200] * 33 + [100],
    'Region': ['North', 'South', 'East', 'West'] * 25
}

df = pd.DataFrame(data)
df.to_sql('sales', conn, if_exists='replace', index=False)

conn.close()
print("Sample database created with table 'sales'.")