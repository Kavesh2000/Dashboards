import sqlite3
import pandas as pd
import numpy as np

# Create a sample database
conn = sqlite3.connect('sample_data.db')

# Sample data: Banking transactions
np.random.seed(42)  # For reproducible data
n_transactions = 500

data = {
    'Date': pd.date_range('2023-01-01', periods=n_transactions, freq='D'),
    'Product': np.random.choice(['Deposits', 'Loan Disbursements', 'Loan Repayments', 'Savings', 'Withdrawals'], n_transactions),
    'Amount': np.random.uniform(50, 5000, n_transactions).round(2),
    'Region': np.random.choice(['North Branch', 'South Branch', 'East Branch', 'West Branch'], n_transactions),
    'Customer_ID': np.random.randint(1000, 9999, n_transactions)
}

df = pd.DataFrame(data)
df.to_sql('sales', conn, if_exists='replace', index=False)

conn.close()
print("Sample banking database created with table 'sales'.")
print(f"Generated {n_transactions} transactions.")