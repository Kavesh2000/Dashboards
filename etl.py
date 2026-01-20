import pandas as pd
import sqlite3

# Load data from SQL
conn = sqlite3.connect('sample_data.db')
df = pd.read_sql("SELECT * FROM sales", conn)

# ETL transformations
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Aggregate data
aggregated_df = df.groupby(['Product', 'Region'])['Amount'].sum().reset_index()
aggregated_df.rename(columns={'Amount': 'Total_Amount'}, inplace=True)

# Load back to SQL
aggregated_df.to_sql('aggregated_sales', conn, if_exists='replace', index=False)
conn.close()

print("ETL completed: Data processed with Pandas and loaded to SQL.")