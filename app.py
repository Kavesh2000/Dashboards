import streamlit as st
import pandas as pd
import plotly.express as px
import sqlalchemy as sa
from sklearn.linear_model import LinearRegression
import numpy as np

# Connect to the database
engine = sa.create_engine('sqlite:///sample_data.db')

# Title
st.title("Advanced Python Dashboard with SQL, Spark ETL, and Predictive Analytics")

# Query data
query_sales = "SELECT * FROM sales"
df_sales = pd.read_sql(query_sales, engine)

query_agg = "SELECT * FROM aggregated_sales"
df_agg = pd.read_sql(query_agg, engine)

# Display data
st.header("Raw Sales Data")
st.dataframe(df_sales.head(20))

st.header("Aggregated Sales Data (Processed by Spark ETL)")
st.dataframe(df_agg)

# Summary stats
st.header("Summary Statistics")
st.write(df_sales.describe())

# Chart: Sales over time
st.header("Sales Over Time")
fig = px.line(df_sales, x='Date', y='Sales', color='Product', title='Sales by Product Over Time')
st.plotly_chart(fig)

# Chart: Sales by Region
st.header("Sales by Region")
region_sales = df_sales.groupby('Region')['Sales'].sum().reset_index()
fig2 = px.bar(region_sales, x='Region', y='Sales', title='Total Sales by Region')
st.plotly_chart(fig2)

# Predictive Analytics: Simple Linear Regression for Sales Forecasting
st.header("Predictive Analytics: Sales Forecasting")
# Prepare data for ML
df_sales['Date'] = pd.to_datetime(df_sales['Date'])
df_sales['Days'] = (df_sales['Date'] - df_sales['Date'].min()).dt.days

X = df_sales[['Days']]
y = df_sales['Sales']

model = LinearRegression()
model.fit(X, y)

# Predict next 30 days
future_days = np.arange(df_sales['Days'].max() + 1, df_sales['Days'].max() + 31).reshape(-1, 1)
predictions = model.predict(future_days)

future_dates = pd.date_range(df_sales['Date'].max() + pd.Timedelta(days=1), periods=30)
pred_df = pd.DataFrame({'Date': future_dates, 'Predicted_Sales': predictions})

st.write("Model trained on historical data. Forecasting next 30 days:")
st.dataframe(pred_df)

fig3 = px.line(pred_df, x='Date', y='Predicted_Sales', title='Sales Forecast')
st.plotly_chart(fig3)

# Filter by product
st.header("Filter by Product")
product = st.selectbox("Select Product", df_sales['Product'].unique())
filtered_df = df_sales[df_sales['Product'] == product]
st.dataframe(filtered_df)