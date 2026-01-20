import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlalchemy as sa
from sklearn.linear_model import LinearRegression
import numpy as np

# Set page config for modern look
st.set_page_config(
    page_title="Banking Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for banking theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #e6f3ff;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Connect to the database
def load_data():
    import sqlite3
    conn = sqlite3.connect('sample_data.db')
    df_sales = pd.read_sql("SELECT * FROM sales", conn)
    df_agg = pd.read_sql("SELECT * FROM aggregated_sales", conn)
    conn.close()
    return df_sales, df_agg

df_sales, df_agg = load_data()

# Convert Date
df_sales['Date'] = pd.to_datetime(df_sales['Date'])

# Sidebar filters
st.sidebar.title("🏦 Banking Dashboard Filters")
st.sidebar.markdown("---")

# Date range filter
min_date = df_sales['Date'].min().date()
max_date = df_sales['Date'].max().date()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df_sales[(df_sales['Date'].dt.date >= start_date) & (df_sales['Date'].dt.date <= end_date)]
else:
    df_filtered = df_sales

# Product filter
transaction_types = st.sidebar.multiselect("Select Transaction Types", df_sales['Product'].unique(), default=df_sales['Product'].unique())
if transaction_types:
    df_filtered = df_filtered[df_filtered['Product'].isin(transaction_types)]

# Region filter
branches = st.sidebar.multiselect("Select Branches", df_sales['Region'].unique(), default=df_sales['Region'].unique())
if branches:
    df_filtered = df_filtered[df_filtered['Region'].isin(branches)]

# Main title
st.markdown('<h1 class="main-header">🏦 Advanced Banking Transaction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Leveraging SQL, ETL Processing, and Predictive Analytics for Large-Scale Transaction Insights**")

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_transactions = len(df_filtered)
    st.metric("Total Transactions", f"{total_transactions:,}")
with col2:
    total_volume = df_filtered['Amount'].sum()
    st.metric("Total Transaction Volume", f"${total_volume:,.0f}")
with col3:
    avg_transaction = df_filtered['Amount'].mean()
    st.metric("Average Transaction", f"${avg_transaction:.2f}")
with col4:
    growth_rate = ((df_filtered.groupby(df_filtered['Date'].dt.to_period('M'))['Amount'].sum().pct_change().mean()) * 100)
    st.metric("Monthly Growth Rate", f"{growth_rate:.2f}%")

st.markdown("---")

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends & Analytics", "🔮 Predictive Insights", "📋 Detailed Data"])

with tab1:
    st.header("Transaction Overview")
    
    # Pie chart for product distribution
    col1, col2 = st.columns(2)
    with col1:
        product_dist = df_filtered.groupby('Product')['Amount'].sum().reset_index()
        fig_pie = px.pie(product_dist, values='Amount', names='Product', title='Transaction Volume by Type')
        st.plotly_chart(fig_pie, width='stretch')
    
    with col2:
        region_dist = df_filtered.groupby('Region')['Amount'].sum().reset_index()
        fig_bar_region = px.bar(region_dist, x='Region', y='Amount', title='Transaction Volume by Branch', color='Region')
        st.plotly_chart(fig_bar_region, width='stretch')
    
    # Additional chart: Transaction amount distribution
    st.subheader("Transaction Amount Distribution")
    fig_hist = px.histogram(df_filtered, x='Amount', nbins=50, title='Distribution of Transaction Amounts')
    st.plotly_chart(fig_hist, width='stretch')

with tab2:
    st.header("Trends & Analytics")
    
    # Transaction over time
    fig_line = px.line(df_filtered, x='Date', y='Amount', color='Product', title='Transaction Trends Over Time')
    fig_line.update_layout(xaxis_title="Date", yaxis_title="Transaction Amount ($)")
    st.plotly_chart(fig_line, width='stretch')
    
    # Monthly aggregation
    df_monthly = df_filtered.groupby(df_filtered['Date'].dt.to_period('M'))['Amount'].sum().reset_index()
    df_monthly['Date'] = df_monthly['Date'].astype(str)
    fig_monthly = px.bar(df_monthly, x='Date', y='Amount', title='Monthly Transaction Volume')
    st.plotly_chart(fig_monthly, width='stretch')

with tab3:
    st.header("Predictive Analytics: Transaction Forecasting")
    
    # Prepare data for ML
    df_ml = df_filtered.copy()
    df_ml['Days'] = (df_ml['Date'] - df_ml['Date'].min()).dt.days
    
    X = df_ml['Days'].values.reshape(-1, 1)
    y = df_ml['Amount']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next 30 days
    future_days = np.arange(df_ml['Days'].max() + 1, df_ml['Days'].max() + 31).reshape(-1, 1)
    predictions = model.predict(future_days)
    
    future_dates = pd.date_range(df_ml['Date'].max() + pd.Timedelta(days=1), periods=30)
    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted_Amount': predictions})
    
    st.write("**Forecasting Model**: Linear Regression trained on historical transaction data.")
    st.dataframe(pred_df.head(10))
    
    # Combined historical and predicted
    combined_df = pd.concat([
        df_ml[['Date', 'Amount']].rename(columns={'Amount': 'Value'}),
        pred_df.rename(columns={'Predicted_Amount': 'Value'})
    ])
    combined_df['Type'] = ['Historical'] * len(df_ml) + ['Predicted'] * len(pred_df)
    
    fig_forecast = px.line(combined_df, x='Date', y='Value', color='Type', title='Historical vs Predicted Transactions')
    st.plotly_chart(fig_forecast, width='stretch')

with tab4:
    st.header("Detailed Transaction Data")
    
    # Aggregated data
    st.subheader("Aggregated Data (Processed via ETL)")
    st.dataframe(df_agg)
    
    # Raw data with pagination
    st.subheader("Raw Transaction Data")
    st.dataframe(df_filtered)
    
    # Download button
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_transactions.csv',
        mime='text/csv'
    )

# Footer
st.markdown("---")
st.markdown("*Dashboard powered by Streamlit, Pandas, Plotly, SQLAlchemy, and Scikit-learn. ETL processed with Pandas (simulating Spark).*")