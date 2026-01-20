# Python Dashboards with SQL Data

This repository contains Python-based dashboards that extract data from SQL databases and visualize it interactively.

## Features

- Connects to SQLite database (easily adaptable to other SQL databases)
- Displays raw data, summary statistics, and interactive charts
- Built with Streamlit for easy web deployment
- Uses Plotly for visualizations

## Setup

1. Clone this repository:
   ```
   git clone <your-repo-url>
   cd DASHBOARDS
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create sample database:
   ```
   python create_sample_db.py
   ```

4. Run the dashboard:
   ```
   streamlit run app.py
   ```

5. Open your browser to the URL shown (usually http://localhost:8501)

## Customization

- To connect to a different SQL database, modify the `engine` in `app.py` (e.g., for PostgreSQL: `postgresql://user:password@localhost/dbname`)
- Add more queries or visualizations as needed
- Deploy to Streamlit Cloud or other platforms for sharing

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Plotly
- SQLAlchemy

## Troubleshooting

- If you encounter database connection issues, ensure your SQL server is running and credentials are correct
- For large datasets, consider adding pagination or sampling in the queries