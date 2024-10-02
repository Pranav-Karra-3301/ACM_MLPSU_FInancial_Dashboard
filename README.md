# Financial Dashboard Setup Guide

This guide will walk you through setting up a financial dashboard using Python and Streamlit to visualize financial data and predict future trends. It covers installation, environment setup, running the code, and explaining the main components of the code.

## Prerequisites

Before starting, make sure you have the following installed:

1. **Python 3.10+** 
2. **pip** (Python's package installer)

### Installing Python:

- **Mac:**  
  Use `brew` to install Python:  
  ```bash
  brew install python
  ```

- **Windows:**  
  Download the Python installer from python.org and run it. Make sure to check the box for "Add Python to PATH."

## Step 1: Setting up the Python Environment

1. Create a virtual environment:
   ```bash
   # Mac/Linux
   python3 -m venv venv

   # Windows
   python -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   # Mac/Linux
   source venv/bin/activate

   # Windows
   .\venv\Scripts\activate
   ```

3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   If you don't have a requirements.txt, you can install packages individually:
   ```bash
   pip install streamlit pandas scikit-learn altair
   ```

## Step 2: Running the Application

Once everything is installed, run the financial dashboard locally:

```bash
streamlit run main.py
```

This will start a local web server, and you can view the dashboard by opening the provided URL (typically http://localhost:8501) in your browser.

## Step 3: Understanding the Code

### main.py

This file contains the Streamlit web application logic.

1. **Imports:**
   ```python
   import streamlit as st
   import pandas as pd
   import altair as alt
   from prediction_model import predict_future_transactions
   ```

2. **Loading Data:**
   ```python
   @st.cache_data
   def load_data():
       df = pd.read_csv('data.csv')
       df['date'] = pd.to_datetime(df['date'])
       df['Transaction Amount'] = df['Transaction Amount'].astype(float)
       return df
   ```

3. **Main Dashboard:**
   ```python
   st.title("Financial Dashboard")

   categories = ['All'] + sorted(df['Category'].unique().tolist())
   selected_category = st.sidebar.selectbox("Select Category", categories)
   selected_month = st.sidebar.selectbox("Select Month", month_options)
   ```

4. **Metrics Display:**
   ```python
   col1, col2, col3 = st.columns(3)
   col1.metric("Total Income", f"${total_income:.2f}")
   col2.metric("Total Expenses", f"${total_expenses:.2f}")
   col3.metric("Net Transactions", f"${net_transactions:.2f}")
   ```

5. **Prediction Model:**
   ```python
   future_income, future_expenditure = predict_future_transactions(df)
   predicted_net_balance = future_income['prediction'].sum() - future_expenditure['prediction'].sum()
   ```

### prediction_model.py

This file contains the logic for predicting future transactions using a regression model.

1. **Data Preparation:**
   ```python
   def prepare_data(df):
       df['date'] = pd.to_datetime(df['date'])
       df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
       return df
   ```

2. **Training the Model:**
   ```python
   def train_model(df, target):
       X = df[['days_since_start']]
       y = df[target]
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       model = LinearRegression()
       model.fit(X_train, y_train)
       return model
   ```

3. **Making Predictions:**
   ```python
   def make_predictions(model, df, days_to_predict=30):
       last_date = df['date'].max()
       future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_to_predict)
       future_df = pd.DataFrame({'date': future_dates})
       future_df['days_since_start'] = (future_df['date'] - df['date'].min()).dt.days
       predictions = model.predict(future_df[['days_since_start']])
       future_df['prediction'] = predictions
       return future_df
   ```

## Step 4: Why Linear Regression May Not Be Ideal

Linear Regression is a basic model that assumes a linear relationship between the number of days and the transaction amounts. However, financial transactions tend to be more complex and influenced by many external factors such as seasonality, irregular spending patterns, and income changes.

### Limitations of Linear Regression:
- Assumption of Linearity: Financial data rarely follows a strict linear trend. People's income and expenses fluctuate.
- Inability to Capture Complex Trends: Linear models cannot capture seasonal patterns or abrupt changes in spending behavior.

### Alternatives to Linear Regression:
For better predictions, consider using more advanced models like:
- Time Series Models (e.g., ARIMA or Prophet)
- Tree-based Models (e.g., Random Forest, Gradient Boosting)
- Neural Networks for more complex and non-linear relationships.

## Conclusion

This guide has walked through the steps to set up and run the financial dashboard. We have also discussed how the prediction model works and why linear regression may not be the best choice for financial data. To improve predictions, consider exploring more advanced modeling techniques.