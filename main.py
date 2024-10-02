import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
from prediction_model import predict_future_transactions

# Set page config
st.set_page_config(page_title="Pranav Karra's Financial Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['Transaction Amount'] = df['Transaction Amount'].astype(float)
    return df

df = load_data()

# Title
st.title("Pranav Karra's Financial Dashboard")

# Sidebar filters
st.sidebar.header("Filters")
categories = ['All'] + sorted(df['Category'].unique().tolist())
selected_category = st.sidebar.selectbox("Select Category", categories)

# Get unique months from the dataset
months = pd.to_datetime(df['date']).dt.to_period('M').unique()
months = sorted(months)

# Create a list of month options for the selectbox
month_options = [month.strftime('%B %Y') for month in months]

# Add 'All' option at the beginning
month_options.insert(0, 'All')

# Create the month selection dropdown
selected_month = st.sidebar.selectbox("Select Month", month_options)

# Apply filters
filtered_df = df
if selected_category != 'All':
    filtered_df = filtered_df[filtered_df['Category'] == selected_category]

if selected_month != 'All':
    selected_period = pd.Period(selected_month)
    filtered_df = filtered_df[pd.to_datetime(filtered_df['date']).dt.to_period('M') == selected_period]

# Key Metrics
col1, col2, col3, col4 = st.columns(4)
total_transactions = filtered_df['Transaction Amount'].sum()
average_transaction = filtered_df['Transaction Amount'].mean()
total_income = filtered_df[filtered_df['Transaction Amount'] > 0]['Transaction Amount'].sum()
total_expenses = abs(filtered_df[filtered_df['Transaction Amount'] < 0]['Transaction Amount'].sum())

net_transactions = total_income - total_expenses
col1.metric("Net Transaction Amount", f"${net_transactions:,.2f}")
col2.metric("Average Transaction Amount", f"${average_transaction:,.2f}")
col3.metric("Total Income", f"${total_income:,.2f}")
col4.metric("Total Expenses", f"${total_expenses:,.2f}")

# Charts
st.header("Financial Analysis")

# Transaction Amount Over Time (Line Chart)
daily_data = filtered_df.groupby('date')['Transaction Amount'].sum().reset_index()
chart_transactions = alt.Chart(daily_data).mark_line().encode(
    x='date:T',
    y='Transaction Amount:Q',
    tooltip=['date:T', 'Transaction Amount:Q']
).properties(
    title='Daily Transaction Amount Over Time'
)
st.altair_chart(chart_transactions, use_container_width=True)

# Income and Expenditure Pie Charts
if selected_category == 'All':
    # Separate income and expenditure
    income_df = filtered_df[filtered_df['Transaction Amount'] > 0]
    expenditure_df = filtered_df[filtered_df['Transaction Amount'] < 0]

    # Income Pie Chart
    income_data = income_df.groupby('Category')['Transaction Amount'].sum().reset_index()
    income_data = income_data.sort_values('Transaction Amount', ascending=False)
    chart_income = alt.Chart(income_data).mark_arc().encode(
        theta='Transaction Amount:Q',
        color='Category:N',
        tooltip=['Category:N', 'Transaction Amount:Q']
    ).properties(
        title='Income Distribution by Category'
    )

    # Expenditure Pie Chart
    expenditure_data = expenditure_df.groupby('Category')['Transaction Amount'].sum().abs().reset_index()
    expenditure_data = expenditure_data.sort_values('Transaction Amount', ascending=False)
    chart_expenditure = alt.Chart(expenditure_data).mark_arc().encode(
        theta='Transaction Amount:Q',
        color='Category:N',
        tooltip=['Category:N', 'Transaction Amount:Q']
    ).properties(
        title='Expenditure Distribution by Category'
    )

    # Display charts side by side
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart_income, use_container_width=True)
    with col2:
        st.altair_chart(chart_expenditure, use_container_width=True)

# Data Table
st.header("Detailed Data")
st.dataframe(filtered_df)

# Improved Conclusions and Insights
st.header("Conclusions and Insights")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Overall Statistics")
    st.info(f"📊 Total Transaction Amount: ${total_transactions:,.2f}")
    st.info(f"💰 Average Transaction Amount: ${average_transaction:,.2f}")

    if selected_category == 'All':
        st.success(f"💹 Total Income: ${total_income:,.2f}")
        st.warning(f"💸 Total Expenditure: ${total_expenses:,.2f}")
        net_balance = total_income - total_expenses
        st.info(f"🏦 Net Balance: ${net_balance:,.2f}")

with col2:
    st.subheader("Time-based Analysis")
    monthly_data = filtered_df.resample('M', on='date')['Transaction Amount'].sum().reset_index()
    best_month = monthly_data.loc[monthly_data['Transaction Amount'].idxmax(), 'date'].strftime('%B %Y')
    worst_month = monthly_data.loc[monthly_data['Transaction Amount'].idxmin(), 'date'].strftime('%B %Y')
    st.success(f"📅 Month with Highest Net Income: {best_month}")
    st.warning(f"📅 Month with Lowest Net Income: {worst_month}")

    recent_trend = filtered_df.sort_values('date').tail(3)['Transaction Amount'].pct_change().mean()
    if recent_trend > 0:
        st.info("📈 There's a positive trend in recent transaction amounts.")
    elif recent_trend < 0:
        st.info("📉 There's a negative trend in recent transaction amounts.")
    else:
        st.info("➡️ Recent transaction amounts have remained stable.")

# New section for Predictions
st.header("Predictions")

# Get predictions
future_income, future_expenditure = predict_future_transactions(df)

# Display predictions
col1, col2 = st.columns(2)

with col1:
    st.subheader("Predicted Income")
    chart_predicted_income = alt.Chart(future_income).mark_line().encode(
        x='date:T',
        y='prediction:Q',
        tooltip=['date:T', 'prediction:Q']
    ).properties(
        title='Predicted Daily Income for Next 30 Days'
    )
    st.altair_chart(chart_predicted_income, use_container_width=True)
    st.write(f"Total Predicted Income: ${future_income['prediction'].sum():.2f}")

with col2:
    st.subheader("Predicted Expenditure")
    chart_predicted_expenditure = alt.Chart(future_expenditure).mark_line().encode(
        x='date:T',
        y='prediction:Q',
        tooltip=['date:T', 'prediction:Q']
    ).properties(
        title='Predicted Daily Expenditure for Next 30 Days'
    )
    st.altair_chart(chart_predicted_expenditure, use_container_width=True)
    st.write(f"Total Predicted Expenditure: ${future_expenditure['prediction'].sum():.2f}")

# Predicted Net Balance
predicted_net_balance = future_income['prediction'].sum() - future_expenditure['prediction'].sum()
st.info(f"Predicted Net Balance for Next 30 Days: ${predicted_net_balance:.2f}")

# Download data
st.header("Download Data")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download filtered data as CSV",
    data=csv,
    file_name="filtered_financial_data.csv",
    mime="text/csv",
)
