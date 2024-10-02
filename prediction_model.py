import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import timedelta

def prepare_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
    return df

def train_model(df, target):
    X = df[['days_since_start']]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def make_predictions(model, df, days_to_predict=30):
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_to_predict)
    future_df = pd.DataFrame({'date': future_dates})
    future_df['days_since_start'] = (future_df['date'] - df['date'].min()).dt.days
    predictions = model.predict(future_df[['days_since_start']])
    future_df['prediction'] = predictions
    return future_df

def predict_future_transactions(df):
    df = prepare_data(df)
    
    income_df = df[df['Transaction Amount'] > 0]
    expenditure_df = df[df['Transaction Amount'] < 0].copy()
    expenditure_df['Transaction Amount'] = expenditure_df['Transaction Amount'].abs()

    income_model = train_model(income_df, 'Transaction Amount')
    expenditure_model = train_model(expenditure_df, 'Transaction Amount')

    future_income = make_predictions(income_model, income_df)
    future_expenditure = make_predictions(expenditure_model, expenditure_df)

    return future_income, future_expenditure