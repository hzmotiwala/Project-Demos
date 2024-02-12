import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def fetch_historical_data(symbol):
    # Fetch historical stock market data using yfinance
    data = yf.download(symbol, start='2020-01-01', end='2022-01-01')['Adj Close']

    return data

def display_historical_data(data):
    # Display historical stock market data using a line chart
    st.subheader("Historical Stock Prices")
    st.line_chart(data)

def predict_future_prices(data):
    # Prepare data for training the model
    X = pd.DataFrame(data[:-1])  # Features (historical prices)
    y = pd.DataFrame(data[1:])   # Target (next day's price)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)

    return y_test, y_pred, mse

def display_prediction_results(y_test, y_pred, mse):
    # Display prediction results
    st.subheader("Prediction Results")
    st.write(f"Mean Squared Error: {mse}")

    # Sort data by date
    y_test = y_test.sort_index()
    y_pred = pd.DataFrame(y_pred, index=y_test.index).sort_index()

    # Plot actual vs. predicted prices
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_test.index, y_test, label="Actual Prices")
    ax.plot(y_pred.index, y_pred, label="Predicted Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Actual vs. Predicted Prices")
    ax.legend()
    st.pyplot(fig)

def main():
    # Set page title
    st.title("Stock Market Analysis and Prediction")

    # Sidebar section
    st.sidebar.title("Options")

    # Add widgets to sidebar
    selected_stock = st.sidebar.selectbox("Select Stock", ["AAPL", "GOOGL", "MSFT", "AMZN"])
    prediction_period = st.sidebar.slider("Prediction Period (days)", min_value=1, max_value=30, value=7)

    # Display selected stock
    st.write(f"You selected stock: {selected_stock}")

    # Fetch historical stock market data
    data = fetch_historical_data(selected_stock)

    # Display historical stock market data
    display_historical_data(data)

    # Predict future stock prices
    y_test, y_pred, mse = predict_future_prices(data)

    # Display prediction results
    display_prediction_results(y_test, y_pred, mse)

if __name__ == "__main__":
    main()
