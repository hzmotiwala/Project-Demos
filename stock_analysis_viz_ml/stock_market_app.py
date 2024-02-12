import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

def fetch_historical_data(symbol):
    # Fetch historical stock market data using yfinance
    data = yf.download(symbol, start='2022-01-01', end='2024-01-01')['Adj Close']
    data = pd.DataFrame(data)
    data = data.rename(columns={'Adj Close': 'Adj_Close'})
    st.write(f"data: {data.columns.tolist()}")
    # Calculate additional features: 50-day moving average, RSI, percent change from previous trading day
    data['mov_avg_50'] = data['Adj_Close'].rolling(window=50).mean()
    delta = data['Adj_Close'].diff(1)
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, -0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['rsi'] = rsi.fillna(0)
    data['prev_trading_day_delta'] = data['Adj_Close'].pct_change()
    return data.dropna()  # Drop rows with NaN values

def display_historical_data(data):
    # Display historical stock market data using a line chart
    st.subheader("Historical Stock Prices")
    st.line_chart(data['Adj_Close'])

def predict_future_prices(data):
    # Prepare data for training the model
    X = data.iloc[:, 1:]  # Features (historical prices)
    y = data.iloc[:, 0]   # Target (next day's price)

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
    ax.plot(y_test.index, y_test, label="Actual Prices", color='blue')
    ax.plot(y_pred.index, y_pred, label="Predicted Prices", color='orange')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Actual vs. Predicted Prices")
    
    # Explicitly set legend labels and colors
    ax.legend(["Actual Prices", "Predicted Prices"], loc='upper left')

    st.pyplot(fig)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return history

def evaluate_lstm_model(model, X_test, y_test):
    loss = model.evaluate(X_test, y_test)
    return loss

def make_lstm_predictions(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def main():
    # Set page title
    st.title("Stock Market Analysis and Prediction")

    # Sidebar section
    st.sidebar.title("Options")

    # Add widgets to sidebar
    selected_stock = st.sidebar.selectbox("Select Stock", ["AAPL", "GOOGL", "MSFT", "AMZN"])
    st.sidebar.markdown("---")
    st.sidebar.subheader("Prediction Period")
    st.sidebar.write("Select the number of days for predicting future stock prices.")
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

    # Add section for ANOVA example
    st.subheader("Example: ANOVA Analysis")
    st.write("Here, we'll perform an analysis of variance (ANOVA) to evaluate the impact of different features on stock prices.")
    
    # Perform ANOVA analysis
    model = ols('Adj_Close ~ mov_avg_50 + rsi + prev_trading_day_delta', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    st.write("ANOVA Results:")
    st.write(anova_table)

    # Plot p-values from ANOVA results
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(anova_table.index, anova_table['PR(>F)'], color='green')
    ax.set_xlabel("Features")
    ax.set_ylabel("p-value")
    ax.set_title("ANOVA p-values for Features")
    st.pyplot(fig)

    # Add section for LSTM example
    st.subheader("Example: LSTM Model")
    st.write("Here, we'll implement a Long Short-Term Memory (LSTM) model for time series forecasting.")
    
    # Split data into features (X) and target (y)
    X = data[['mov_avg_50', 'rsi', 'prev_trading_day_delta']]
    y = data['Adj_Close']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape input data for LSTM model (samples, timesteps, features)
    X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

    # Build the LSTM model
    lstm_model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # Train the LSTM model
    train_lstm_model(lstm_model, X_train, y_train)

    # Evaluate the LSTM model
    loss = evaluate_lstm_model(lstm_model, X_test, y_test)

    # Make predictions using the LSTM model
    y_pred_lstm = make_lstm_predictions(lstm_model, X_test)

    # Plot actual vs. predicted prices
    fig_lstm, ax_lstm = plt.subplots(figsize=(10, 6))
    ax_lstm.plot(y_test.index, y_test, label="Actual Prices", color='blue')
    ax_lstm.plot(y_test.index, y_pred_lstm, label="Predicted Prices (LSTM)", color='green')
    ax_lstm.set_xlabel("Date")
    ax_lstm.set_ylabel("Price")
    ax_lstm.set_title("Actual vs. Predicted Prices (LSTM)")
    ax_lstm.legend()
    st.pyplot(fig_lstm)

    # Display evaluation metrics
    st.write(f"Mean Squared Error (LSTM): {loss}")

if __name__ == "__main__":
    main()
