import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime, date
import yfinance as yf

# Function to check if it's a weekend day (Saturday or Sunday)
def is_weekend():
    today = date.today()
    return today.weekday() in [5, 6]  # 5 is Saturday, 6 is Sunday

# Check if it's a weekend and display warning message
if is_weekend():
    print("Sorry, the app doesn't work on Saturdays and Sundays.")
    exit()

# Get stock symbol from user input
stock_symbol = input("ENTER STOCK SYMBOL FROM YAHOO STOCK: ")

# Get stock data
df = yf.download(stock_symbol, start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'))
df = df.dropna()

# Rest of your code...

# Check if the user has entered a stock symbol
if stock_symbol:
    # Get stock data
    df = yf.download(stock_symbol, start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'))
    df = df.dropna()

    # Prepare data for training
    X = df[['Open', 'High', 'Low', 'Volume']].values
    y_close = df['Close'].values
    y_open = df['Open'].values
    y_avg = (df['Open'].values + df['Close'].values + df['High'].values + df['Low'].values) / 4

    # Split data into training and testing sets
    X_train, X_test, y_close_train, y_close_test, y_open_train, y_open_test, y_avg_train, y_avg_test = train_test_split(
        X, y_close, y_open, y_avg, test_size=0.2, random_state=0
    )

    # Train the model for closing price
    regressor_close = LinearRegression()
    regressor_close.fit(X_train, y_close_train)

    # Train the model for opening price
    regressor_open = LinearRegression()
    regressor_open.fit(X_train, y_open_train)

    # Train the model for average price
    regressor_avg = LinearRegression()
    regressor_avg.fit(X_train, y_avg_train)

    # Make predictions for closing price
    y_close_pred = regressor_close.predict(X_test)

    # Make predictions for opening price
    y_open_pred = regressor_open.predict(X_test)

    # Make predictions for average price
    y_avg_pred = regressor_avg.predict(X_test)

    # Print predictions
    predictions = pd.DataFrame({'Actual Open': y_open_test, 'Predicted Open': y_open_pred})
    st.write(predictions)

    # Predict tomorrow's prices
    last_row = df.tail(1)
    last_row = np.array(last_row[['Open', 'High', 'Low', 'Volume']])
    tomorrow_close = regressor_close.predict(last_row)[0]
    tomorrow_open = regressor_open.predict(last_row)[0]
    tomorrow_avg = regressor_avg.predict(last_row)[0]
    st.write("Tomorrow's opening price - Open: {}".format(tomorrow_close, tomorrow_open, tomorrow_avg))
