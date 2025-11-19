import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# --------------------------
# App Title
# --------------------------
st.title("📈 Stock Price Next-Day Prediction")
st.write("Upload a CSV file with historical stock prices (must contain `Date` and `Close`).")

# --------------------------
# Step 1: Upload CSV
# --------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # Extract symbol from filename (remove extension)
    file_name = os.path.splitext(uploaded_file.name)[0]
    symbol = file_name.upper()

    # Fetch stock details using yfinance
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        st.subheader("📊 Stock Details")
        st.write(f"**Symbol:** {symbol}")
        st.write(f"**Name:** {info.get('longName', 'N/A')}")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        st.write(f"**Exchange:** {info.get('exchange', 'N/A')}")
        st.write(f"**Market Cap:** {info.get('marketCap', 'N/A'):,}")
    except Exception as e:
        st.warning("⚠️ Could not fetch stock details. Symbol may be invalid or yfinance is missing info.")

    # Load data
    df = pd.read_csv(uploaded_file)

    # Check required columns
    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("CSV must contain 'Date' and 'Close' columns.")
    else:
        # Prepare data
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # --------------------------
        # Step 2: Plot historical prices
        # --------------------------
        st.subheader("📉 Historical Closing Prices")
        plt.figure(figsize=(12,6))
        plt.plot(df['Date'], df['Close'], label="Closing Price", color="blue")
        plt.plot(df['Date'], df['Close'].rolling(window=30).mean(), label="30-Day Moving Avg", color="orange")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title(f"{symbol} Historical Prices")
        plt.legend()
        plt.grid(alpha=0.3)
        st.pyplot(plt)

        # --------------------------
        # Step 3: Prepare Data for LSTM
        # --------------------------
        data = df['Close'].values.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data)

        # Use all but last day for training
        train_data = scaled_data[:-1]

        def create_sequences(data, seq_length=60):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i-seq_length:i, 0])
                y.append(data[i,0])
            return np.array(X), np.array(y)

        seq_length = 60
        X_train, y_train = create_sequences(train_data, seq_length)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # --------------------------
        # Step 4: Build Model
        # --------------------------
        model = Sequential()
        model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1],1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam')

        # --------------------------
        # Step 5: Train Model
        # --------------------------
        st.subheader("⚙️ Training the model...")
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])
        st.success("✅ Model trained successfully!")

        # --------------------------
        # Step 6: Predict Next Day
        # --------------------------
        last_60_days = scaled_data[-seq_length:]
        X_test = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
        predicted_scaled = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

        st.subheader("📌 Next-Day Prediction")
        st.write(f"Predicted Closing Price for **{symbol}**: **${predicted_price:.2f}**")

        # --------------------------
        # Step 7: Plot Prediction
        # --------------------------
        st.subheader("🔮 Last 30 Days + Prediction")
        plt.figure(figsize=(12,6))
        plt.plot(df['Date'][-30:], df['Close'][-30:], label='Actual Price (Last 30 Days)', color='blue')
        plt.scatter(df['Date'].iloc[-1] + pd.Timedelta(days=1), predicted_price, color='red', label='Predicted Next Day')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title(f"{symbol} Prediction vs Last 30 Days")
        plt.legend()
        plt.grid(alpha=0.3)
        st.pyplot(plt)
