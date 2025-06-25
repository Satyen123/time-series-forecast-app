import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

st.set_page_config(page_title="Time Series Forecast App", layout="wide")
st.title("📈 Time Series Forecasting App")

file = st.file_uploader("Upload your time series CSV file", type=['csv'])

if file:
    df = pd.read_csv(file)

    # Automatically detect datetime column
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if not date_cols:
        st.error("❌ No date/time column found. Please include a column like 'Date' or 'Timestamp'.")
        st.stop()

    datetime_col = date_cols[0]
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.set_index(datetime_col, inplace=True)

    # Ensure regular frequency for LSTM forecasting
    df = df.asfreq('D')

    st.success(f"Detected date column: `{datetime_col}`")
    st.write("### Preview of Data", df.head())
    st.write("🔢 Dataset Shape:", df.shape)
    st.line_chart(df)

    target = st.selectbox("Select target column for forecasting", df.columns)

    model = st.selectbox("Choose Forecasting Model", ["ARIMA", "SARIMA", "Prophet", "LSTM"])
    periods = st.slider("Forecast Periods", min_value=10, max_value=365, value=30)

    if st.button("Run Forecast"):

        if model == "ARIMA":
            st.subheader("ARIMA Forecast")
            arima_model = ARIMA(df[target], order=(5,1,0))
            arima_result = arima_model.fit()
            arima_forecast = arima_result.forecast(steps=periods)
            st.line_chart(pd.concat([df[target], arima_forecast]))

        elif model == "SARIMA":
            st.subheader("SARIMA Forecast")
            sarima_model = SARIMAX(df[target], order=(1,1,1), seasonal_order=(1,1,1,12))
            sarima_result = sarima_model.fit(disp=False)
            sarima_forecast = sarima_result.forecast(steps=periods)
            st.line_chart(pd.concat([df[target], sarima_forecast]))

        elif model == "Prophet":
            st.subheader("Prophet Forecast")
            prophet_df = df[[target]].reset_index()
            prophet_df.columns = ['ds', 'y']
            prophet_model = Prophet()
            prophet_model.fit(prophet_df)
            future = prophet_model.make_future_dataframe(periods=periods)
            forecast = prophet_model.predict(future)
            fig = prophet_model.plot(forecast)
            st.pyplot(fig)

        elif model == "LSTM":
            st.subheader("LSTM Forecast")

            data = df[[target]].dropna().values
            if len(data) < 50:
                st.warning("⚠️ Not enough data for LSTM. Minimum ~50 data points recommended.")
                st.stop()

            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)

            def create_dataset(data, time_step=30):
                X, y = [], []
                for i in range(len(data)-time_step-1):
                    X.append(data[i:i+time_step, 0])
                    y.append(data[i+time_step, 0])
                return np.array(X), np.array(y)

            time_step = 30
            X, y = create_dataset(data_scaled, time_step)

            st.write("✅ LSTM Training Data Shapes")
            st.write("X:", X.shape)
            st.write("y:", y.shape)

            X = X.reshape(X.shape[0], X.shape[1], 1)

            lstm_model = Sequential()
            lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
            lstm_model.add(LSTM(50))
            lstm_model.add(Dense(1))
            lstm_model.compile(loss='mean_squared_error', optimizer='adam')
            lstm_model.fit(X, y, epochs=10, batch_size=1, verbose=0)

            # Forecast
            last_input = data_scaled[-time_step:]
            predictions = []
            for _ in range(periods):
                pred = lstm_model.predict(last_input.reshape(1, time_step, 1), verbose=0)
                predictions.append(pred[0, 0])
                last_input = np.append(last_input, pred)[1:]

            forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            forecast_index = pd.date_range(start=df.index[-1], periods=periods+1, freq='D')[1:]
            forecast_series = pd.Series(forecast.flatten(), index=forecast_index)

            st.write("🔮 Forecast Preview:", forecast_series.tail())

            combined = pd.concat([df[target], forecast_series])
            st.line_chart(combined)
