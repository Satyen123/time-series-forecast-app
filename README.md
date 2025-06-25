# time-series-forecast-app
# Time Series Forecasting Web App

A user-friendly Streamlit web application for time series forecasting using popular models like ARIMA, SARIMA, Prophet, and LSTM. This project helps users upload their own datasets (e.g., stock prices, weather data, etc.) and forecast future values with interactive visualizations.

---

##  Features

- Upload your own `.csv` time series dataset
- Apply and compare four forecasting models:
  - ARIMA
  - SARIMA
  - Prophet
  - LSTM (Deep Learning)
- Visualize forecast results with Matplotlib/Plotly
- Customize forecast duration and model parameters
- Clean, responsive UI using Streamlit

---

##  Project Structure
project_internship/
├── Project internship/
│   ├── GOOG.csv                               # Raw dataset (Google stock prices)
│   ├── time_series_analysis (1).ipynb         # Jupyter Notebook for analysis
│   ├── time series forecast app/
│   │   ├── GOOG_cleaned.csv                   # Cleaned time series dataset
│   │   ├── requirements.txt                   # Python dependencies
│   │   ├── streamlit_app.py                   # Main Streamlit application

