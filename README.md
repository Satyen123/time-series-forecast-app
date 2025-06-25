# Time Series Forecast App

This is a Streamlit-based web application for forecasting stock price data using multiple time series models including ARIMA, SARIMA, Prophet, and LSTM.

---

## Project Description

This app helps visualize, analyze, and forecast stock prices (e.g., Google stock) using different forecasting models. It is built as part of a data science internship project and allows users to:

- Upload their own CSV time series data
- Choose a forecasting model (ARIMA, SARIMA, Prophet, LSTM)
- View plots of historical and predicted values
- Download forecast results (optional)

---

## Technologies Used

- Python 3.x
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn
- Statsmodels (for ARIMA & SARIMA)
- Prophet (Facebook)
- Keras/TensorFlow (for LSTM)

---

## 📁 Project Structure

```
time-series-forecast-app/
├── streamlit_app.py         #  Main Streamlit web application
├── GOOG.csv                 #  Sample dataset (Google stock prices)
├── requirements.txt         #  Required Python packages for deployment
└── README.md                #  Project documentation and instructions
```

---

##  Run the App Locally

### 1. Clone the repository

```bash
git clone https://github.com/Satyen123/time-series-forecast-app.git
cd time-series-forecast-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

---

##  Deploy on Streamlit Cloud

1. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Sign in with GitHub
3. Click “New app”
4. Connect to this repository: `Satyen123/time-series-forecast-app`
5. Set the file path to: `streamlit_app.py`
6. Click **Deploy**

---

##  Sample Output

- Line plots of forecasted vs. actual values
- Model selection dropdown
- Time range input controls
- Performance metrics (coming soon)

---

##  Author

**Satyen Sharma**  
B.Tech CSE, Assam Kaziranga University  
GitHub: [Satyen123](https://github.com/Satyen123)

---

##  License

This project is for educational purposes and is not intended for commercial use. All rights reserved.
