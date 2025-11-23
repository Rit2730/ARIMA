import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import investpy_reborn as investpy
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ARIMA Dashboard", layout="wide")

# -------------------------------------
# Safe Chart Wrapper (No matplotlib needed)
# -------------------------------------
def safe_line_chart(df, title=""):
    st.subheader(title)
    st.line_chart(df)

# -------------------------------------
# Fetch Data
# -------------------------------------
def fetch_nifty():
    df = investpy.get_index_historical_data(
        index="Nifty 50",
        country="India",
        from_date="01/01/2000",
        to_date=pd.Timestamp.today().strftime("%d/%m/%Y")
    )
    df.index = pd.to_datetime(df.index)
    return df

def monthly_close(df):
    m = df["Close"].resample("M").last()
    m.index = m.index.to_period('M').to_timestamp()
    m.name = "price"
    return m

def ensure_series(s):
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:,0]
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).to_period('M').to_timestamp()
    return s

def forecast_arima(train, order=(1,1,1), steps=12, start="2020-01-01"):
    model = ARIMA(train, order=order)
    fit = model.fit()
    pred = fit.get_forecast(steps=steps)
    mean = pred.predicted_mean
    idx = pd.date_range(start=start, periods=steps, freq="M")
    mean.index = idx
    return fit, mean

def calc_metrics(actual, forecast):
    n = min(len(actual), len(forecast))
    a = actual.values[:n]
    f = forecast.values[:n]
    mae = mean_absolute_error(a, f)
    rmse = np.sqrt(mean_squared_error(a, f))
    mape = np.mean(np.abs((a - f) / a))*100
    return mae, rmse, mape

# -------------------------------------
# Load Data
# -------------------------------------
st.title("ARIMA Forecasting – Nifty Monthly Close (NO MATPLOTLIB)")

with st.spinner("Fetching Nifty 50 data..."):
    df = fetch_nifty()

monthly = monthly_close(df)

p = st.sidebar.number_input("ARIMA p", 0, 5, 1)
d = st.sidebar.number_input("ARIMA d", 0, 2, 1)
q = st.sidebar.number_input("ARIMA q", 0, 5, 1)
order = (p,d,q)

forecast_months = st.sidebar.slider("Months to Forecast", 6, 24, 12)

tab1, tab2 = st.tabs(["Project 1 (2010–2018 → 2019)", "Project 2 (2021–2025 → 2026)"])


# -----------------------
# Project 1
# -----------------------
with tab1:
    st.header("Project 1 — Train 2010–2018 → Forecast 2019")

    train = monthly["2010":"2018"]
    actual2019 = monthly["2019"]

    safe_line_chart(train, "Monthly Close (2010–2018)")

    fit1, f1 = forecast_arima(train, order=order, steps=forecast_months, start="2019-01-01")

    df_plot = pd.DataFrame({"History": train, "Forecast": f1})
    safe_line_chart(df_plot, "History + Forecast (2019)")

    if len(actual2019) > 0:
        compare = pd.DataFrame({
            "Actual": actual2019.values[:len(f1)],
            "Forecast": f1.values[:len(actual2019)]
        }, index=actual2019.index[:len(f1)])

        st.subheader("Forecast vs Actual (2019)")
        st.dataframe(compare)

        mae, rmse, mape = calc_metrics(actual2019, f1)
        st.write(f"**MAE:** {mae:.3f}")
        st.write(f"**RMSE:** {rmse:.3f}")
        st.write(f"**MAPE:** {mape:.2f}%")
    else:
        st.info("No 2019 data available in monthly series.")


# -----------------------
# Project 2
# -----------------------
with tab2:
    st.header("Project 2 — Train 2021–2025 → Forecast 2026")

    train2 = monthly["2021":"2025"]
    actual2026 = monthly["2026"]

    safe_line_chart(train2, "Monthly Close (2021–2025)")

    fit2, f2 = forecast_arima(train2, order=order, steps=forecast_months, start="2026-01-01")

    df_plot2 = pd.DataFrame({"History": train2, "Forecast": f2})
    safe_line_chart(df_plot2, "History + Forecast (2026)")


    # Backtest 2025
    st.subheader("Backtest: Train 2021–2024 → Predict 2025")

    bt_train = monthly["2021":"2024"]
    bt_test = monthly["2025"]

    if len(bt_train) > 6:
        _, f_bt = forecast_arima(bt_train, order=order, steps=len(bt_test), start="2025-01-01")

        comp_bt = pd.DataFrame({
            "Actual": bt_test.values,
            "Forecast": f_bt.values
        }, index=bt_test.index)

        st.dataframe(comp_bt)

        mae, rmse, mape = calc_metrics(bt_test, f_bt)
        st.write(f"**MAE:** {mae:.3f}")
        st.write(f"**RMSE:** {rmse:.3f}")
        st.write(f"**MAPE:** {mape:.2f}%")
    else:
        st.info("Not enough data for backtest.")
