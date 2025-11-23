# ---- helpers.py ----
# Utility functions used by app.py
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from io import BytesIO


def fetch_daily(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    return df


def monthly_close_from_daily(df):
    # Use last observed Close of each month
    s = df["Close"].resample("M").last().dropna()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    s.name = "price"
    return s.sort_index()


def to_series_1d(x):
    if isinstance(x, pd.DataFrame):
        if "price" in x.columns:
            s = x["price"].copy()
        else:
            s = x.iloc[:, 0].copy()
    elif isinstance(x, pd.Series):
        s = x.copy()
    else:
        s = pd.Series(x)
    s = pd.to_numeric(s, errors="coerce").dropna()
    try:
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    except Exception:
        pass
    s.name = "price"
    return s


def fit_arima_and_forecast(train_series, order=(1,1,1), steps=12, forecast_start=None):
    s = to_series_1d(train_series)
    if len(s) < 6:
        raise ValueError("Not enough monthly points to fit ARIMA (>=6 required).")
    model = ARIMA(s, order=order)
    fitted = model.fit()
    res = fitted.get_forecast(steps=steps)
    mean = pd.Series(np.array(res.predicted_mean).flatten())
    ci = res.conf_int()
    lower = pd.Series(np.array(ci.iloc[:,0]).flatten())
    upper = pd.Series(np.array(ci.iloc[:,1]).flatten())
    if forecast_start is None:
        idx_start = s.index[-1] + pd.offsets.MonthBegin()
    else:
        idx_start = pd.to_datetime(forecast_start)
    idx = pd.date_range(start=idx_start, periods=steps, freq="MS")
    mean.index = idx
    lower.index = idx
    upper.index = idx
    mean.name = "forecast"
    lower.name = "lower"
    upper.name = "upper"
    return fitted, mean, lower, upper


def compute_metrics(actual, forecast):
    a = to_series_1d(actual)
    f = to_series_1d(forecast)
    n = min(len(a), len(f))
    if n == 0:
        return np.nan, np.nan, np.nan
    a_n = a.values[:n].astype(float)
    f_n = f.values[:n].astype(float)
    mae = mean_absolute_error(a_n, f_n)
    rmse = np.sqrt(mean_squared_error(a_n, f_n))
    mape = np.mean(np.abs((a_n - f_n) / a_n)) * 100
    return mae, rmse, mape


def acf_pacf_figs(series, nlags=24):
    s = to_series_1d(series)
    if len(s) < 2:
        return None, None
    acf_vals = acf(s, nlags=nlags, fft=False, missing='conservative')
    pacf_vals = pacf(s, nlags=nlags, method='ywmle')
    return acf_vals, pacf_vals


def df_to_bytes(df):
    buf = BytesIO()
    df.to_csv(buf)
    buf.seek(0)
    return buf


# ---- app.py ----
# Main Streamlit application. Place this file as app.py and helpers.py side-by-side in your repo.
import streamlit as st
import plotly.graph_objects as go
from helpers import (fetch_daily, monthly_close_from_daily, fit_arima_and_forecast,
                     compute_metrics, acf_pacf_figs, to_series_1d, df_to_bytes)
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ARIMA Forecasting Dashboard", layout="wide")

st.title("ARIMA Forecasting Dashboard — Two Projects")

# Sidebar
st.sidebar.header("Configuration")
# Default ticker: NIFTY 50
ticker = st.sidebar.text_input("Ticker (Yahoo Finance)", value="^NSEI")
project = st.sidebar.selectbox("Project", ["Project 1 (2010–2018 → 2019)", "Project 2 (2021–2025 → 2026)"])
p = int(st.sidebar.number_input("ARIMA p", min_value=0, max_value=5, value=1))
d = int(st.sidebar.number_input("ARIMA d", min_value=0, max_value=2, value=1))
q = int(st.sidebar.number_input("ARIMA q", min_value=0, max_value=5, value=1))
order = (p, d, q)
forecast_horizon = int(st.sidebar.slider("Forecast months (out-of-sample)", min_value=6, max_value=24, value=12))

# show uploaded local image path (developer-provided)
st.sidebar.markdown("Reference image (local):")
st.sidebar.write("/mnt/data/error.png")
try:
    st.sidebar.image("/mnt/data/error.png", width=200)
except Exception:
    pass

# Fetch daily data wide enough
with st.spinner("Fetching daily data from Yahoo Finance..."):
    raw = fetch_daily(ticker, start="2009-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d"))

if raw.empty:
    st.error("No data returned. Check ticker or network.")
    st.stop()

monthly = monthly_close_from_daily(raw)

# Project windows
if project.startswith("Project 1"):
    train_start, train_end = "2010-01-01", "2018-12-31"
    actual_start, actual_end = "2019-01-01", "2019-12-31"
    title = "Project 1: Train 2010–2018 → Forecast 2019"
else:
    train_start, train_end = "2021-01-01", "2025-12-31"
    actual_start, actual_end = "2026-01-01", "2026-12-31"
    title = "Project 2: Train 2021–2025 → Forecast 2026"

st.header(title)

train = monthly.loc[train_start:train_end]
actual = monthly.loc[actual_start:actual_end]

# Graph 1: monthly movement
st.subheader("1) Monthly Price Movement (Training)")
if len(train) == 0:
    st.info("Training window has no monthly-close data — verify ticker/date ranges.")
else:
    fig = go.Figure(go.Scatter(x=train.index, y=train.values, mode='lines'))
    fig.update_layout(title=f"Monthly Close — {train_start[:4]} to {train_end[:4]}", template='simple_white')
    st.plotly_chart(fig, use_container_width=True)

# Fit ARIMA and forecast
try:
    fitted, forecast_mean, lower_ci, upper_ci = fit_arima_and_forecast(train, order=order, steps=forecast_horizon, forecast_start=actual_start)
except Exception as e:
    st.error(f"ARIMA fit failed: {e}")
    st.stop()

# Graph 2: overlay
st.subheader("2) ARIMA Forecast Overlaid on Actual")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=train.index, y=train.values, mode='lines', name='History'))
fig2.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean.values, mode='lines', name='Forecast', line=dict(dash='dash')))
if len(actual) > 0:
    min_len = min(len(actual), len(forecast_mean))
    fig2.add_trace(go.Scatter(x=actual.index[:min_len], y=actual.values[:min_len], mode='lines', name='Actual'))
fig2.update_layout(title='History + Forecast + Actual (where available)', template='simple_white')
st.plotly_chart(fig2, use_container_width=True)

# Graph 3: future forecast
st.subheader("3) Forecast (Out-of-Sample)")
fig3 = go.Figure(go.Scatter(x=forecast_mean.index, y=forecast_mean.values, mode='lines'))
fig3.update_layout(title='Out-of-Sample Forecast', template='simple_white')
st.plotly_chart(fig3, use_container_width=True)

# Comparison
st.subheader('Forecast vs Actual Comparison')
actual_1d = to_series_1d(actual)
forecast_1d = to_series_1d(forecast_mean)
min_len = min(len(actual_1d), len(forecast_1d))
if min_len > 0:
    comp_df = pd.DataFrame({'Actual': actual_1d.values[:min_len], 'Forecast': forecast_1d.values[:min_len]}, index=actual_1d.index[:min_len])
    st.dataframe(comp_df.style.format('{:.2f}'))
    mae, rmse, mape = compute_metrics(actual_1d[:min_len], forecast_1d[:min_len])
    c1, c2, c3 = st.columns(3)
    c1.metric('MAE', f"{mae:.3f}")
    c2.metric('RMSE', f"{rmse:.3f}")
    c3.metric('MAPE', f"{mape:.2f}%")
else:
    st.info('No actual monthly-close data available for the forecast window.')
    st.dataframe(forecast_mean.to_frame('Forecast').style.format('{:.2f}'))

# Residuals & diagnostics
st.subheader('Residual Diagnostics')
resid = to_series_1d(fitted.resid)
fig_res = go.Figure(go.Scatter(x=resid.index, y=resid.values, mode='lines'))
fig_res.update_layout(title='Residuals (Training)', template='simple_white')
st.plotly_chart(fig_res, use_container_width=True)

acf_vals, pacf_vals = acf_pacf_figs(train, nlags=24)
if acf_vals is not None:
    fig_acf = go.Figure(go.Bar(x=list(range(len(acf_vals))), y=acf_vals))
    fig_acf.update_layout(title='ACF', template='simple_white')
    st.plotly_chart(fig_acf, use_container_width=True)
if pacf_vals is not None:
    fig_pacf = go.Figure(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals))
    fig_pacf.update_layout(title='PACF', template='simple_white')
    st.plotly_chart(fig_pacf, use_container_width=True)

# Summary
st.subheader('Statistical Summary (Training)')
desc = train.describe()
desc_df = desc.reset_index()
desc_df.columns = ['Statistic', 'Value']
st.table(desc_df)

# Downloads
st.subheader('Downloads')
train_df = train.to_frame(name='price')
forecast_df = forecast_mean.to_frame(name='forecast')
st.download_button('Download training series (CSV)', data=df_to_bytes(train_df), file_name=f'train_{train_start}_{train_end}.csv')
st.download_button('Download forecast (CSV)', data=df_to_bytes(forecast_df), file_name=f'forecast_{forecast_mean.index[0].strftime('%Y%m')}.csv')
if min_len > 0:
    st.download_button('Download comparison (CSV)', data=df_to_bytes(comp_df), file_name='comparison.csv')

# Observation
st.subheader('Observation')
st.write('ARIMA model trained on monthly close prices. Residual diagnostics and metrics provided. Forecasts presented for a professional view of expected short-term movement.')

# ---- requirements.txt ----
# Put this file at repo root (exact versions recommended)
# streamlit==1.22.0
# pandas==2.1.4
# numpy==1.26.4
# yfinance==0.2.40
# statsmodels==0.14.1
# scikit-learn==1.3.2
# plotly==5.18.0
# python-dateutil==2.8.2
