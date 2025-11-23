# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ARIMA Forecasting Dashboard", layout="wide")

# -------------------------
# Utility helpers (defensive)
# -------------------------
def fetch_daily(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    return df

def monthly_close_from_daily(df):
    # Use monthly close (last trading day of each month)
    s = df["Close"].resample("M").last().dropna()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    s.name = "price"
    return s.sort_index()

def to_series_1d(x):
    # Ensure 1-D pandas Series with datetime index if possible
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
    # Build forecast index safely
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

def plot_line(series, title, name="Value"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=name, line=dict(width=2)))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", template="simple_white", height=420)
    return fig

def plot_overlay(history, forecast, actual=None, lower=None, upper=None, title="Overlay"):
    fig = go.Figure()
    if history is not None and len(history) > 0:
        fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines", name="History", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines", name="Forecast", line=dict(width=2, dash="dash")))
    if (lower is not None) and (upper is not None) and (len(lower)==len(upper)==len(forecast)):
        fig.add_trace(go.Scatter(x=upper.index, y=upper.values, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=lower.index, y=lower.values, mode="lines", fill='tonexty', fillcolor='rgba(173,216,230,0.2)', name="95% CI", line=dict(width=0)))
    if actual is not None and len(actual) > 0:
        min_len = min(len(actual), len(forecast))
        fig.add_trace(go.Scatter(x=actual.index[:min_len], y=actual.values[:min_len], mode="lines", name="Actual", line=dict(width=2)))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", template="simple_white", height=440)
    return fig

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
        empty = go.Figure()
        empty.update_layout(title="Not enough data")
        return empty, empty
    acf_vals = acf(s, nlags=nlags, fft=False, missing='conservative')
    pacf_vals = pacf(s, nlags=nlags, method='ywmle')  # supported and stable
    fig_acf = go.Figure(go.Bar(x=list(range(len(acf_vals))), y=acf_vals))
    fig_acf.update_layout(title="ACF", template="simple_white", height=320)
    fig_pacf = go.Figure(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals))
    fig_pacf.update_layout(title="PACF", template="simple_white", height=320)
    return fig_acf, fig_pacf

def df_to_bytes(df):
    buf = BytesIO()
    df.to_csv(buf)
    buf.seek(0)
    return buf

# -------------------------
# UI & flow
# -------------------------
st.title("ARIMA Forecasting Dashboard — Two Projects (NIFTY default)")

st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker (Yahoo Finance) — default NIFTY50", value="^NSEI")
project = st.sidebar.selectbox("Project", ["Project 1 (2010–2018 → 2019)", "Project 2 (2021–2025 → 2026)"])
p = int(st.sidebar.number_input("ARIMA p", min_value=0, max_value=5, value=1))
d = int(st.sidebar.number_input("ARIMA d", min_value=0, max_value=2, value=1))
q = int(st.sidebar.number_input("ARIMA q", min_value=0, max_value=5, value=1))
order = (p, d, q)
forecast_horizon = int(st.sidebar.slider("Forecast months (out-of-sample)", min_value=6, max_value=24, value=12))

# local uploaded image (developer-provided path — available in environment)
uploaded_image_path = "/mnt/data/error.png"
st.sidebar.markdown("Reference image (local):")
st.sidebar.write(uploaded_image_path)
try:
    st.sidebar.image(uploaded_image_path, width=200)
except Exception:
    pass

# fetch data broadly
with st.spinner("Fetching historical daily data from Yahoo Finance..."):
    raw = fetch_daily(ticker, start="2009-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d"))

if raw.empty:
    st.error("No data returned from Yahoo Finance. Check ticker or network.")
    st.stop()

# compute monthly close (month-end last close)
monthly = monthly_close_from_daily(raw)

# choose project windows
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
    st.info("Training window has no monthly-close data — please check ticker and date ranges.")
else:
    st.plotly_chart(plot_line(train, f"Monthly Close — {train_start[:4]} to {train_end[:4]}"), use_container_width=True)

# Fit ARIMA and forecast (forecast_start aligned to actual window start)
try:
    fitted, forecast_mean, lower_ci, upper_ci = fit_arima_and_forecast(train, order=order, steps=forecast_horizon, forecast_start=actual_start)
except Exception as e:
    st.error(f"ARIMA fit failed: {e}")
    st.stop()

# Graph 2: overlay
st.subheader("2) ARIMA Forecast Overlaid on Actual")
st.plotly_chart(plot_overlay(train, forecast_mean, actual=actual, lower=lower_ci, upper=upper_ci, title="History + Forecast"), use_container_width=True)

# Graph 3: forecast only
st.subheader("3) Forecast (Out-of-Sample)")
st.plotly_chart(plot_line(forecast_mean, f"Out-of-Sample Forecast starting {forecast_mean.index[0].strftime('%Y-%m')}"), use_container_width=True)

# Comparison & metrics
st.subheader("Forecast vs Actual Comparison")
actual_1d = to_series_1d(actual)
forecast_1d = to_series_1d(forecast_mean)
min_len = min(len(actual_1d), len(forecast_1d))
if min_len > 0:
    comp_df = pd.DataFrame({"Actual": actual_1d.values[:min_len], "Forecast": forecast_1d.values[:min_len]}, index=actual_1d.index[:min_len])
    st.dataframe(comp_df.style.format("{:.2f}"))
    mae, rmse, mape = compute_metrics(actual_1d[:min_len], forecast_1d[:min_len])
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.3f}")
    c2.metric("RMSE", f"{rmse:.3f}")
    c3.metric("MAPE", f"{mape:.2f}%")
else:
    st.info("No actual monthly-close data for the forecast window. Forecast is shown for presentation.")
    st.dataframe(forecast_mean.to_frame("Forecast").style.format("{:.2f}"))

# Residual diagnostics
st.subheader("Residual Diagnostics")
resid = to_series_1d(fitted.resid)
st.plotly_chart(plot_line(resid, "Residuals — Training"), use_container_width=True)
fig_acf, fig_pacf = acf_pacf_figs(train, nlags=24)
st.plotly_chart(fig_acf, use_container_width=True)
st.plotly_chart(fig_pacf, use_container_width=True)

# Statistical summary
st.subheader("Statistical Summary (Training)")
desc = train.describe()
desc_df = desc.reset_index()
desc_df.columns = ["Statistic", "Value"]
st.table(desc_df)

# Downloads (convert to DataFrame to avoid numpy .to_frame issues)
st.subheader("Downloads")
train_df = train.to_frame(name="price")
forecast_df = forecast_mean.to_frame(name="forecast")
st.download_button("Download training series (CSV)", data=df_to_bytes(train_df), file_name=f"train_{train_start}_{train_end}.csv")
st.download_button("Download forecast (CSV)", data=df_to_bytes(forecast_df), file_name=f"forecast_{forecast_mean.index[0].strftime('%Y%m')}.csv")
if min_len > 0:
    st.download_button("Download comparison (CSV)", data=df_to_bytes(comp_df), file_name="comparison.csv")

# Observation
st.subheader("Observation")
st.write(
    "The ARIMA model is trained on monthly close prices and provides an out-of-sample monthly forecast. "
    "When actual monthly-close values for the forecast window exist they are used to evaluate performance. Residual diagnostics and ACF/PACF are presented for model validation. Forecasts are produced using ARIMA only."
)
