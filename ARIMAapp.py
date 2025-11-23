# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import investpy_reborn as investpy
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ARIMA Dashboard — NIFTY (monthly close)", layout="wide")

UPLOADED_CSV_PATH = "/mnt/data/NYSE_arima_data.csv"  # available in environment

# -----------------------
# Utilities (defensive)
# -----------------------
def fetch_nifty_daily(from_date="01/01/2000", to_date=None):
    if to_date is None:
        to_date = pd.Timestamp.today().strftime("%d/%m/%Y")
    try:
        df = investpy.get_index_historical_data(index="Nifty 50", country="India",
                                                from_date=from_date, to_date=to_date)
        # ensure Close column exists
        if 'Close' not in df.columns and 'close' in df.columns:
            df.rename(columns={'close': 'Close'}, inplace=True)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception as e:
        st.error("Error fetching data from investpy_reborn: " + str(e))
        return pd.DataFrame()

def monthly_close(series_df):
    if series_df.empty:
        return pd.Series(dtype=float)
    s = series_df['Close'].resample('M').last().dropna()
    s.index = pd.to_datetime(s.index).to_period('M').to_timestamp()
    s.name = 'price'
    return s

def ensure_series(x):
    if isinstance(x, pd.DataFrame):
        if 'price' in x.columns:
            s = x['price'].copy()
        else:
            s = x.iloc[:, 0].copy()
    elif isinstance(x, pd.Series):
        s = x.copy()
    else:
        s = pd.Series(x)
    s = pd.to_numeric(s, errors='coerce').dropna()
    try:
        s.index = pd.to_datetime(s.index).to_period('M').to_timestamp()
    except Exception:
        pass
    s.name = 'price'
    return s

def fit_arima_forecast(train_series, order=(1,1,1), steps=12, forecast_start=None):
    s = ensure_series(train_series)
    if len(s) < 6:
        raise ValueError("Need at least 6 monthly observations to fit ARIMA.")
    model = ARIMA(s, order=order)
    fitted = model.fit()
    res = fitted.get_forecast(steps=steps)
    mean = pd.Series(np.array(res.predicted_mean).flatten())
    ci = res.conf_int()
    lower = pd.Series(np.array(ci.iloc[:,0]).flatten())
    upper = pd.Series(np.array(ci.iloc[:,1]).flatten())
    if forecast_start is None:
        start_idx = s.index[-1] + pd.offsets.MonthBegin()
    else:
        start_idx = pd.to_datetime(forecast_start)
    idx = pd.date_range(start=start_idx, periods=steps, freq='MS')
    mean.index = idx; lower.index = idx; upper.index = idx
    mean.name = 'forecast'; lower.name = 'lower'; upper.name = 'upper'
    return fitted, mean, lower, upper

def metrics(actual, forecast):
    a = ensure_series(actual); f = ensure_series(forecast)
    n = min(len(a), len(f))
    if n==0:
        return np.nan, np.nan, np.nan
    a_n, f_n = a.values[:n].astype(float), f.values[:n].astype(float)
    mae = mean_absolute_error(a_n, f_n)
    rmse = np.sqrt(mean_squared_error(a_n, f_n))
    mape = np.mean(np.abs((a_n - f_n) / a_n)) * 100
    return mae, rmse, mape

def acf_pacf_plot(series, nlags=24):
    s = ensure_series(series)
    if len(s) < 2:
        return None, None
    acf_vals = acf(s, nlags=nlags, fft=False, missing='conservative')
    pacf_vals = pacf(s, nlags=nlags, method='ywmle')
    return acf_vals, pacf_vals

def to_csv_bytes(df):
    buf = BytesIO()
    df.to_csv(buf)
    buf.seek(0)
    return buf

# small helper to save matplotlib fig to bytes
def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

# -----------------------
# UI - Sidebar
# -----------------------
st.sidebar.header("Configuration")
st.sidebar.write("Local uploaded file (available):")
st.sidebar.text(UPLOADED_CSV_PATH)
try:
    st.sidebar.image(UPLOADED_CSV_PATH, width=160)
except Exception:
    pass

st.sidebar.markdown("Data source: investpy-reborn (Nifty 50 historical)")
p = int(st.sidebar.number_input("ARIMA p", min_value=0, max_value=5, value=1))
d = int(st.sidebar.number_input("ARIMA d", min_value=0, max_value=2, value=1))
q = int(st.sidebar.number_input("ARIMA q", min_value=0, max_value=5, value=1))
order = (p,d,q)
forecast_months = int(st.sidebar.slider("Forecast months (out-of-sample)", 6, 24, 12))

# -----------------------
# Fetch & prepare data
# -----------------------
with st.spinner("Fetching Nifty 50 data (this may take a few seconds)..."):
    raw = fetch_nifty_daily(from_date="01/01/2000")
if raw.empty:
    st.error("Failed to fetch historical data. Ensure investpy-reborn is installed and network is available.")
    st.stop()

monthly = monthly_close(raw)

# -----------------------
# Tabs (Project 1 & Project 2)
# -----------------------
tab1, tab2 = st.tabs(["Project 1 (2010–2018 → 2019)", "Project 2 (2021–2025 → 2026)"])

# Project 1
with tab1:
    st.header("Project 1 — Train 2010–2018 → Forecast 2019")
    train = monthly.loc["2010-01-01":"2018-12-31"]
    actual_2019 = monthly.loc["2019-01-01":"2019-12-31"]

    st.subheader("1) Monthly Price Movement (Training)")
    if len(train)>0:
        fig, ax = plt.subplots(figsize=(10,3.5))
        ax.plot(train.index, train.values, linewidth=1.8)
        ax.set_title("Monthly Close — 2010 to 2018")
        ax.set_xlabel("Date"); ax.set_ylabel("Price")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Training window has no monthly data.")

    # Fit & forecast
    try:
        fitted1, mean1, lower1, upper1 = fit_arima_forecast(train, order=order, steps=forecast_months, forecast_start="2019-01-01")
    except Exception as e:
        st.error("ARIMA fit failed: " + str(e)); st.stop()

    st.subheader("2) ARIMA Forecast Overlaid on Actual (2019)")
    fig, ax = plt.subplots(figsize=(10,4))
    if len(train)>0:
        ax.plot(train.index, train.values, label="History", linewidth=1.5)
    ax.plot(mean1.index, mean1.values, linestyle='--', label="Forecast", linewidth=1.5)
    if len(actual_2019)>0:
        minlen = min(len(actual_2019), len(mean1))
        ax.plot(actual_2019.index[:minlen], actual_2019.values[:minlen], label="Actual (2019)", linewidth=1.5)
    ax.legend(); ax.set_title("History + Forecast (2019)")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("3) Forecast Only (2019)")
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(mean1.index, mean1.values, linewidth=1.8)
    ax.set_title("Forecasted Monthly Prices — 2019")
    st.pyplot(fig); plt.close(fig)

    # Comparison & metrics
    st.subheader("Forecast vs Actual Comparison (2019)")
    a = ensure_series(actual_2019); f = ensure_series(mean1)
    min_len = min(len(a), len(f))
    if min_len>0:
        comp = pd.DataFrame({"Actual": a.values[:min_len], "Forecast": f.values[:min_len]}, index=a.index[:min_len])
        st.dataframe(comp.style.format("{:.2f}"))
        mae, rmse, mape = metrics(a[:min_len], f[:min_len])
        c1,c2,c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.3f}")
        c2.metric("RMSE", f"{rmse:.3f}")
        c3.metric("MAPE", f"{mape:.2f}%")
        # downloads
        st.download_button("Download comparison CSV", data=to_csv_bytes(comp), file_name="project1_comparison_2019.csv")
    else:
        st.info("No actual monthly-close data for 2019 found in fetched series.")
        st.download_button("Download forecast CSV", data=to_csv_bytes(mean1.to_frame("forecast")), file_name="project1_forecast_2019.csv")

    # Residuals & ACF/PACF
    st.subheader("Residual Diagnostics")
    resid1 = ensure_series(fitted1.resid)
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(resid1.index, resid1.values); ax.set_title("Residuals (Training)")
    st.pyplot(fig); plt.close(fig)
    acf_vals, pacf_vals = acf_pacf_plot(train)
    if acf_vals is not None:
        fig, ax = plt.subplots(figsize=(8,2.5)); ax.bar(range(len(acf_vals)), acf_vals); ax.set_title("ACF")
        st.pyplot(fig); plt.close(fig)
    if pacf_vals is not None:
        fig, ax = plt.subplots(figsize=(8,2.5)); ax.bar(range(len(pacf_vals)), pacf_vals); ax.set_title("PACF")
        st.pyplot(fig); plt.close(fig)

    # Summary & downloads
    st.subheader("Statistical Summary (Training)")
    st.table(train.describe().reset_index().rename(columns={"index":"Statistic", 0:"Value"}))

    st.subheader("Downloads")
    st.download_button("Download training series (CSV)", data=to_csv_bytes(train.to_frame("price")), file_name="project1_train_2010_2018.csv")
    st.download_button("Download forecast (CSV)", data=to_csv_bytes(mean1.to_frame("forecast")), file_name="project1_forecast_2019.csv")

    st.subheader("Observation")
    st.write("ARIMA trained on 2010–2018 monthly closes. Forecast vs actual comparison shown. Residuals and ACF/PACF help validate model adequacy.")

# Project 2
with tab2:
    st.header("Project 2 — Train 2021–2025 → Forecast 2026")
    train2 = monthly.loc["2021-01-01":"2025-12-31"]
    actual_2026 = monthly.loc["2026-01-01":"2026-12-31"]

    st.subheader("1) Monthly Price Movement (Training)")
    if len(train2)>0:
        fig, ax = plt.subplots(figsize=(10,3.5))
        ax.plot(train2.index, train2.values, linewidth=1.8)
        ax.set_title("Monthly Close — 2021 to 2025")
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("Training window has no monthly data.")

    try:
        fitted2, mean2, lower2, upper2 = fit_arima_forecast(train2, order=order, steps=forecast_months, forecast_start="2026-01-01")
    except Exception as e:
        st.error("ARIMA fit failed: " + str(e)); st.stop()

    st.subheader("2) ARIMA Forecast Overlaid on Actual (if present)")
    fig, ax = plt.subplots(figsize=(10,4))
    if len(train2)>0:
        ax.plot(train2.index, train2.values, label="History")
    ax.plot(mean2.index, mean2.values, linestyle='--', label="Forecast")
    if len(actual_2026)>0:
        minlen = min(len(actual_2026), len(mean2))
        ax.plot(actual_2026.index[:minlen], actual_2026.values[:minlen], label="Actual (2026)")
    ax.legend(); st.pyplot(fig); plt.close(fig)

    st.subheader("3) Forecast Only (2026)")
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(mean2.index, mean2.values, linewidth=1.8)
    ax.set_title("Forecasted Monthly Prices — 2026")
    st.pyplot(fig); plt.close(fig)

    # Backtest
    st.subheader("Backtest: Train 2021–2024, Test 2025")
    back_train = monthly.loc["2021-01-01":"2024-12-31"]
    back_test = monthly.loc["2025-01-01":"2025-12-31"]
    if len(back_train)>=6 and len(back_test)>0:
        try:
            fitted_bt, mean_bt, lb_bt, ub_bt = fit_arima_forecast(back_train, order=order, steps=len(back_test), forecast_start=back_test.index[0])
            minlen = min(len(back_test), len(mean_bt))
            comp_bt = pd.DataFrame({"Actual": back_test.values[:minlen], "Forecast": mean_bt.values[:minlen]}, index=back_test.index[:minlen])
            st.dataframe(comp_bt.style.format("{:.2f}"))
            mae_bt, rmse_bt, mape_bt = metrics(back_test[:minlen], mean_bt[:minlen])
            c1,c2,c3 = st.columns(3)
            c1.metric("MAE (backtest)", f"{mae_bt:.3f}")
            c2.metric("RMSE (backtest)", f"{rmse_bt:.3f}")
            c3.metric("MAPE (backtest)", f"{mape_bt:.2f}%")
            st.download_button("Download backtest comparison (CSV)", data=to_csv_bytes(comp_bt), file_name="project2_backtest_2025.csv")
        except Exception as e:
            st.info("Backtest fitting failed: " + str(e))
    else:
        st.info("Not enough data for 2025 backtest.")

    # Residual diagnostics
    st.subheader("Residual Diagnostics")
    resid2 = ensure_series(fitted2.resid)
    fig, ax = plt.subplots(figsize=(10,3)); ax.plot(resid2.index, resid2.values); st.pyplot(fig); plt.close(fig)
    acf_vals2, pacf_vals2 = acf_pacf_plot(train2)
    if acf_vals2 is not None:
        fig, ax = plt.subplots(figsize=(8,2.5)); ax.bar(range(len(acf_vals2)), acf_vals2); st.pyplot(fig); plt.close(fig)
    if pacf_vals2 is not None:
        fig, ax = plt.subplots(figsize=(8,2.5)); ax.bar(range(len(pacf_vals2)), pacf_vals2); st.pyplot(fig); plt.close(fig)

    # Summary & downloads
    st.subheader("Statistical Summary (Training)")
    st.table(train2.describe().reset_index().rename(columns={"index":"Statistic"}))

    st.subheader("Downloads")
    st.download_button("Download training series (CSV)", data=to_csv_bytes(train2.to_frame("price")), file_name="project2_train_2021_2025.csv")
    st.download_button("Download forecast (CSV)", data=to_csv_bytes(mean2.to_frame("forecast")), file_name="project2_forecast_2026.csv")

    st.subheader("Observation")
    st.write("Model trained on 2021–2025 monthly closes. Forecast and backtest results are displayed; residual diagnostics shown for model validation.")
