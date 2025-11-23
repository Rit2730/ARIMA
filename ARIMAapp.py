# app.py
import streamlit as st
import pandas as pd
import numpy as np
import investpy
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ARIMA Forecasting — NIFTY (investpy)", layout="wide")

# Developer-provided uploaded CSV path (if you want to reference)
UPLOADED_CSV_PATH = "/mnt/data/NYSE_arima_data.csv"

# -------------------------
# Utility functions (defensive)
# -------------------------
def fetch_nifty_daily(from_date="01/01/2000", to_date=None):
    """Fetch NIFTY 50 daily data using investpy (dates in dd/mm/yyyy)."""
    if to_date is None:
        to_date = pd.Timestamp.today().strftime("%d/%m/%Y")
    try:
        df = investpy.get_index_historical_data(index='NIFTY 50', country='india',
                                                from_date=from_date, to_date=to_date)
        # investpy returns index as datetime already; ensure columns include 'Close'
        if 'Close' in df.columns:
            df = df[['Close']].rename(columns={'Close': 'Close'})
        elif 'close' in df.columns:
            df = df[['close']].rename(columns={'close': 'Close'})
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception as e:
        # Return empty DataFrame (app will handle gracefully)
        return pd.DataFrame()

def monthly_close_from_daily(df):
    """Return month-end last close series (clean 1-D pandas Series)."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    s = df['Close'].resample('M').last().dropna()
    s.index = pd.to_datetime(s.index).to_period('M').to_timestamp()
    s.name = 'price'
    return s.sort_index()

def to_1d_series(x):
    """Convert input into a clean 1-D pandas Series indexed by month-start timestamps where possible."""
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

def fit_arima_and_forecast(train_series, order=(1,1,1), steps=12, forecast_start=None):
    s = to_1d_series(train_series)
    if len(s) < 6:
        raise ValueError("At least 6 monthly observations are required to fit ARIMA.")
    model = ARIMA(s, order=order)
    fitted = model.fit()
    res = fitted.get_forecast(steps=steps)
    mean = pd.Series(np.array(res.predicted_mean).flatten())
    ci = res.conf_int()
    lower = pd.Series(np.array(ci.iloc[:, 0]).flatten())
    upper = pd.Series(np.array(ci.iloc[:, 1]).flatten())
    # Build forecast index safely
    if forecast_start is None:
        idx_start = s.index[-1] + pd.offsets.MonthBegin()
    else:
        idx_start = pd.to_datetime(forecast_start)
    fc_index = pd.date_range(start=idx_start, periods=steps, freq='MS')
    mean.index = fc_index
    lower.index = fc_index
    upper.index = fc_index
    mean.name = 'forecast'
    lower.name = 'lower'
    upper.name = 'upper'
    return fitted, mean, lower, upper

def compute_metrics(actual, forecast):
    a = to_1d_series(actual)
    f = to_1d_series(forecast)
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
    s = to_1d_series(series)
    if len(s) < 2:
        return None, None
    acf_vals = acf(s, nlags=nlags, fft=False, missing='conservative')
    pacf_vals = pacf(s, nlags=nlags, method='ywmle')  # stable method
    fig_acf = go.Figure(go.Bar(x=list(range(len(acf_vals))), y=acf_vals))
    fig_acf.update_layout(title='ACF', template='simple_white', height=320)
    fig_pacf = go.Figure(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals))
    fig_pacf.update_layout(title='PACF', template='simple_white', height=320)
    return fig_acf, fig_pacf

def df_to_bytes(df):
    buf = BytesIO()
    df.to_csv(buf)
    buf.seek(0)
    return buf

def plot_line(series, title, name='Value'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=name, line=dict(width=2)))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price', template='simple_white', height=420)
    return fig

def plot_forecast_overlay(history, forecast, actual=None, lower=None, upper=None, title='Forecast vs Actual'):
    fig = go.Figure()
    if history is not None and len(history) > 0:
        fig.add_trace(go.Scatter(x=history.index, y=history.values, mode='lines', name='History', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines', name='Forecast', line=dict(width=2, dash='dash')))
    if lower is not None and upper is not None and len(lower) == len(upper) == len(forecast):
        fig.add_trace(go.Scatter(x=upper.index, y=upper.values, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=lower.index, y=lower.values, mode='lines', fill='tonexty', fillcolor='rgba(173,216,230,0.2)', name='95% CI', line=dict(width=0)))
    if actual is not None and len(actual) > 0:
        min_len = min(len(actual), len(forecast))
        fig.add_trace(go.Scatter(x=actual.index[:min_len], y=actual.values[:min_len], mode='lines', name='Actual', line=dict(width=2)))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price', template='simple_white', height=440)
    return fig

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Configuration")
st.sidebar.write("Uploaded CSV (local):")
st.sidebar.write(UPLOADED_CSV_PATH)
try:
    st.sidebar.image(UPLOADED_CSV_PATH, width=160)
except Exception:
    pass

st.sidebar.markdown("Default data source: **investpy (NIFTY 50)**")
ticker = st.sidebar.text_input("Index (fixed to NIFTY 50 for investpy usage)", value="NIFTY 50")
p = int(st.sidebar.number_input("ARIMA p", min_value=0, max_value=5, value=1))
d = int(st.sidebar.number_input("ARIMA d", min_value=0, max_value=2, value=1))
q = int(st.sidebar.number_input("ARIMA q", min_value=0, max_value=5, value=1))
order = (p, d, q)
forecast_months = int(st.sidebar.slider("Forecast months (out-of-sample)", min_value=6, max_value=24, value=12))

# -------------------------
# Fetch NIFTY daily and build monthly close
# -------------------------
with st.spinner("Fetching NIFTY 50 historical data from investpy..."):
    raw_daily = fetch_nifty_daily(from_date="01/01/2000")

if raw_daily.empty:
    st.error("Could not fetch data from investpy. Check connectivity or investpy package installation.")
    st.stop()

monthly = monthly_close_from_daily(raw_daily)

# -------------------------
# Tabs for Project 1 & Project 2
# -------------------------
tab1, tab2 = st.tabs(["Project 1 (2010–2018 → 2019)", "Project 2 (2021–2025 → 2026)"])

# -------- Project 1 --------
with tab1:
    st.header("Project 1 — Train 2010–2018 → Forecast 2019")
    train = monthly.loc["2010-01-01":"2018-12-31"]
    actual_2019 = monthly.loc["2019-01-01":"2019-12-31"]

    st.subheader("1) Monthly Price Movement (Training)")
    if len(train) == 0:
        st.info("No training data in selected window.")
    else:
        st.plotly_chart(plot_line(train, "Monthly Close — 2010–2018"), use_container_width=True)

    # Fit & forecast
    try:
        fitted1, mean1, lower1, upper1 = fit_arima_and_forecast(train, order=order, steps=forecast_months, forecast_start="2019-01-01")
    except Exception as e:
        st.error(f"ARIMA fit failed: {e}")
        st.stop()

    st.subheader("2) ARIMA Forecast Overlaid on Actual (2019)")
    st.plotly_chart(plot_forecast_overlay(train, mean1, actual=actual_2019, lower=lower1, upper=upper1, title="History + Forecast (2019)"), use_container_width=True)

    st.subheader("3) Forecast Only (2019 out-of-sample)")
    st.plotly_chart(plot_line(mean1, "Forecasted Monthly Prices — 2019"), use_container_width=True)

    # Comparison & metrics
    st.subheader("Forecast vs Actual Comparison (2019)")
    a = to_1d_series(actual_2019)
    f = to_1d_series(mean1)
    min_len = min(len(a), len(f))
    if min_len > 0:
        comp_df = pd.DataFrame({"Actual": a.values[:min_len], "Forecast": f.values[:min_len]}, index=a.index[:min_len])
        st.dataframe(comp_df.style.format("{:.2f}"))
        mae, rmse, mape = compute_metrics(a[:min_len], f[:min_len])
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.3f}")
        c2.metric("RMSE", f"{rmse:.3f}")
        c3.metric("MAPE", f"{mape:.2f}%")
    else:
        st.info("Actual monthly-close data for 2019 not available in series.")
        st.dataframe(mean1.to_frame("Forecast").style.format("{:.2f}"))

    # Residual diagnostics
    st.subheader("Residual Diagnostics (Training)")
    resid1 = to_1d_series(fitted1.resid)
    st.plotly_chart(plot_line(resid1, "Residuals (Training)"), use_container_width=True)
    fig_acf1, fig_pacf1 = acf_pacf_figs(train, nlags=24)
    if fig_acf1 is not None:
        st.plotly_chart(fig_acf1, use_container_width=True)
    if fig_pacf1 is not None:
        st.plotly_chart(fig_pacf1, use_container_width=True)

    # Statistical summary
    st.subheader("Statistical Summary (Training)")
    desc1 = train.describe()
    desc1_df = desc1.reset_index()
    desc1_df.columns = ["Statistic", "Value"]
    st.table(desc1_df)

    # Downloads & observation
    st.subheader("Downloads")
    train_df1 = train.to_frame("price")
    fcast_df1 = mean1.to_frame("forecast")
    st.download_button("Download training series (CSV)", data=df_to_bytes(train_df1), file_name="project1_train_2010_2018.csv")
    st.download_button("Download forecast (CSV)", data=df_to_bytes(fcast_df1), file_name="project1_forecast_2019.csv")
    if min_len > 0:
        st.download_button("Download comparison (CSV)", data=df_to_bytes(comp_df), file_name="project1_comparison_2019.csv")

    st.subheader("Observation")
    st.write(
        "The ARIMA model was trained on monthly closes 2010–2018. Forecast for 2019 is produced and compared to actuals where available. Residual diagnostics and summary statistics are shown for model validation."
    )

# -------- Project 2 --------
with tab2:
    st.header("Project 2 — Train 2021–2025 → Forecast 2026")
    train2 = monthly.loc["2021-01-01":"2025-12-31"]
    actual_2026 = monthly.loc["2026-01-01":"2026-12-31"]

    st.subheader("1) Monthly Price Movement (Training)")
    if len(train2) == 0:
        st.info("No training data in selected window.")
    else:
        st.plotly_chart(plot_line(train2, "Monthly Close — 2021–2025"), use_container_width=True)

    try:
        fitted2, mean2, lower2, upper2 = fit_arima_and_forecast(train2, order=order, steps=forecast_months, forecast_start="2026-01-01")
    except Exception as e:
        st.error(f"ARIMA fit failed: {e}")
        st.stop()

    st.subheader("2) ARIMA Forecast Overlaid on Actual (if present)")
    st.plotly_chart(plot_forecast_overlay(train2, mean2, actual=actual_2026, lower=lower2, upper=upper2, title="Training + Forecast (2026)"), use_container_width=True)

    st.subheader("3) Forecast Only (2026 out-of-sample)")
    st.plotly_chart(plot_line(mean2, "Forecasted Monthly Prices — 2026"), use_container_width=True)

    # Backtest: train 2021-2024, test 2025
    st.subheader("Backtest: Train 2021–2024, Test 2025")
    back_train = monthly.loc["2021-01-01":"2024-12-31"]
    back_test = monthly.loc["2025-01-01":"2025-12-31"]
    if len(back_train) >= 6 and len(back_test) > 0:
        try:
            fitted_bt, mean_bt, lb_bt, ub_bt = fit_arima_and_forecast(back_train, order=order, steps=len(back_test), forecast_start=back_test.index[0])
            minlen = min(len(back_test), len(mean_bt))
            comp_bt = pd.DataFrame({"Actual": back_test.values[:minlen], "Forecast": mean_bt.values[:minlen]}, index=back_test.index[:minlen])
            st.dataframe(comp_bt.style.format("{:.2f}"))
            mae_bt, rmse_bt, mape_bt = compute_metrics(comp_bt["Actual"], comp_bt["Forecast"])
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE (backtest)", f"{mae_bt:.3f}")
            c2.metric("RMSE (backtest)", f"{rmse_bt:.3f}")
            c3.metric("MAPE (backtest)", f"{mape_bt:.2f}%")
        except Exception as e:
            st.info("Backtest model fit failed: " + str(e))
    else:
        st.info("Not enough data for backtest comparison.")

    # Residual diagnostics
    st.subheader("Residual Diagnostics (Training)")
    resid2 = to_1d_series(fitted2.resid)
    st.plotly_chart(plot_line(resid2, "Residuals (Training)"), use_container_width=True)
    fig_acf2, fig_pacf2 = acf_pacf_figs(train2, nlags=24)
    if fig_acf2 is not None:
        st.plotly_chart(fig_acf2, use_container_width=True)
    if fig_pacf2 is not None:
        st.plotly_chart(fig_pacf2, use_container_width=True)

    # Statistical summary
    st.subheader("Statistical Summary (Training)")
    desc2 = train2.describe()
    desc2_df = desc2.reset_index()
    desc2_df.columns = ["Statistic", "Value"]
    st.table(desc2_df)

    # Downloads & observation
    st.subheader("Downloads")
    train2_df = train2.to_frame("price")
    fcast2_df = mean2.to_frame("forecast")
    st.download_button("Download training series (CSV)", data=df_to_bytes(train2_df), file_name="project2_train_2021_2025.csv")
    st.download_button("Download forecast (CSV)", data=df_to_bytes(fcast2_df), file_name="project2_forecast_2026.csv")
    if 'comp_bt' in locals():
        st.download_button("Download backtest comparison (CSV)", data=df_to_bytes(comp_bt), file_name="project2_backtest_comparison_2025.csv")

    st.subheader("Observation")
    st.write(
        "Model trained on 2021–2025 monthly closes and used to produce a 2026 monthly forecast. A backtest provides a recent performance estimate when 2026 actuals are not fully present."
    )
