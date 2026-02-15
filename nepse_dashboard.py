import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import ta
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="NEPSE Index Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "/Users/shitalbhandary/Downloads/nepse_adb_data.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [col.strip().upper().replace(" ", "_") for col in df.columns]
    df["BUSINESS_DATE"] = pd.to_datetime(df["BUSINESS_DATE"])
    df = df.sort_values("BUSINESS_DATE").reset_index(drop=True)
    df["DAILY_RETURN"] = df["CLOSE_PRICE"].pct_change() * 100
    return df


def calculate_indicators(df):
    df = df.copy()
    df["SMA_20"] = ta.trend.SMAIndicator(df["CLOSE_PRICE"], window=20).sma_indicator()
    df["SMA_50"] = ta.trend.SMAIndicator(df["CLOSE_PRICE"], window=50).sma_indicator()
    df["EMA_20"] = ta.trend.EMAIndicator(df["CLOSE_PRICE"], window=20).ema_indicator()
    df["EMA_50"] = ta.trend.EMAIndicator(df["CLOSE_PRICE"], window=50).ema_indicator()
    df["RSI"] = ta.momentum.RSIIndicator(df["CLOSE_PRICE"], window=14).rsi()
    df["MACD"] = ta.trend.MACD(df["CLOSE_PRICE"]).macd()
    df["MACD_SIGNAL"] = ta.trend.MACD(df["CLOSE_PRICE"]).macd_signal()
    df["MACD_HIST"] = ta.trend.MACD(df["CLOSE_PRICE"]).macd_diff()
    df["BB_UPPER"] = ta.volatility.BollingerBands(
        df["CLOSE_PRICE"], window=20, window_dev=2
    ).bollinger_hband()
    df["BB_MIDDLE"] = ta.volatility.BollingerBands(
        df["CLOSE_PRICE"], window=20, window_dev=2
    ).bollinger_mavg()
    df["BB_LOWER"] = ta.volatility.BollingerBands(
        df["CLOSE_PRICE"], window=20, window_dev=2
    ).bollinger_lband()
    df["ATR"] = ta.volatility.AverageTrueRange(
        df["HIGH_PRICE"], df["LOW_PRICE"], df["CLOSE_PRICE"], window=14
    ).average_true_range()
    return df


def forecast_ets(train_data, forecast_days):
    try:
        model = ExponentialSmoothing(
            train_data, trend="add", seasonal=None, damped_trend=True
        )
        fitted = model.fit(optimized=True)
        forecast = fitted.forecast(forecast_days)
        return forecast, fitted
    except Exception as e:
        st.error(f"ETS Model Error: {e}")
        return None, None


def forecast_arima(train_data, forecast_days, order=(5, 1, 2)):
    try:
        model = ARIMA(train_data, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=forecast_days)
        return forecast, fitted
    except Exception as e:
        st.error(f"ARIMA Model Error: {e}")
        return None, None


def get_forecast_dates(last_date, days):
    dates = []
    current = last_date + timedelta(days=1)
    while len(dates) < days:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def apply_theme():
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

    if st.session_state.theme == "dark":
        st.markdown(
            """
        <style>
        .stApp { background-color: #0e1117; color: #fafafa; }
        .metric-card { background-color: #262730; padding: 15px; border-radius: 10px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; }
        .metric-label { font-size: 14px; color: #aaa; }
        </style>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <style>
        .stApp { background-color: #ffffff; color: #262730; }
        .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; }
        .metric-label { font-size: 14px; color: #555; }
        </style>
        """,
            unsafe_allow_html=True,
        )


def main():
    apply_theme()

    df = load_data()
    df = calculate_indicators(df)

    min_date = df["BUSINESS_DATE"].min().date()
    max_date = df["BUSINESS_DATE"].max().date()

    st.title("📈 NEPSE Index Dashboard")
    st.markdown(
        f"**Data Period:** {min_date} to {max_date} | **Trading Days:** {len(df)}"
    )

    with st.sidebar:
        st.header("⚙️ Settings")

        st.session_state.theme = st.radio(
            "Theme",
            ["dark", "light"],
            index=0 if st.session_state.theme == "dark" else 1,
        )

        st.subheader("📅 Date Range")
        start_date = st.date_input(
            "Start Date", min_date, min_value=min_date, max_value=max_date
        )
        end_date = st.date_input(
            "End Date", max_date, min_value=min_date, max_value=max_date
        )

        st.subheader("📊 Chart Type")
        chart_type = st.selectbox("Select Chart", ["Candlestick", "Line"])

        st.subheader("📉 Technical Indicators")
        show_sma20 = st.checkbox("SMA 20", value=True)
        show_sma50 = st.checkbox("SMA 50", value=False)
        show_ema20 = st.checkbox("EMA 20", value=False)
        show_ema50 = st.checkbox("EMA 50", value=False)
        show_bb = st.checkbox("Bollinger Bands", value=False)
        show_rsi = st.checkbox("RSI", value=True)
        show_macd = st.checkbox("MACD", value=True)
        show_atr = st.checkbox("ATR", value=False)

        st.subheader("🔮 Forecasting")
        enable_forecast = st.checkbox("Enable Forecast", value=False)
        forecast_days = st.slider("Forecast Days", min_value=1, max_value=30, value=7)
        model_type = st.selectbox("Model", ["ETS (Holt-Winters)", "ARIMA", "Both"])
        if model_type == "ARIMA":
            arima_p = st.slider("ARIMA p (autoregressive)", 0, 5, 5)
            arima_d = st.slider("ARIMA d (differencing)", 0, 2, 1)
            arima_q = st.slider("ARIMA q (moving average)", 0, 5, 2)
            arima_order = (arima_p, arima_d, arima_q)
        else:
            arima_order = (5, 1, 2)

    mask = (df["BUSINESS_DATE"].dt.date >= start_date) & (
        df["BUSINESS_DATE"].dt.date <= end_date
    )
    df_filtered = df[mask].copy()

    if len(df_filtered) == 0:
        st.error("No data available for selected date range!")
        return

    latest = df_filtered.iloc[-1]
    prev = df_filtered.iloc[-2] if len(df_filtered) > 1 else latest
    price_change = latest["CLOSE_PRICE"] - prev["CLOSE_PRICE"]
    price_change_pct = (price_change / prev["CLOSE_PRICE"]) * 100

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Close Price",
            f"₹{latest['CLOSE_PRICE']:.2f}",
            f"{price_change:+.2f} ({price_change_pct:+.2f}%)",
        )
    with col2:
        st.metric("High Price", f"₹{latest['HIGH_PRICE']:.2f}")
    with col3:
        st.metric("Low Price", f"₹{latest['LOW_PRICE']:.2f}")
    with col4:
        st.metric("Volume", f"{latest['TOTAL_TRADED_QUANTITY']:,.0f}")
    with col5:
        st.metric("Trades", f"{latest['TOTAL_TRADES']:,.0f}")

    st.divider()

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=("Price Chart", "RSI", "MACD", "Volume"),
    )

    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df_filtered["BUSINESS_DATE"],
                open=df_filtered["CLOSE_PRICE"]
                - (df_filtered["HIGH_PRICE"] - df_filtered["LOW_PRICE"]) / 2,
                high=df_filtered["HIGH_PRICE"],
                low=df_filtered["LOW_PRICE"],
                close=df_filtered["CLOSE_PRICE"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df_filtered["BUSINESS_DATE"],
                y=df_filtered["CLOSE_PRICE"],
                mode="lines",
                name="Close Price",
                line=dict(color="#00cc96", width=2),
            ),
            row=1,
            col=1,
        )

    if show_sma20 and not df_filtered["SMA_20"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df_filtered["BUSINESS_DATE"],
                y=df_filtered["SMA_20"],
                mode="lines",
                name="SMA 20",
                line=dict(color="#ffa15a", width=1.5),
            ),
            row=1,
            col=1,
        )

    if show_sma50 and not df_filtered["SMA_50"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df_filtered["BUSINESS_DATE"],
                y=df_filtered["SMA_50"],
                mode="lines",
                name="SMA 50",
                line=dict(color="#ff6692", width=1.5),
            ),
            row=1,
            col=1,
        )

    if show_ema20 and not df_filtered["EMA_20"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df_filtered["BUSINESS_DATE"],
                y=df_filtered["EMA_20"],
                mode="lines",
                name="EMA 20",
                line=dict(color="#19d3f3", width=1.5),
            ),
            row=1,
            col=1,
        )

    if show_ema50 and not df_filtered["EMA_50"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df_filtered["BUSINESS_DATE"],
                y=df_filtered["EMA_50"],
                mode="lines",
                name="EMA 50",
                line=dict(color="#9467bd", width=1.5),
            ),
            row=1,
            col=1,
        )

    if show_bb and not df_filtered["BB_UPPER"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df_filtered["BUSINESS_DATE"],
                y=df_filtered["BB_UPPER"],
                mode="lines",
                name="BB Upper",
                line=dict(color="#7f7f7f", width=1, dash="dash"),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_filtered["BUSINESS_DATE"],
                y=df_filtered["BB_LOWER"],
                mode="lines",
                name="BB Lower",
                line=dict(color="#7f7f7f", width=1, dash="dash"),
                fill="tonexty",
                fillcolor="rgba(127, 127, 127, 0.1)",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    if show_rsi:
        fig.add_trace(
            go.Scatter(
                x=df_filtered["BUSINESS_DATE"],
                y=df_filtered["RSI"],
                mode="lines",
                name="RSI",
                line=dict(color="#ab63fa", width=1.5),
            ),
            row=2,
            col=1,
        )
        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color="red",
            row=2,
            col=1,
            annotation_text="Overbought",
        )
        fig.add_hline(
            y=30,
            line_dash="dash",
            line_color="green",
            row=2,
            col=1,
            annotation_text="Oversold",
        )
        fig.update_yaxes(range=[0, 100], row=2, col=1)

    if show_macd:
        colors = [
            "#00cc96" if v >= 0 else "#ef553b"
            for v in df_filtered["MACD_HIST"].fillna(0)
        ]
        fig.add_trace(
            go.Bar(
                x=df_filtered["BUSINESS_DATE"],
                y=df_filtered["MACD_HIST"],
                name="MACD Hist",
                marker_color=colors,
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_filtered["BUSINESS_DATE"],
                y=df_filtered["MACD"],
                mode="lines",
                name="MACD",
                line=dict(color="#00cc96", width=1.5),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_filtered["BUSINESS_DATE"],
                y=df_filtered["MACD_SIGNAL"],
                mode="lines",
                name="Signal",
                line=dict(color="#ffa15a", width=1.5),
            ),
            row=3,
            col=1,
        )

    colors_vol = [
        "#00cc96"
        if i > 0
        and df_filtered.iloc[i]["CLOSE_PRICE"] >= df_filtered.iloc[i - 1]["CLOSE_PRICE"]
        else "#ef553b"
        for i in range(len(df_filtered))
    ]
    fig.add_trace(
        go.Bar(
            x=df_filtered["BUSINESS_DATE"],
            y=df_filtered["TOTAL_TRADED_QUANTITY"],
            name="Volume",
            marker_color="#636efa",
        ),
        row=4,
        col=1,
    )

    template = "plotly_dark" if st.session_state.theme == "dark" else "plotly"
    fig.update_layout(
        template=template,
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    if enable_forecast:
        st.divider()
        st.subheader("🔮 Price Forecast")

        train_data = df_filtered["CLOSE_PRICE"].values
        last_date = df_filtered["BUSINESS_DATE"].iloc[-1]

        forecast_fig = go.Figure()

        forecast_fig.add_trace(
            go.Scatter(
                x=df_filtered["BUSINESS_DATE"],
                y=df_filtered["CLOSE_PRICE"],
                mode="lines",
                name="Historical",
                line=dict(color="#00cc96", width=2),
            )
        )

        forecast_col1, forecast_col2 = st.columns(2)

        with forecast_col1:
            if model_type in ["ETS (Holt-Winters)", "Both"]:
                with st.spinner("Running ETS Model..."):
                    ets_forecast, ets_fitted = forecast_ets(train_data, forecast_days)
                    if ets_forecast is not None:
                        forecast_dates = get_forecast_dates(last_date, forecast_days)
                        ets_vals = (
                            ets_forecast.values
                            if hasattr(ets_forecast, "values")
                            else ets_forecast
                        )
                        st.success(
                            f"ETS Forecast: ₹{ets_vals[-1]:.2f} (Day {forecast_days})"
                        )

                        forecast_fig.add_trace(
                            go.Scatter(
                                x=forecast_dates,
                                y=ets_vals,
                                mode="lines+markers",
                                name="ETS Forecast",
                                line=dict(color="#ff6692", width=2, dash="dot"),
                            )
                        )

                        with st.expander("ETS Model Details"):
                            st.write(
                                f"**Smoothing Level (α):** {ets_fitted.params.get('smoothing_level', 'N/A'):.4f}"
                            )
                            st.write(
                                f"**Smoothing Trend (β):** {ets_fitted.params.get('smoothing_trend', 'N/A'):.4f}"
                            )
                            st.write(
                                f"**Damping Trend (φ):** {ets_fitted.params.get('damping_trend', 'N/A'):.4f}"
                            )
                            st.markdown("---")
                            st.markdown("### ETS (Holt-Winters) Equation")
                            st.latex(
                                r"\hat{y}_{t+h|t} = (l_t + \phi b_t) + s_{t-m+h_m^+}"
                            )
                            st.markdown("""
                            **Where:**
                            - **$\\hat{y}_{t+h|t}$**: Forecasted value at horizon h
                            - **$l_t$**: Level (smoothed value at time t)
                            - **$b_t$**: Trend component
                            - **$\\phi$**: Damping factor (reduces trend over forecast horizon)
                            - **$s_t$**: Seasonal component
                            - **$m$**: Seasonal period (not used here - additive trend only)
                            - **$h$**: Forecast horizon (days ahead)
                            """)
                            st.markdown("---")
                            st.markdown("### Why Holt-Winters ETS?")
                            st.info("""
                            **Holt-Winters (Damped Trend ETS)** is chosen because:
                            
                            1. **Exponential Smoothing**: Gives more weight to recent observations, making it responsive to recent price changes
                            2. **Damped Trend**: The damping factor ($\\phi$) prevents the forecast from projecting unbounded trends, which is crucial for stock prices that can't grow indefinitely
                            3. **No Seasonality**: Since stock market data lacks clear short-term seasonal patterns, we use additive trend without seasonal components
                            4. **Simple & Robust**: Fewer parameters to estimate, reducing overfitting risk for short-term forecasts
                            """)

        with forecast_col2:
            if model_type in ["ARIMA", "Both"]:
                with st.spinner("Running ARIMA Model..."):
                    arima_forecast, arima_fitted = forecast_arima(
                        train_data, forecast_days, arima_order
                    )
                    if arima_forecast is not None:
                        forecast_dates = get_forecast_dates(last_date, forecast_days)
                        arima_vals = (
                            arima_forecast.values
                            if hasattr(arima_forecast, "values")
                            else arima_forecast
                        )
                        st.success(
                            f"ARIMA Forecast: ₹{arima_vals[-1]:.2f} (Day {forecast_days})"
                        )

                        forecast_fig.add_trace(
                            go.Scatter(
                                x=forecast_dates,
                                y=arima_vals,
                                mode="lines+markers",
                                name="ARIMA Forecast",
                                line=dict(color="#ffa15a", width=2, dash="dot"),
                            )
                        )

                        with st.expander("ARIMA Model Details"):
                            st.write(f"**Order (p,d,q):** {arima_order}")
                            st.write(f"**AIC:** {arima_fitted.aic:.2f}")
                            st.write(f"**BIC:** {arima_fitted.bic:.2f}")
                            st.markdown("---")
                            st.markdown("### ARIMA Equation")
                            st.latex(
                                r"(1 - \sum_{i=1}^{p} \phi_i L^i)(1-L)^d y_t = (1 + \sum_{i=1}^{q} \theta_i L^i) \epsilon_t"
                            )
                            st.markdown(f"""
                            **Where:**
                            - **AR({arima_order[0]})**: Autoregressive part - uses {arima_order[0]} past values
                            - **I({arima_order[1]})**: Integrated (differencing) - applied {arima_order[1]} time(s) to make series stationary
                            - **MA({arima_order[2]})**: Moving Average part - uses {arima_order[2]} past forecast errors
                            - **$y_t$**: Actual value at time t
                            - **$\\epsilon_t$**: White noise error term at time t
                            - **$L$**: Lag operator (L$y_t$ = $y_{{t-1}}$)
                            - **$\\phi_i$**: AR coefficients (autoregressive weights)
                            - **$\\theta_i$**: MA coefficients (moving average weights)
                            """)
                            st.markdown("---")
                            st.markdown("### Why ARIMA(5,1,2)?")
                            st.info(f"""
                            **Default order (5,1,2)** is chosen based on:
                            
                            1. **d=1 (Differencing)**: Stock prices are typically non-stationary. First-order differencing removes the stochastic trend and makes the series stationary for modeling.
                            
                            2. **p=5 (AR terms)**: Captures autocorrelations with up to 5-day lags. NEPSE index shows significant autocorrelation at multiple lag periods.
                            
                            3. **q=2 (MA terms)**: Uses 2 lagged forecast errors to account for short-term shocks in the market.
                            
                            4. **Common Practice**: (5,1,2) is a widely used default that balances complexity and performance for financial time series.
                            
                            5. **AIC/BIC Optimization**: You can adjust p, d, q values in the sidebar - lower AIC/BIC indicates better model fit.
                            """)

        forecast_fig.update_layout(
            template=template,
            title="Price Forecast - Next 7 Days",
            xaxis_title="Date",
            yaxis_title="Price (NPR)",
            showlegend=True,
            hovermode="x unified",
        )

        st.plotly_chart(forecast_fig, use_container_width=True)

        def to_array(val):
            if val is None:
                return [None] * forecast_days
            return val.values if hasattr(val, "values") else val

        ets_vals = (
            to_array(ets_forecast)
            if model_type in ["ETS (Holt-Winters)", "Both"]
            else [None] * forecast_days
        )
        arima_vals = (
            to_array(arima_forecast)
            if model_type in ["ARIMA", "Both"]
            else [None] * forecast_days
        )

        forecast_df = pd.DataFrame(
            {
                "Date": get_forecast_dates(last_date, forecast_days),
                "ETS Forecast": ets_vals,
                "ARIMA Forecast": arima_vals,
            }
        )

        st.dataframe(forecast_df, use_container_width=True)

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Statistics Summary")

        stats_data = {
            "Metric": [
                "Mean Price",
                "Max Price",
                "Min Price",
                "Std Deviation",
                "Volatility (%)",
                "Avg Volume",
                "Avg Trades",
                "Total Value (NPR)",
            ],
            "Value": [
                f"₹{df_filtered['CLOSE_PRICE'].mean():.2f}",
                f"₹{df_filtered['CLOSE_PRICE'].max():.2f}",
                f"₹{df_filtered['CLOSE_PRICE'].min():.2f}",
                f"₹{df_filtered['CLOSE_PRICE'].std():.2f}",
                f"{df_filtered['DAILY_RETURN'].std():.2f}%",
                f"{df_filtered['TOTAL_TRADED_QUANTITY'].mean():,.0f}",
                f"{df_filtered['TOTAL_TRADES'].mean():.0f}",
                f"₹{df_filtered['TOTAL_TRADED_VALUE'].sum():,.0f}",
            ],
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("📥 Export Data")

        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="nepse_filtered_data.csv",
            mime="text/csv",
        )

    with st.expander("📈 Daily Returns Distribution"):
        returns_fig = go.Figure()
        returns_fig.add_trace(
            go.Histogram(
                x=df_filtered["DAILY_RETURN"].dropna(),
                nbinsx=30,
                marker_color="#00cc96",
                name="Daily Returns",
            )
        )
        returns_fig.update_layout(
            template=template,
            title="Daily Returns Distribution",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
        )
        st.plotly_chart(returns_fig, use_container_width=True)

    with st.expander("📋 Raw Data Preview"):
        st.dataframe(df_filtered.head(50), use_container_width=True)

    st.markdown("---")
    st.markdown("**NEPSE Index Dashboard** | Built with Streamlit & Plotly")


if __name__ == "__main__":
    main()
