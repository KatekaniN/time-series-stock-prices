import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="AI Stock Forecast Intelligence",
    layout="wide",
    page_icon="https://cdn.jsdelivr.net/npm/@tabler/icons@2.47.0/icons/line-chart.svg",
)

# Icon utilities
ICONS = {
    "market": "https://cdn.jsdelivr.net/npm/@tabler/icons@2.47.0/icons/currency-dollar.svg",
    "history": "https://cdn.jsdelivr.net/npm/@tabler/icons@2.47.0/icons/chart-line.svg",
    "forecast": "https://cdn.jsdelivr.net/npm/@tabler/icons@2.47.0/icons/chart-dots-2.svg",
    "insights": "https://cdn.jsdelivr.net/npm/@tabler/icons@2.47.0/icons/bulb.svg",
    "patterns": "https://cdn.jsdelivr.net/npm/@tabler/icons@2.47.0/icons/chart-arrows-vertical.svg",
    "help": "https://cdn.jsdelivr.net/npm/@tabler/icons@2.47.0/icons/help-circle.svg",
    "table": "https://cdn.jsdelivr.net/npm/@tabler/icons@2.47.0/icons/table.svg",
}

def h2(title: str, icon_key: str):
    icon = ICONS.get(icon_key)
    st.markdown(
        f"<h3 style='margin:0 0 0.5rem 0; display:flex; align-items:center;'>"
        f"<img src='{icon}' width='20' height='20' style='margin-right:8px; opacity:0.9'/>"
        f"{title}</h3>",
        unsafe_allow_html=True,
    )

# Company profiles with branding and business information
COMPANY_PROFILES = {
    "Apple": {
        "ticker": "AAPL",
        "color": "#000000",
        "bg_color": "#F5F5F7",
        "text_color": "#1d1d1f",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg",
        "tagline": "Think Different",
        "business": """**Apple Inc.** is a technology giant that revolutionized personal computing, mobile devices, and digital services. 
        
**Revenue Streams:**
- **iPhone Sales** (52% of revenue) - Premium smartphones with ecosystem lock-in
- **Services** (20%) - App Store, iCloud, Apple Music, Apple TV+, AppleCare
- **Mac & iPad** (18%) - Personal computers and tablets
- **Wearables** (10%) - Apple Watch, AirPods, accessories

**Investment Thesis:** Strong brand loyalty, ecosystem stickiness, growing services revenue with high margins, massive cash reserves."""
    },
    "Microsoft": {
        "ticker": "MSFT",
        "color": "#00A4EF",
        "bg_color": "#F3F2F1",
        "text_color": "#ffffff",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg",
        "tagline": "Empowering Every Person and Organization",
        "business": """**Microsoft Corporation** is the world's leading enterprise software and cloud computing company.

**Revenue Streams:**
- **Intelligent Cloud** (40%) - Azure, SQL Server, Windows Server, enterprise services
- **Productivity & Business** (32%) - Office 365, LinkedIn, Dynamics 365
- **Personal Computing** (28%) - Windows OS, Xbox, Surface devices, search advertising

**Investment Thesis:** Azure growth driving cloud transition, AI leadership with OpenAI partnership, recurring subscription revenue model."""
    },
    "Google": {
        "ticker": "GOOGL",
        "color": "#4285F4",
        "bg_color": "#F8F9FA",
        "text_color": "#ffffff",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/2/2f/Google_2015_logo.svg",
        "tagline": "Organize the World's Information",
        "business": """**Alphabet Inc. (Google)** dominates digital advertising and leads in AI research and development.

**Revenue Streams:**
- **Google Search & Ads** (58%) - Search advertising, YouTube ads
- **Google Cloud** (10%) - Cloud infrastructure, workspace tools
- **Google Other** (11%) - Play Store, hardware, subscriptions
- **Other Bets** (1%) - Waymo, Verily, experimental ventures

**Investment Thesis:** Unmatched search dominance, YouTube's growth, cloud expansion, AI leadership with Gemini."""
    },
    "Amazon": {
        "ticker": "AMZN",
        "color": "#FF9900",
        "bg_color": "#ffffff",
        "text_color": "#ffffff",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg",
        "tagline": "Earth's Most Customer-Centric Company",
        "business": """**Amazon.com Inc.** transformed from online bookstore to global e-commerce and cloud computing leader.

**Revenue Streams:**
- **E-commerce** (50%) - Online retail marketplace, Prime subscriptions
- **Amazon Web Services** (16%) - Cloud computing infrastructure (60% of operating profit)
- **Third-party Seller Services** (22%) - Marketplace fees, fulfillment services
- **Advertising** (8%) - Sponsored products, display ads
- **Physical Stores** (4%) - Whole Foods, Amazon Go

**Investment Thesis:** AWS dominance in cloud, e-commerce moat, advertising growth, logistics network advantages."""
    },
    "Tesla": {
        "ticker": "TSLA",
        "color": "#E82127",
        "bg_color": "#000000",
        "text_color": "#ffffff",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/b/bd/Tesla_Motors.svg",
        "tagline": "Accelerating the World's Transition to Sustainable Energy",
        "business": """**Tesla Inc.** is the world's leading electric vehicle manufacturer and clean energy company.

**Revenue Streams:**
- **Automotive Sales** (81%) - Model 3, Y, S, X electric vehicles
- **Energy Generation & Storage** (6%) - Solar panels, Powerwall, Megapack
- **Services** (7%) - Supercharger network, vehicle servicing, insurance
- **Regulatory Credits** (6%) - Emissions credits sold to other automakers

**Investment Thesis:** EV market leadership, vertical integration, autonomous driving technology, energy storage potential."""
    },
    "NVIDIA": {
        "ticker": "NVDA",
        "color": "#76B900",
        "bg_color": "#ffffff",
        "text_color": "#ffffff",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/2/21/Nvidia_logo.svg",
        "tagline": "The AI Computing Company",
        "business": """**NVIDIA Corporation** dominates AI computing chips and graphics processing technology.

**Revenue Streams:**
- **Data Center** (75%) - AI chips (A100, H100), enterprise computing
- **Gaming** (17%) - GeForce GPUs, gaming platforms
- **Professional Visualization** (5%) - Workstation GPUs, design software
- **Automotive** (3%) - Self-driving car chips, in-vehicle computing

**Investment Thesis:** AI revolution driving massive GPU demand, data center dominance, CUDA software moat, gaming resilience."""
    },
    "Meta": {
        "ticker": "META",
        "color": "#0668E1",
        "bg_color": "#F0F2F5",
        "text_color": "#ffffff",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/7/7b/Meta_Platforms_Inc._logo.svg",
        "tagline": "Connecting the World",
        "business": """**Meta Platforms Inc. (formerly Facebook)** is the world's largest social media company building the metaverse.

**Revenue Streams:**
- **Advertising** (97%) - Facebook, Instagram, Messenger, WhatsApp ads
- **Reality Labs** (2%) - VR headsets (Quest), AR glasses, metaverse platforms
- **Other Revenue** (1%) - Payments, business tools

**Investment Thesis:** Massive user base (3.2B daily active users), Instagram growth, WhatsApp monetization potential, AI-driven ad targeting."""
    },
    "Netflix": {
        "ticker": "NFLX",
        "color": "#E50914",
        "bg_color": "#000000",
        "text_color": "#ffffff",
        "logo_url": "https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg",
        "tagline": "Entertainment for Everyone",
        "business": """**Netflix Inc.** pioneered streaming entertainment and leads in original content production.

**Revenue Streams:**
- **Subscription Revenue** (100%) - Streaming memberships across tiers
  - Standard with Ads
  - Standard (HD)
  - Premium (4K)
- **260+ million subscribers globally**
- **Original content spend**: ~$17B annually

**Investment Thesis:** Global streaming leader, content library moat, pricing power, ad-tier growth, international expansion."""
    }
}

# Sidebar configuration
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("---")

# Stock selection
selected_stock_name = st.sidebar.selectbox(
    "Select Company", 
    list(COMPANY_PROFILES.keys()),
    help="Choose a company to analyze and forecast"
)

profile = COMPANY_PROFILES[selected_stock_name]
stock_ticker = profile["ticker"]

# Date range
st.sidebar.markdown("---")
st.sidebar.subheader("Time Period Settings")
years_back = st.sidebar.slider(
    "Historical Data (years)", 
    1, 5, 2,
    help="Amount of historical data to analyze"
)
forecast_days = st.sidebar.slider(
    "Forecast Horizon (days)", 
    30, 365, 90,
    help="How far into the future to predict"
)

# Model and chart options
st.sidebar.markdown("---")
st.sidebar.subheader("Model Settings")
st.sidebar.caption(
    "Seasonality controls repeating patterns (weekly/yearly). Trend flexibility sets how quickly the model reacts to changes."
)
seasonality_mode = st.sidebar.selectbox(
    "Seasonality Mode",
    ["additive", "multiplicative"],
    index=0,
    help=(
        "How seasonal ups/downs are handled: \n"
        "• Additive: seasonal swings are a fixed amount. \n"
        "• Multiplicative: swings scale with the price (good when moves are more like percentages)."
    )
)
changepoint_prior = st.sidebar.slider(
    "Trend Flexibility (changepoint)",
    min_value=0.01, max_value=0.5, value=0.05, step=0.01,
    help=(
        "How quickly the model adapts to new trends. \n"
        "• Lower = smoother, less reactive. \n"
        "• Higher = more responsive to changes."
    )
)

st.sidebar.subheader("Chart Options")
st.sidebar.caption(
    "SMA = Simple Moving Average (smooths prices). Log scale makes equal % moves look equal on the chart."
)
use_log_scale = st.sidebar.checkbox(
    "Log scale (price)", value=False,
    help="Shows equal percentage moves as equal distances. Helpful when price spans a wide range."
)
show_sma20 = st.sidebar.checkbox(
    "Show SMA 20", value=True,
    help="SMA (Simple Moving Average): the average closing price over the last 20 days — smooths short-term noise."
)
show_sma50 = st.sidebar.checkbox(
    "Show SMA 50", value=True,
    help="SMA (Simple Moving Average): the average closing price over the last 50 days — medium‑term trend."
)

with st.sidebar.expander("Company Snapshot", expanded=False):
    try:
        tk_side = yf.Ticker(stock_ticker)
        fast = getattr(tk_side, "fast_info", {}) or {}
        def _fmt_num(x):
            try:
                x = float(x)
            except Exception:
                return "-"
            for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6)]:
                if abs(x) >= div:
                    return f"{x/div:.2f}{unit}"
            return f"{x:.0f}"
        last_price = fast.get("last_price")
        mcap = fast.get("market_cap")
        year_high = fast.get("year_high")
        year_low = fast.get("year_low")
        dividend_yield = fast.get("dividend_yield")

        cols = st.columns(2)
        with cols[0]:
            st.metric("Last Price", f"${last_price:.2f}" if last_price else "-")
            st.metric("Market Cap", _fmt_num(mcap))
        with cols[1]:
            st.metric("52W High", f"${year_high:.2f}" if year_high else "-")
            st.metric("52W Low", f"${year_low:.2f}" if year_low else "-")
        if dividend_yield:
            st.caption(f"Dividend Yield: {dividend_yield*100:.2f}%")
    except Exception:
        st.caption("Snapshot unavailable for this ticker.")

# Load data
@st.cache_data
def load_data(ticker, years):
    """Download stock data from Yahoo Finance using reliable period-based API"""
    try:
        period = f"{max(1, int(years))}y"
    except Exception:
        period = "2y"

    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval="1d", auto_adjust=True)

    # If history returns empty, fall back to download
    if df is None or len(df) == 0:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=max(365, years * 365))
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df is None or len(df) == 0:
        return None

    # Normalize index/columns
    df = df.reset_index()
    if 'Date' not in df.columns:
        # yfinance may use 'Datetime'
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
        else:
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    # Ensure types and cleanliness
    # Ensure timezone-naive datetime (Prophet requires no timezone)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True).dt.tz_convert(None)
    df = df.sort_values('Date')
    df = df.dropna(subset=['Close'])

    # Compute simple moving averages for context
    if len(df) >= 20:
        df['SMA20'] = df['Close'].rolling(20).mean()
    else:
        df['SMA20'] = pd.NA
    if len(df) >= 50:
        df['SMA50'] = df['Close'].rolling(50).mean()
    else:
        df['SMA50'] = pd.NA

    return df

# Show loading state
with st.spinner(f"Loading {selected_stock_name} market data..."):
    data = load_data(stock_ticker, years_back)

# Check if data was loaded successfully
if data is None or len(data) == 0:
    st.error("Failed to load data. Please try again or select a different stock.")
    st.stop()

# Company Header with branding
st.markdown(f"""
<div style='background-color: {profile['bg_color']}; 
            padding: 2.5rem; 
            border-radius: 12px; 
            margin-bottom: 2rem; 
            border-left: 6px solid {profile['color']};
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
    <div style='display: flex; align-items: center; gap: 2rem;'>
        <img src='{profile['logo_url']}' style='height: 60px; width: auto;' alt='{selected_stock_name} logo'/>
        <div>
            <h1 style='margin: 0; font-size: 2.5rem; color: {profile['color']};'>{selected_stock_name} Stock Intelligence</h1>
            <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem; color: {profile['text_color'] if profile['bg_color'] == '#000000' else '#666'};'>{profile['tagline']}</p>
            <p style='margin: 0.3rem 0 0 0; font-size: 0.9rem; color: {profile['text_color'] if profile['bg_color'] == '#000000' else '#999'}; font-weight: 600;'>NYSE: {stock_ticker}</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Company Business Overview
with st.expander("About the Business – How This Company Makes Money", expanded=True):
    st.markdown(profile['business'])
    
st.markdown("---")

# Market Performance Metrics
h2("Current Market Performance", "market")
st.caption(f"Latest stock data as of {data['Date'].max().strftime('%B %d, %Y')}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Current Price", f"${float(data['Close'].iloc[-1]):.2f}")
with col2:
    st.metric(f"{years_back}-Year High", f"${float(data['Close'].max()):.2f}")
with col3:
    st.metric(f"{years_back}-Year Low", f"${float(data['Close'].min()):.2f}")
with col4:
    price_change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
    st.metric("Period Change", f"{float(price_change):.2f}%", delta=f"{float(price_change):.2f}%")

st.markdown("---")

# Historical Price + Volume Chart
h2("Historical Price Movement", "history")
st.caption(f"Stock performance from {data['Date'].min().strftime('%B %Y')} to {data['Date'].max().strftime('%B %Y')}")
st.caption("Tip: Toggle SMA 20/50 to see short and medium-term trends. Turn on Log scale for better readability across large price ranges.")

def hex_to_rgba(hex_color, alpha=0.2):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'

fig_hist = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         row_heights=[0.75, 0.25])

# Price line with markers in brand color
fig_hist.add_trace(
    go.Scatter(
        x=data['Date'], y=data['Close'], name='Close',
        mode='lines', line=dict(color=profile['color'], width=2.5),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Close: $%{y:.2f}<extra></extra>'
    ), row=1, col=1
)

# Moving averages (toggle via sidebar)
if show_sma20 and 'SMA20' in data.columns and data['SMA20'].notna().any():
    fig_hist.add_trace(
        go.Scatter(x=data['Date'], y=data['SMA20'], name='SMA 20',
                   line=dict(color='#888', width=1.5, dash='dash')),
        row=1, col=1
    )
if show_sma50 and 'SMA50' in data.columns and data['SMA50'].notna().any():
    fig_hist.add_trace(
        go.Scatter(x=data['Date'], y=data['SMA50'], name='SMA 50',
                   line=dict(color='#BBB', width=1.5, dash='dot')),
        row=1, col=1
    )

# Volume bars in semi-transparent brand color
if 'Volume' in data.columns:
    fig_hist.add_trace(
        go.Bar(x=data['Date'], y=data['Volume'], name='Volume',
               marker_color=hex_to_rgba(profile['color'], 0.35)),
        row=2, col=1
    )

fig_hist.update_layout(
    height=520,
    hovermode='x unified',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    template='plotly_white',
)
fig_hist.update_yaxes(title_text='Price (USD)', tickprefix='$', row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
fig_hist.update_yaxes(title_text='Volume', row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
fig_hist.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

# Range controls and log scale
fig_hist.update_xaxes(
    rangeselector=dict(
        buttons=[
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=3, label="3M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all")
        ]
    ),
    rangeslider=dict(visible=True),
    row=2, col=1
)
if use_log_scale:
    fig_hist.update_yaxes(type='log', row=1, col=1)

st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# Prepare data for Prophet
df_train = data[['Date', 'Close']].copy()
df_train.columns = ['ds', 'y']

# Prophet requires timezone-naive datetimes in 'ds'
df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce')
if getattr(df_train['ds'].dtype, 'tz', None) is not None:
    df_train['ds'] = df_train['ds'].dt.tz_localize(None)

# Train Prophet model (with minimal UI)
with st.spinner("Training AI forecasting model..."):
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=changepoint_prior,
        seasonality_mode=seasonality_mode
    )
    model.fit(df_train)

# Make predictions
future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

st.markdown("---")

# Display forecast
h2(f"{forecast_days}-Day Price Forecast", "forecast")
st.caption("AI prediction with confidence intervals - lighter shading shows uncertainty range")

# Plot forecast
fig_forecast = plot_plotly(model, forecast)
fig_forecast.update_traces(
    line=dict(color=profile['color'], width=3),
    selector=dict(name='Actual')
)
fig_forecast.update_layout(
    height=550,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    template='plotly_white',
)
st.plotly_chart(fig_forecast, use_container_width=True)

# Simple backtest on recent history
with st.expander("Model Backtest (last 60 days)", expanded=False):
    try:
        st.markdown(
            """
            This quick check compares the model's recent predictions to actual prices:
            - MAE: Average absolute error in dollars (lower is better)
            - MAPE: Average percentage error (lower is better)
            """
        )
        test_days = 60 if len(df_train) > 180 else max(30, min(60, len(df_train)//5))
        if len(df_train) > test_days + 30:
            df_train_bt = df_train.iloc[:-test_days]
            model_bt = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=changepoint_prior,
                seasonality_mode=seasonality_mode
            )
            model_bt.fit(df_train_bt)
            future_bt = model_bt.make_future_dataframe(periods=test_days)
            forecast_bt = model_bt.predict(future_bt)
            # Align with actuals
            actual_bt = df_train.tail(test_days).set_index('ds')
            pred_bt = forecast_bt.tail(test_days).set_index('ds')
            comp = actual_bt[['y']].join(pred_bt[['yhat']], how='inner')
            mae = float(np.mean(np.abs(comp['y'] - comp['yhat'])))
            mape = float(np.mean(np.abs((comp['y'] - comp['yhat']) / comp['y'])) * 100)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Backtest MAE (USD)", f"${mae:.2f}")
            with c2:
                st.metric("Backtest MAPE", f"{mape:.2f}%")

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=comp.index, y=comp['y'], name='Actual',
                                        line=dict(color=profile['color'], width=2)))
            fig_bt.add_trace(go.Scatter(x=comp.index, y=comp['yhat'], name='Predicted',
                                        line=dict(color='#555', width=2, dash='dot')))
            fig_bt.update_layout(height=300, template='plotly_white',
                                 plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bt, use_container_width=True)
        else:
            st.caption("Not enough data for a meaningful backtest.")
    except Exception as e:
        st.caption("Backtest unavailable: " + str(e))

# Investment Insights & Recommendations
st.markdown("---")
h2("AI-Generated Investment Insights", "insights")

forecast_future = forecast.tail(forecast_days)
current_price = float(data['Close'].iloc[-1])
predicted_price = float(forecast_future['yhat'].iloc[-1])
predicted_change_pct = ((predicted_price - current_price) / current_price) * 100

# Calculate trend strength
recent_trend = forecast_future['trend'].iloc[-1] - forecast_future['trend'].iloc[0]
volatility = forecast_future['yhat_upper'].iloc[-1] - forecast_future['yhat_lower'].iloc[-1]

col1, col2 = st.columns([2, 1])

with col1:
    # Build a status badge without emojis
    if predicted_change_pct > 2:
        badge_color = "#16A34A"  # green
        badge_text = "BULLISH"
    elif predicted_change_pct < -2:
        badge_color = "#DC2626"  # red
        badge_text = "BEARISH"
    else:
        badge_color = "#D97706"  # amber
        badge_text = "NEUTRAL"

    badge_html = f"<span style='display:inline-block;padding:2px 8px;border-radius:9999px;background:{badge_color};color:#fff;font-weight:600;font-size:0.85rem;'>{badge_text}</span>"

    st.markdown(
        f"""
    ### Forecast Summary

    **Current Price:** ${current_price:.2f}  
    **Predicted Price ({forecast_days} days):** ${predicted_price:.2f}  
    **Expected Change:** {predicted_change_pct:+.2f}%  
    **Market Signal:** {badge_html}
    """,
        unsafe_allow_html=True,
    )
    
    # Generate insights based on forecast
    if predicted_change_pct > 5:
        signal = "**Strong Buy Signal**"
        insight = f"The model predicts significant upward momentum of {predicted_change_pct:.1f}%. Consider dollar-cost averaging into positions."
    elif predicted_change_pct > 2:
        signal = "**Moderate Buy**"
        insight = f"Positive outlook with {predicted_change_pct:.1f}% expected growth. Good entry point for long-term investors."
    elif predicted_change_pct < -5:
        signal = "**Caution - Consider Selling**"
        insight = f"Model shows downward pressure of {predicted_change_pct:.1f}%. May want to reduce exposure or wait for better entry."
    elif predicted_change_pct < -2:
        signal = "**Hold or Reduce**"
        insight = f"Slight bearish trend of {predicted_change_pct:.1f}%. Monitor closely before adding to positions."
    else:
        signal = "**Hold Current Position**"
        insight = f"Relatively stable outlook with {predicted_change_pct:+.1f}% change. Good for income/dividend investors."
    
    st.markdown(f"""
    ### {signal}
    {insight}
    
    **Volatility Assessment:** {('High' if volatility > current_price * 0.15 else 'Moderate' if volatility > current_price * 0.08 else 'Low')}  
    **Confidence Range:** ±${volatility/2:.2f}
    """)

with col2:
    # Key metrics
    st.metric(
        "Predicted Price Target",
        f"${predicted_price:.2f}",
        f"{predicted_change_pct:+.2f}%"
    )
    
    best_entry = float(forecast_future['yhat_lower'].min())
    st.metric(
        "Best Potential Entry",
        f"${best_entry:.2f}",
        "Lower bound"
    )
    
    best_exit = float(forecast_future['yhat_upper'].max())
    st.metric(
        "Target Exit Price",
        f"${best_exit:.2f}",
        "Upper bound"
    )

# Risk disclaimer (no emoji)
st.info(
    "**Investment Disclaimer:** These predictions are generated by prediction models based on historical data and should NOT be the sole basis "
    "for investment decisions. Always conduct your own research, consider your risk tolerance, and consult with a financial advisor. "
    "Past performance does not guarantee future results."
)

st.markdown("---")

# Plot components
h2("Deep Dive: Market Pattern Analysis", "patterns")
st.caption("Breaking down the price movement into underlying patterns and seasonality")
from prophet.plot import plot_components_plotly
fig_components = plot_components_plotly(model, forecast)
st.plotly_chart(fig_components, use_container_width=True)

with st.expander("How to Read This Chart"):
    st.markdown("""
    - **Trend:** The overall direction of the stock over time (up, down, or sideways)
    - **Weekly Pattern:** Shows which days of the week tend to be stronger or weaker
    - **Yearly Pattern:** Reveals seasonal patterns throughout the year
    
    These patterns help identify the best times to buy or sell based on historical behavior.
    """)

with st.expander("Glossary – Simple Explanations"):
    st.markdown(
        """
        - **SMA (Simple Moving Average):** The average closing price over a set number of days. It smooths short‑term ups and downs.
        - **Log Scale:** Shows equal percentage changes as equal spacing on the chart (useful when prices vary a lot over time).
        - **Seasonality Mode:** How repeating patterns (like weekly or yearly cycles) are modeled.
          - Additive: fixed-size swings.
          - Multiplicative: percentage‑like swings that grow with price.
        - **Trend Flexibility (Changepoint):** How quickly the model adapts to new directions. Lower = smoother; Higher = more reactive.
        - **Volatility/Confidence Range:** Wider bands mean the model is less certain about the future price.
        - **MAE (Mean Absolute Error):** Average prediction error in dollars. Lower is better.
        - **MAPE (Mean Absolute Percentage Error):** Average percentage error. Lower is better.
        """
    )

st.markdown("---")

# Detailed forecast table
h2("Detailed Forecast Data", "table")
st.caption("Daily predictions with confidence intervals for portfolio planning")

# Show forecast data table
if st.checkbox("Show detailed forecast table (next 30 days)"):
    display_forecast = forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).copy()
    display_forecast.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
    display_forecast['Predicted Price'] = display_forecast['Predicted Price'].apply(lambda x: f"${x:.2f}")
    display_forecast['Lower Bound'] = display_forecast['Lower Bound'].apply(lambda x: f"${x:.2f}")
    display_forecast['Upper Bound'] = display_forecast['Upper Bound'].apply(lambda x: f"${x:.2f}")
    st.dataframe(display_forecast, use_container_width=True, hide_index=True)
    st.caption("Predicted Price is the central estimate; Lower/Upper Bounds show the uncertainty range (wider = less certain).")

# Downloads
hist_csv = data.to_csv(index=False).encode('utf-8')
fore_csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).to_csv(index=False).encode('utf-8')
dl_cols = st.columns(2)
with dl_cols[0]:
    st.download_button("Download historical data (CSV)", hist_csv, file_name=f"{stock_ticker}_historical.csv", mime="text/csv")
with dl_cols[1]:
    st.download_button("Download forecast (CSV)", fore_csv, file_name=f"{stock_ticker}_forecast_{forecast_days}d.csv", mime="text/csv")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 2rem; background-color: {profile['bg_color']}; border-radius: 10px; border: 2px solid {profile['color']}20;'>
    <img src='{profile['logo_url']}' style='height: 40px; width: auto; margin-bottom: 1rem;' alt='{selected_stock_name} logo'/>
    <h3 style='color: {profile['color']}; margin: 0.5rem 0;'>{selected_stock_name} Stock Intelligence Dashboard</h3>
    <p style='color: {profile['text_color'] if profile['bg_color'] == '#000000' else '#666'}; margin-top: 1rem; font-size: 0.9rem;'>
        Powered by AI/ML: Facebook Prophet, Python, Streamlit, yfinance, Plotly<br>
        <strong>Built for portfolio demonstration - Not financial advice</strong>
    </p>
</div>
""", unsafe_allow_html=True)
