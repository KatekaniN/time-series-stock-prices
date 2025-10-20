# Time Series Forecasting Dashboard 📈

An interactive web application that forecasts stock prices using Facebook Prophet and displays insights through a beautiful Streamlit dashboard.

## Features

- **Real-time Stock Data**: Fetches historical stock data from Yahoo Finance
- **Multiple Stocks**: Choose from popular stocks (Apple, Microsoft, Google, Tesla, etc.)
- **Prophet Forecasting**: Uses Facebook's Prophet algorithm for time series prediction
- **Interactive Visualizations**: Beautiful charts with Plotly for data exploration
- **Trend Analysis**: Breaks down trends, seasonality, and patterns
- **Configurable Forecasts**: Adjust historical data range and forecast period

## Tech Stack

- **Python 3.8+**
- **Streamlit**: Web dashboard framework
- **Prophet**: Time series forecasting
- **yfinance**: Stock data API
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation

## Installation

1. Clone this repository or download the files

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

## Deploy options

### 1) Streamlit Community Cloud (fastest)
1. Push the repo to GitHub (already done).
2. Go to https://share.streamlit.io/ and click "New app".
3. Select this repo and set the main file as `app.py`.
4. Optional: In "Advanced settings", set Python version to 3.12+ and add any environment variables.

### 2) Render (free tier)
1. Create a new Web Service.
2. Connect your GitHub repo and pick this project.
3. Set Runtime to Python, and add this `Procfile` automatically detected:
	- `web: sh -c "python -m pip install -r requirements.txt && python -m streamlit run app.py --server.port $PORT --server.address 0.0.0.0"`
4. Set environment to use Python 3.12 or later.
5. Deploy; Render will install dependencies and run Streamlit.

Notes:
- Streamlit settings are in `.streamlit/config.toml` (headless enabled for servers).
- This app uses yfinance (no API key). If you add external APIs, set secrets via platform settings.

The dashboard will open in your browser at `http://localhost:8501`

## How to Use

1. **Select a Stock**: Choose from the dropdown in the sidebar
2. **Set Historical Range**: Adjust how many years of data to analyze
3. **Set Forecast Period**: Choose how many days to forecast
4. **Explore**: View historical data, forecast, and trend components

## Project Structure

```
time-series-forecasting/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Model Details

The project uses **Facebook Prophet**, which is designed for:

- Handling missing data and outliers
- Automatic detection of trends and seasonality
- Easy-to-interpret forecasts with uncertainty intervals

## Screenshots

(Add screenshots of your dashboard here when running)

## Future Enhancements

- [ ] Add more data sources (crypto, commodities, weather)
- [ ] Implement LSTM models for comparison
- [ ] Add model performance metrics (RMSE, MAE)
- [ ] Save and export forecasts
- [ ] Add custom data upload feature

## Disclaimer

This project is for educational purposes only. Stock market predictions are inherently uncertain and should not be used for actual investment decisions.

## License

MIT License - Feel free to use this for your portfolio!

## Author

Katekani Nyamandi

---

Built with ❤️ using Python and Streamlit
