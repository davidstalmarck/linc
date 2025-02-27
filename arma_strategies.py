# ar_strategies_common.py
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def train_arima(df, p, q, stock_name="STOCK1"):
    """
    Trains an AR/ARIMA model on the returns of a single stock,
    given a rolling window or historical slice of data.
    """
    # Average price
    df["avgPrice"] = (df["askMedian"] + df["bidMedian"]) / 2
    df["returns"] = df["avgPrice"].pct_change()
    df.dropna(subset=["returns"], inplace=True)

    # Filter by symbol
    stock_df = df[df["symbol"] == stock_name]
    # Fit ARIMA (p,0,q)
    model = ARIMA(stock_df["returns"], order=(p, 0, q)).fit()
    return model


def basic_long_short_signal(forecast, last_position):
    """
    Returns 'BUY', 'SELL', or 'HOLD' for a basic long/short approach.
    """
    if forecast > 0 and last_position <= 0:
        return "BUY"
    elif forecast < 0 and last_position >= 0:
        return "SELL"
    else:
        return "HOLD"


def threshold_signal(forecast, last_position, threshold=0.0005):
    """
    Returns 'BUY', 'SELL', or 'HOLD' if forecast crosses thresholds.
    """
    if forecast > threshold and last_position <= 0:
        return "BUY"
    elif forecast < -threshold and last_position >= 0:
        return "SELL"
    else:
        return "HOLD"


def magnitude_signal(forecast, scale_factor=0.001, max_shares=50):
    """
    Returns an integer position based on forecast magnitude.
    """
    pos = int(abs(forecast) / scale_factor)
    pos = min(pos, max_shares)
    return pos if forecast > 0 else -pos if forecast < 0 else 0
