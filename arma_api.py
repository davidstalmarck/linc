# live_broker.py
import time
import pandas as pd
import numpy as np
import hackathon_linc as lh
import warnings
import os
from dotenv import load_dotenv

# Import the shared AR logic and signals
from arma_strategies import (
    train_arima,
    basic_long_short_signal,
    threshold_signal,
    magnitude_signal,
)

load_dotenv()

API_KEY = os.environ.get("API_KEY")


def rolling_forecast(
    stock_name="STOCK1",
    p=1,
    q=0,
    window_size=50,
    sleep_time=60,
    strategy="basic",
    threshold=0.0005,
    scale_factor=0.001,
    max_shares=50,
):
    lh.init(API_KEY)

    last_position_dir = 0

    while True:
        try:
            print(f"[*] Fetching historical data for {stock_name}")
            data_dict = lh.get_historical_data(365, stock_name)
            df_full = pd.DataFrame(data_dict)
            df_full["gmtTime"] = pd.to_datetime(df_full["gmtTime"])
            df_full.sort_values("gmtTime", inplace=True)
            df_full.reset_index(drop=True, inplace=True)

            # Subset to last 'window_size' rows
            df_rolling = df_full.tail(window_size).copy()

            # Train AR/ARMA/ARIMA
            model = train_arima(df_rolling, p, q, stock_name=stock_name)

            # Forecast
            forecast_val = model.forecast(steps=1).iloc[0]

            # Decide action
            if strategy == "basic":
                signal = basic_long_short_signal(forecast_val, last_position_dir)
                if signal == "BUY":
                    lh.buy(stock_name, amount=10, days_to_cancel=1)
                    last_position_dir = 1
                    print(f"[+] Buying {stock_name}, forecast={forecast_val:.5f}")
                elif signal == "SELL":
                    lh.sell(stock_name, amount=10, days_to_cancel=1)
                    last_position_dir = -1
                    print(f"[-] Selling {stock_name}, forecast={forecast_val:.5f}")

            elif strategy == "threshold":
                signal = threshold_signal(forecast_val, last_position_dir, threshold)
                if signal == "BUY":
                    lh.buy(stock_name, amount=10, days_to_cancel=1)
                    last_position_dir = 1
                elif signal == "SELL":
                    lh.sell(stock_name, amount=10, days_to_cancel=1)
                    last_position_dir = -1

            elif strategy == "magnitude":
                pos_target = magnitude_signal(forecast_val, scale_factor, max_shares)
                if pos_target > 0:
                    lh.buy(stock_name, amount=abs(pos_target), days_to_cancel=1)
                    last_position_dir = 1
                elif pos_target < 0:
                    lh.sell(stock_name, amount=abs(pos_target), days_to_cancel=1)
                    last_position_dir = -1
                # else 0 => hold

            # Sleep
            time.sleep(sleep_time)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    rolling_forecast(
        stock_name="STOCK1",
        p=1,
        q=0,
        window_size=50,
        sleep_time=3,
        strategy="basic",
    )
