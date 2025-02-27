# import matplotlib.pyplot as plt
# import hackathon_linc as lh
# import pandas as pd
# import numpy as np
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# def train(file_path, p, q, stock_name="STOCK1"):
#     """
#     Train an ARMA model on the stock returns
#     """
#     # Read data
#     df = pd.read_csv(file_path)
#     df.drop(columns=["gmtTime"], inplace=True)

#     # We buy based the average price of the ask and bid
#     df["avgPrice"] = (df["askMedian"] + df["bidMedian"]) / 2

#     # Compute returns
#     df["returns"] = df["avgPrice"].pct_change()
#     df.dropna(subset=["returns"], inplace=True)

#     # Fit ARMA on the given stock
#     stock_df = df[df["symbol"] == stock_name]
#     model = ARIMA(stock_df["returns"], order=(p, 0, q)).fit()

#     return df, model


# def plot_data_acf_pacf(df, y_min, y_max):
#     fig, ax = plt.subplots(2, 1, figsize=(10, 6))
#     plot_acf(df["returns"], lags=50, ax=ax[0], zero=False)
#     plot_pacf(df["returns"], lags=50, ax=ax[1], zero=False)
#     ax[0].set_ylim(y_min, y_max)
#     ax[1].set_ylim(y_min, y_max)
#     plt.show()


# def plot_resid_acf_pacf(model, y_min, y_max):
#     fig, ax = plt.subplots(2, 1, figsize=(10, 6))
#     plot_acf(model.resid, lags=50, ax=ax[0], zero=False)
#     plot_pacf(model.resid, lags=50, ax=ax[1], zero=False)
#     ax[0].set_ylim(y_min, y_max)
#     ax[1].set_ylim(y_min, y_max)
#     plt.show()


# # Ljung-Box (Q) at L1:
# # Tests for autocorrelation in the residuals. If the p-value is high (like 0.13 here),
# # it suggests no significant autocorrelation left in the residuals at lag 1.
# # That’s generally good because we want white-noise residuals.

# # Jarque-Bera (JB):
# # Tests whether residuals deviate from a normal distribution.
# # A p-value of 0.15 means we don’t reject normality.
# # This is also typically good if the model assumptions require or
# # prefer normally distributed residuals.

# # Heteroskedasticity (H) test:
# # If p-value is near 0.00 (two-sided < 0.05),
# # it indicates the residuals’ variance might change over time
# # (i.e., possible heteroskedasticity). That’s often seen in financial or economic data.
# # It might not invalidate your model, but it’s something to be aware of - sometimes
# # a GARCH model is used to handle that.


# if __name__ == "__main__":
#     lh.init("")

#     df, model = train("datasets/stockPrices_hourly.csv", 2, 2, stock_name="STOCK1")
#     print(model.summary())
#     # Note: high heteroskedasticity p-value (0.00) indicates that the residuals' variance might change over time.

#     # Plot ACF and PACF of the return without the zero lag and auto y limit
#     plot_data_acf_pacf(df, -0.2, 0.2)

#     # Plot ACF and PACF of residuals
#     plot_resid_acf_pacf(model, -0.2, 0.2)

#     rate = 2.0

#     # TODO A loop that exectures at the rate of 2.0 (2 seconds between iterations)


import time
import pandas as pd
import numpy as np
import hackathon_linc as lh
from statsmodels.tsa.arima.model import ARIMA


def train_arima(df, p, q, stock_name="STOCK1"):
    """
    Train an ARIMA/ARMA model on the stock returns for a specified window of data.
    """
    # We buy based on the average price of the ask and bid
    df["avgPrice"] = (df["askMedian"] + df["bidMedian"]) / 2

    # Compute returns
    df["returns"] = df["avgPrice"].pct_change()
    df.dropna(subset=["returns"], inplace=True)

    # Filter for the selected symbol
    stock_df = df[df["symbol"] == stock_name]

    # Fit ARIMA (p,0,q)
    model = ARIMA(stock_df["returns"], order=(p, 0, q)).fit()
    return model


def rolling_forecast(stock_name="STOCK1", p=1, q=0, window_size=50, sleep_time=60):
    """
    Continuously performs a rolling forecast on the specified stock.
    Re-fetches recent data, trains/refits the model on a rolling window,
    generates a forecast, and places trades via hackathon_linc.

    Parameters
    ----------
    stock_name : str
        Ticker symbol to trade.
    p : int
        AR order.
    q : int
        MA order.
    window_size : int
        Number of data points (e.g., hours or days) used in each rolling training window.
    sleep_time : int
        How many seconds to wait between each forecast/trade cycle.
    """
    # Initialize your trading key once at the start
    lh.init("")

    # Simple placeholders to track last action or position
    last_position = 0  # +1 for long, -1 for short, 0 for flat

    while True:
        try:
            data_dict = lh.get_historical_data(365, stock_name)
            df_full = pd.DataFrame(data_dict)

            # For demonstration: keep only the last `window_size` rows for the rolling window
            df_rolling = df_full.tail(window_size).copy()

            # Train/retrain on the rolling window
            model = train_arima(df_rolling, p, q, stock_name=stock_name)

            # 1-step-ahead forecast of returns
            forecast = model.forecast(steps=1)[0]

            # Basic strategy:
            # If forecast > 0 => buy signal
            # If forecast < 0 => sell signal
            # This is extremely naive and is purely illustrative.

            if forecast > 0 and last_position <= 0:
                # Buy if we are not already long.
                # Example: buy 10 shares at market price
                current_price = lh.get_current_price(stock_name)[
                    stock_name
                ]  # :contentReference[oaicite:1]{index=1}
                lh.buy(
                    stock_name, amount=10, price=int(current_price), days_to_cancel=1
                )  # :contentReference[oaicite:2]{index=2}
                last_position = 1
                print(f"[+] Buying {stock_name} because forecast={forecast:.5f}")

            elif forecast < 0 and last_position >= 0:
                # Sell if we are not already short.
                # For a short, you might need to hold negative shares (depending on sim constraints).
                # If not supported, you can simply do a 'sell' if you currently hold a position.
                current_price = lh.get_current_price(stock_name)[stock_name]
                # For a new short: you might do something like:
                # lh.sell(stock_name, amount=10, price=int(current_price), days_to_cancel=1)
                # If you just want to close a previous long:
                lh.sell(
                    stock_name, amount=10, price=int(current_price), days_to_cancel=1
                )  # :contentReference[oaicite:3]{index=3}
                last_position = -1
                print(f"[-] Selling {stock_name} because forecast={forecast:.5f}")

            # Sleep until the next iteration
            time.sleep(sleep_time)

        except Exception as e:
            print(f"Error occurred: {e}")
            # Sleep briefly before retrying
            time.sleep(5)


if __name__ == "__main__":
    # Example usage: AR(1) model with no MA terms, 100 data points in each rolling window
    # 120-second sleep time between retraining/forecast cycles
    rolling_forecast(stock_name="STOCK1", p=1, q=0, window_size=100, sleep_time=120)
