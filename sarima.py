import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from backtest_strategy import backtest_strategy, plot_results
from generate_data import generate_random_walk_price_series, generate_ARIMA_process


def train_test_split(df, split_ratio=0.7):
    # Split dataset
    split_idx = int(len(df) * split_ratio)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    # Differentiate input to get return
    df_train["Return"] = df_train["Close"].pct_change()
    # Drop initial NaN from pct_change
    df_train.dropna(subset=["Return"], inplace=True)

    return df_train, df_test


def rolling_forecast_arima_prices(df_train, df_test, p=2, d=1, q=2):
    """
    Rolling (walk-forward) one-step-ahead ARIMA forecast on 'Close' prices,
    using integer indexing for the forecast step.

    df_train, df_test: DataFrames with at least a 'Close' column.
    p, d, q: integers for the ARIMA order.
    """
    # Combine train + test for easy slicing, ensuring time order
    df_all = pd.concat([df_train, df_test], sort=False).sort_index()

    # We'll collect forecasts here, aligned with df_test.index
    forecasts = []

    for test_date in df_test.index:
        # Find location of this test_date in df_all
        test_loc = df_all.index.get_loc(test_date)
        train_end_loc = test_loc - 1

        # Edge case: if test_loc == 0, skip
        if train_end_loc < 0:
            forecasts.append(np.nan)
            continue

        # Slice the training subset from the start up to test_loc-1
        y_train = df_all["Close"].iloc[: train_end_loc + 1]

        # Fit ARIMA on the *price* data, letting statsmodels handle differencing (d=1)
        model = ARIMA(y_train, order=(p, d, q)).fit()
        print(model.summary())

        # Instead of date-based indexing for 'start'/'end', we use integer positions.
        # y_train has length len(y_train). The next "out-of-sample" step is len(y_train).
        forecast_value = model.predict(start=len(y_train), end=len(y_train))
        # forecast_value is a pd.Series with a single value at integer index [len(y_train)]

        forecasts.append(forecast_value.iloc[0])

    # Attach the forecasts to df_test
    df_test["Forecast_Price"] = forecasts
    return df_test


def use_nasdaq_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess NASDAQ stock data from a CSV file.
    """
    df = pd.read_csv(file_path)
    df["Close"] = df["Close/Last"].str.replace("$", "")  # remove the dollar signs
    df["Close"] = df["Close"].astype(float)  # convert the prices to floats
    df = df.iloc[::-1]  # Reverse order
    df.drop(
        columns=["Close/Last", "Open", "High", "Low", "Volume"], inplace=True
    )  # Drop the original columns

    # Index frequency must be set for ARIMA
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index(["Date"])

    return df


def generate_signal(df, forecast_col="Forecast_Price"):
    """
    Generate a trading signal based on the forecasted price.
    """
    # SHIFT FORECAST by 1 day to avoid lookahead bias
    df_test["Forecast_Price_Shifted"] = df_test[forecast_col].shift(1)

    # SHIFT the actual Close if you want the "yesterday's close" as reference
    df_test["Close_Shifted"] = df_test["Close"].shift(1)

    # Generate signals:
    # +1 if forecasted price(t) > actual price(t-1) => expecting an upward move
    # -1 if forecasted price(t) < actual price(t-1) => expecting a downward move
    df_test["Signal"] = 0
    df_test.loc[
        df_test["Forecast_Price_Shifted"] > df_test["Close_Shifted"], "Signal"
    ] = 1
    df_test.loc[
        df_test["Forecast_Price_Shifted"] < df_test["Close_Shifted"], "Signal"
    ] = -1

    # Replace any NaN with 0
    df_test["Signal"] = df_test["Signal"].fillna(0)

    # Threshold-based signal generation
    # threshold = 0.5  # e.g., require a 0.5 unit price difference
    # df_test["Expected_Move"] = (
    #     df_test["Forecast_Price_Shifted"] - df_test["Close_Shifted"]
    # )

    # df_test["Signal"] = 0
    # df_test.loc[df_test["Expected_Move"] > threshold, "Signal"] = 1
    # df_test.loc[df_test["Expected_Move"] < -threshold, "Signal"] = -1
    return df


if __name__ == "__main__":
    # Use AMD stock data
    df = use_nasdaq_data("datasets/AMD.csv")

    # Generate ARIMA process
    # df = generate_ARIMA_process(p=2, d=1, q=2, start_price=100, num_days=1000)

    # Train test split
    df_train, df_test = train_test_split(df, split_ratio=0.7)

    # Differentiate training set
    train_diff = df_train["Close"].diff().dropna()

    # Plot training set (returns, differentiated close prices)
    plt.plot(train_diff)
    plt.title("Training set close prices")
    plt.show()

    # Plot ACF and PACF
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(train_diff, lags=50, ax=ax[0])
    plot_pacf(train_diff, lags=50, ax=ax[1])
    plt.show()

    # Perform rolling forecast (ARIMA)
    p, d, q = 2, 1, 2

    # Specify constraints to fix lags 1, 3,4,5,6,7,8 at 0.
    # constraints = {
    #     "ar.L3": 0,
    #     "ar.L4": 0,
    #     "ar.L5": 0,
    #     "ar.L6": 0,
    #     "ar.L7": 0,
    #     "ar.L8": 0,
    # }

    # model = sm.tsa.ARIMA(
    #     df_train["Close"], order=(p, d, q), enforce_stationarity=False
    # ).fit_constrained(constraints=constraints)

    model = ARIMA(df_train["Close"], order=(p, d, q)).fit()
    print(model.summary())

    # Plot ACF and PACF of residuals
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(model.resid, lags=50, ax=ax[0])
    plot_pacf(model.resid, lags=50, ax=ax[1])
    plt.show()

    forecast = model.forecast(steps=10)
    # Set forecast index to continue from the last date in df_train
    forecast.index = pd.date_range(start=df_train.index[-1], periods=10, freq="D")
    plt.plot(df_train["Close"])
    plt.plot(forecast)
    plt.show()

    exit()

    df_test = rolling_forecast_arima_prices(df_train, df_test, p, d, q)

    plt.figure(figsize=(10, 6))
    plt.plot(df_test.index, df_test["Close"], label="Actual Price")
    plt.plot(
        df_test.index,
        df_test["Forecast_Price"],
        label="Forecasted Price",
        linestyle="--",
    )
    plt.title("ARIMA Rolling Forecast vs. Actual Price")
    plt.legend()
    plt.show()

    # Generate trading signals
    df_test = generate_signal(df_test)

    # Backtest the strategy
    df_test, total_return = backtest_strategy(df_test, signal_col="Signal")
    print(f"Strategy total return: {total_return:.2%}")
    plot_results(df_test, title="ARIMA Strategy Backtest")

    # Save the results
    df_test.to_csv("results/arima_results.csv")
