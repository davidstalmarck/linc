import pandas as pd
import numpy as np


def generate_random_walk_price_series(start_price=100, num_days=500):
    """Simulate a random-walk price series, currently only close price."""
    dates = pd.date_range(start="2020-01-01", periods=num_days, freq="D")
    # Start from some price, e.g. 100
    price = start_price + np.random.normal(loc=0, scale=1, size=len(dates)).cumsum()

    df = pd.DataFrame({"Date": dates, "Close": price})
    df.set_index("Date", inplace=True)

    df.head()
    return df


def generate_ARIMA_process(p, d, q, start_price=100, num_days=500):
    """Simulate an ARIMA(p,d,q) process with random AR, MA coefficients."""
    # Generate random AR, MA coefficients
    ar = np.random.normal(loc=0, scale=0.1, size=p)
    ma = np.random.normal(loc=0, scale=0.1, size=q)

    # Generate innovation noise
    noise = np.random.normal(loc=0, scale=1, size=num_days)

    # 1. Generate an ARMA(p,q) on a differenced/stationary series
    # We'll store that in 'arma_diff' (the differenced data).
    arma_diff = np.zeros(num_days)

    max_lag = max(p, q)
    for i in range(max_lag, num_days):
        # AR term
        ar_term = 0
        for lag_idx in range(p):
            ar_term += ar[lag_idx] * arma_diff[i - lag_idx - 1]

        # MA term
        ma_term = 0
        for lag_idx in range(q):
            ma_term += ma[lag_idx] * noise[i - lag_idx - 1]

        arma_diff[i] = ar_term + ma_term + noise[i]

    # 2. Now "integrate" arma_diff d times to get the actual ARIMA series
    #   i.e., if d=1, do a cumulative sum once;
    #         if d=2, do it again, etc.
    final_series = arma_diff.copy()
    for _ in range(d):
        final_series = np.cumsum(final_series)

    # 3. Shift by start_price so the process doesn't start near zero
    #    If you want exactly 'start_price' at t=0, you can do:
    #    final_series += (start_price - final_series[0])
    #    or just add start_price unconditionally as below.
    final_series += start_price

    # Construct dataframe
    dates = pd.date_range(start="2020-01-01", periods=num_days, freq="D")
    df = pd.DataFrame({"Date": dates, "Close": final_series})
    df.set_index("Date", inplace=True)

    return df


# For testing the functions
if __name__ == "__main__":
    df = generate_ARIMA_process(p=3, d=1, q=0)
    print(df.head())
    print(df.tail())

    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(df["Close"], lags=50, ax=ax[0])
    plot_pacf(df["Close"], lags=50, ax=ax[1])
    plt.show()

    plt.plot(df["Close"])
    plt.show()
