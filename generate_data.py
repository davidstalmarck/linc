import pandas as pd
import numpy as np

def generate_random_walk_price_series(start_price=100, num_days=500):
    # Simulate a random-walk price series
    dates = pd.date_range(start="2020-01-01", periods=num_days, freq="D")
    # Start from some price, e.g. 100
    price = start_price + np.random.normal(loc=0, scale=1, size=len(dates)).cumsum()

    df = pd.DataFrame({"Date": dates, "Close": price})
    df.set_index("Date", inplace=True)

    df.head()
    return df