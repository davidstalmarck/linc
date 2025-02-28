import time
import pandas as pd
import numpy as np
import hackathon_linc as lh
import warnings
import os
from dotenv import load_dotenv
from mvo_strategies import mvo_optimize

load_dotenv()

API_KEY = os.environ.get("API_KEY")


def get_returns_wide_live(days_back=365):
    """
    Fetch data from the broker for all tickers, convert to a wide returns DataFrame.
    We assume you want to consider multiple symbols (since MVO typically is multi-asset).
    """
    data_dict = lh.get_historical_data(days_back)
    df = pd.DataFrame(data_dict)
    df["gmtTime"] = pd.to_datetime(df["gmtTime"])
    df.sort_values("gmtTime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["midPrice"] = (df["askMedian"] + df["bidMedian"]) / 2.0
    df["returns"] = df.groupby("symbol")["midPrice"].pct_change()

    # Add a "CASH" column with 0.0 returns at every timestamp

    returns_wide = df.pivot(index="gmtTime", columns="symbol", values="returns")
    returns_wide.dropna(how="all", inplace=True)

    returns_wide["CASH"] = 0.0
    return returns_wide


def live_mvo_trading(
    target_return=0.001,
    capital=10_000,
    sleep_time=60,
    window_size=100,
    trade_threshold=5,
):
    """
    Live trading loop:
      - uses MVO to compute weights from the last `window_size` days
      - only trades the difference between current holdings and target holdings
      - trades are skipped if the difference in shares is below `trade_threshold`
    """
    lh.init(API_KEY)

    while True:
        try:
            print("[*] Fetching historical data...")
            returns_wide = get_returns_wide_live(days_back=window_size)

            print("[*] Solving MVO...")
            w_opt = mvo_optimize(returns_wide, target_return, long_only=True)
            print("MVO Weights:\n", w_opt)

            # 1) Fetch current portfolio
            portfolio = lh.get_portfolio()

            # 2) For each symbol in MVO solution, find out how many shares we WANT
            #    shares_desired = (weight * capital) / current_price
            current_prices = {}
            for symbol in w_opt.index:
                if symbol == "CASH":
                    continue
                response = lh.get_current_price(symbol)
                data_obj = response["data"][0]
                mid = (data_obj["askMedian"] + data_obj["bidMedian"]) / 2.0
                current_prices[symbol] = mid

            # Also consider any symbol we currently hold but isn't in w_opt
            # (the MVO weight is effectively zero for that symbol)
            all_symbols = set(portfolio.keys()).union(set(w_opt.index))

            for symbol in all_symbols:
                if symbol == "CASH":
                    continue
                # Desired weight is 0 if symbol not in w_opt
                weight = w_opt.get(symbol, 0.0)

                # Current price
                if symbol not in current_prices:
                    # If we don't have a price (maybe symbol not in w_opt?), fetch it
                    response = lh.get_current_price(symbol)
                    data_obj = response["data"][0]
                    mid = (data_obj["askMedian"] + data_obj["bidMedian"]) / 2.0
                    current_prices[symbol] = mid
                price = current_prices[symbol]

                # Compute how many shares we want
                desired_shares = int((weight * capital) / price) if price > 0 else 0

                # Check how many we currently have
                current_shares = portfolio.get(symbol, 0)

                # difference in shares
                diff = desired_shares - current_shares

                if abs(diff) >= trade_threshold:
                    # Positive => we need to buy the difference
                    # Negative => we need to sell
                    if diff > 0:
                        # buy `diff` shares
                        lh.buy(symbol, amount=diff, days_to_cancel=1)
                        print(
                            f"[+] Buying {diff} {symbol} (current={current_shares}, desired={desired_shares})"
                        )
                    else:
                        # sell `abs(diff)` shares
                        to_sell = abs(diff)
                        lh.sell(symbol, amount=to_sell, days_to_cancel=1)
                        print(
                            f"[-] Selling {to_sell} {symbol} (current={current_shares}, desired={desired_shares})"
                        )

                    # Sleep a bit to avoid spamming server
                    # time.sleep(1)
                else:
                    # If difference is below threshold, skip
                    pass

            print("[*] Done rebalancing; sleeping...")
            time.sleep(sleep_time)

        except Exception as e:
            print("Error:", e)
            time.sleep(sleep_time)


def main():
    warnings.filterwarnings("ignore")
    live_mvo_trading(
        target_return=0.00015,  # Example target
        capital=100_000,
        sleep_time=5,
        window_size=100,
        trade_threshold=3,  # Only rebalance if difference >= 5 shares
    )


if __name__ == "__main__":
    main()
