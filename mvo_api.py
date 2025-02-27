import time
import pandas as pd
import numpy as np
import hackathon_linc as lh
import warnings
from mvo_strategies import mvo_optimize


def get_returns_wide_live(days_back=365):
    """
    Fetch data from the broker for all tickers, convert to a wide returns DataFrame.
    We assume you want to consider multiple symbols (since MVO typically is multi-asset).
    """
    # Example: fetch data for all or multiple symbols
    # If the API can fetch for multiple symbols at once, do so.
    # Otherwise you might have to loop over the symbols.
    data_dict = lh.get_historical_data(
        days_back
    )  # returns data for all? depends on the API
    df = pd.DataFrame(data_dict)
    df["gmtTime"] = pd.to_datetime(df["gmtTime"])
    df.sort_values("gmtTime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["midPrice"] = (df["askMedian"] + df["bidMedian"]) / 2.0
    df["returns"] = df.groupby("symbol")["midPrice"].pct_change()

    returns_wide = df.pivot(index="gmtTime", columns="symbol", values="returns")
    returns_wide.dropna(how="all", inplace=True)
    return returns_wide


def live_mvo_trading(target_return=0.001, capital=10_000, sleep_time=60):
    lh.init("YOUR-API-KEY")

    while True:
        try:
            # 1) fetch data
            returns_wide = get_returns_wide_live(days_back=365)

            # 2) solve MVO
            w_opt = mvo_optimize(returns_wide, target_return, long_only=True)
            print("MVO Weights:\n", w_opt)

            # 3) convert weights -> positions
            # Suppose we buy each stock weighting of 'w_opt[i]' times 'capital'
            # Then the number of shares = (weight * capital) / current_price
            # So we need the current price for each symbol from the broker
            current_prices = {}
            # Example: we fetch each symbol’s current price
            for symbol in w_opt.index:
                response = lh.get_current_price(symbol)
                # response = { "data": [ { "askMedian":..., "bidMedian":..., "symbol":... } ] }
                data_obj = response["data"][0]
                mid = (data_obj["askMedian"] + data_obj["bidMedian"]) / 2.0
                current_prices[symbol] = mid

            # 4) place orders
            # Before placing new orders, you might want to cancel existing ones or flatten your portfolio.
            # For simplicity, let's flatten everything by selling all first:
            # (In reality you'd track existing positions or do partial rebalancing.)
            portfolio = lh.get_portfolio()  # dict: { symbol: shares }
            for sym, qty in portfolio.items():
                if qty > 0:
                    lh.sell(sym, amount=qty, days_to_cancel=1)
                elif qty < 0:
                    # If negative positions are allowed or exist (short?), then we might buy to cover.
                    lh.buy(sym, amount=abs(qty), days_to_cancel=1)

            # Now place new orders according to w_opt
            for symbol, weight in w_opt.items():
                if weight <= 0:
                    continue
                price = current_prices[symbol]
                # # of shares we want to buy
                shares = int((weight * capital) / price)
                if shares > 0:
                    lh.buy(symbol, amount=shares, days_to_cancel=1)
                    print(
                        f"[+] Buying {shares} of {symbol} at ~{price:.2f}, weight={weight:.3f}"
                    )

            print("[*] Done placing orders; sleeping...")
            time.sleep(sleep_time)

        except Exception as e:
            print("Error:", e)
            time.sleep(5)


def main():
    warnings.filterwarnings("ignore")
    live_mvo_trading(
        target_return=0.001, capital=10_000, sleep_time=60  # 0.1% per day (or hour)
    )


if __name__ == "__main__":
    main()
