import pandas as pd

class GoldenCrossStrategy:
    def __init__(self, data, short_window=160, long_window=800):  # 20-day & 100-day moving average
        self.data = data.copy()
        self.short_window = short_window
        self.long_window = long_window

    def compute_moving_averages(self):
        """Calculate moving averages using the market price."""
        self.data['market_price'] = (self.data['bid_price'] + self.data['ask_price']) / 2  # Market price
        self.data['short_ma'] = self.data['market_price'].rolling(self.short_window).mean()
        self.data['long_ma'] = self.data['market_price'].rolling(self.long_window).mean()

    def generate_signals(self):
        """Generate trading signals: 1 = Buy, 0 = Hold, -1 = Sell."""
        self.data['signal'] = 0  # Default to Hold

        # Buy Signal (Golden Cross)
        self.data.loc[
            (self.data['short_ma'] > self.data['long_ma']) & 
            (self.data['short_ma'].shift(1) <= self.data['long_ma'].shift(1)), 'signal'
        ] = 1

        # Sell Signal (Death Cross)
        self.data.loc[
            (self.data['short_ma'] < self.data['long_ma']) & 
            (self.data['short_ma'].shift(1) >= self.data['long_ma'].shift(1)), 'signal'
        ] = -1

        # Prevent trading before enough data is available
        self.data.loc[:self.long_window, 'signal'] = 0

    def run(self):
        """Run the strategy and return the processed DataFrame with buy/hold/sell signals."""
        self.compute_moving_averages()
        self.generate_signals()
        return self.data[['market_price', 'short_ma', 'long_ma', 'signal']]
