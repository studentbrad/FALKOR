from .data_feeds import LiveTrade, BackTest

class BudFox:
    """Executes trades for Engine and keeps Portfolio updated"""

    def __init__(self, api_wrapper=None):
        self.api_wrapper = api_wrapper
        self.trades = []

    def execute(self, signal, price):
        if signal == 'buy':
            if self.api_wrapper:
                self.api_wrapper.buy_order(amount=1, price=price)

        elif signal == 'sell':
            if self.api_wrapper:
                self.api_wrapper.sell_order(amount=1, price=price)

            self.trades.append( signal, price )

class Engine:
    """Main method of FALKOR"""

    def __init__(self, mode, strategy, candles_path=None, api_wrapper=None, symbol=None, interval=None):
        self.strategy = strategy

        if mode == 'backtest':
            self.data_feed = BackTest(candles_path)
            self.budfox = BudFox()
            
        elif mode == 'live':
            self.data_feed = LiveTrade(api_wrapper, symbol, interval)
            self.budfox = BudFox(api_wrapper)

    def iterate(self):
        # note - this line will throw an Exception once backtesting is complete. surround this in a try catch statement
        # in the parent
        latest_candles = self.data_feed.latest_candles()

        self.strategy.feed_data(latest_candles)
        signal = self.strategy.predict()
        self.strategy.update()

        curr_price = latest_candles['close'][-1]
        self.budfox.execute(signal, curr_price)

    def run(self):
        while True:
            try:
                self.iterate()
            except:
                break

        print(self.budfox.trades)
        print('done')