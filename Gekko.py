from BudFox import BudFox
from BookWorm import BookWorm
from Portfolio import Portfolio

from strategies.Strategy import Strategy

class Gekko:
    """
    The engine room of Falkor. Every iteration, Gekko receives an input 
    of live ochl data. It sends this data to the selected Strategy, which 
    generates buy/sell signals and real-time model updating. Gekko takes 
    these signals and sends them to BudFox who will realize them. 
    Gekko then updates portfolio with any profits made
    
    Attributes:
        
        bud_fox: BudFox
            - An instance of BudFox class used to send buy/sell signals to 
            an APIWrapper for realization

        portfolio: Portfolio
            - An instance of Portfolio class that stores all trades and 
            securities  

        book_worm: BookWorm
            - An instance of BookWorm class used for pulling live data 
            from APIWrappers

    Methods:

        _trade(self, s)
            - helper method used to trade one Security, s.

        trade_portfolio(self)
            - calls _trade on every Security inside portfolio

    """

    def __init__(self, portfolio: Portfolio):
        """Initialize Gekko"""
        self.portfolio = portfolio

        self.bud_fox = BudFox()
        self.book_worm = BookWorm()

    def _trade(self, sec):
        """
        Helper method for self.trade_portfolio(). Runs strategy for security 
        and sends signals to api_wrapper
        """

        # If this security is waiting for a further time period to be traded,
        # don't trade it but update waiting periods
        if sec.status == "waiting":
            # subtract one from _waiting_periods_left
            sec.update_waiting_status(amount=-1)
            return "Gekko: This security has {} waiting periods left".format(
                sec._waiting_periods_left)
        
        # trade this security if its ready
        elif sec.status == "ready":
            
            # Get 100 most recent candles of data
            # NOTE: Creating technical indicators cuts 100 candles down to ~ 50-80 depending on the indicator. We want 30 
            # periods, so we take 100 to leave us with plenty of room to spare       
            last_candles = self.book_worm.last_candles(100, sec.api_wrapper, sec.symbol, sec.interval)

            # Get trading signals from strategy

            # Feed last_candles to strategy
            sec.strategy.feed_data(last_candles)
            # Make a prediction based on last_candles
            signal = sec.strategy.predict()
            # Update model however specified
            sec.strategy.update()


            # If strategy predicts price growth in 5 periods, we set 
            # s.status == waiting until 5 periods from now

            if signal == "buy":
                sec.set_status(status="waiting", periods=5)

            # Send signal to BudFox for realization

            # tell BudFox to paper trade instead of real trade
            self.bud_fox.paper_trade = True

            # send trading signal to BudFox
            trade_info = self.bud_fox.send_trading_signal(sec.symbol, signal, amount=20, api_wrapper=sec.api_wrapper, price="market")

            # return information pertaining to this trade
            return trade_info

        else:
            raise SyntaxError("sec.status is set to something other than ready or waiting")

    def trade_portfolio(self):
        """Iterates through every security in self.portfolio, using their specified Strategy and APIWrapper"""
        for security in self.portfolio.securities_trading:
            trade_info = self._trade(security)
            print(trade_info)

    def _split_into_periods(self, candles_df, split_row_num):
        """split candles_df into a list of smaller DataFrames each
        with row_num of rows"""
        orig_row_num = candles_df.shape[0]

        candles_split = []
        for i in range(orig_row_num - split_row_num):
            small_df = candles_df[i : i + split_row_num]
            candles_split.append(small_df)

        return candles_split

    def _curr_fut_prices(self, candles_df, split_row_num):
        """used in concordance with _split_into_periods().Returns a
        list of curr_price and fut_price for each item yi inside 
        y = _split_into_periods(x)"""
        orig_row_num = candles_df.shape[0]

        curr_prices, fut_prices = [], []

        for i in range(orig_row_num - split_row_num):
            curr_prices.append(candles_df[i])
            fut_prices.append(candles_df[i+split_row_num])

        return curr_prices, fut_prices

    def backtest(self, strategy, candles_df):
        split_candles = self._split_into_periods(candles_df, 30)
        curr_prices, fut_prices = self._curr_fut_prices(candles_df, 30)

        total_profit = 0
        for i in range(len(split_candles)):
            input_df = split_candles[i]

            pred = strategy.generate_signals(input_df)

            money = 0
            if pred > 1:
                money -= curr_prices[i]
                money += fut_prices[i]
            elif pred < 1:
                money += curr_prices[i]

            total_profit += money
        
        return total_profit