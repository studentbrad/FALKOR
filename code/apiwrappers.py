"""
This module contains all API Wrappers used to interact with online brokerages
"""

from binance.client import Client
import pandas as pd

class APIWrapper:
    """Abstract class representing an API Wrapper for a financial security. 
    All required methods must be completed by child class"""

    def historical_candles(self, symbol: str, candle_width:str, start_time: str, end_time: str):
        """Returns DataFrame with columns = ['time', 'open', 'close', 'high', 'low', 'volume'] 
        for the security. Number of rows is equal to (end_time - start_time) / candle_width"""

        raise NotImplementedError

    def last_candles(self, symbol, interval):
        """Returns DataFrame containing the most recent candlesticks information"""

        raise NotImplementedError

    def live_tickers(self):
        """Returns a dictionary containing current ticker info"""

        raise NotImplementedError

    def buy_order(self, amount: int, price="market"):
        """Created a buy order for amount. If price != "market", then a buy order will be created at price=price"""

        raise NotImplementedError

    def sell_order(self, amount:int, price="market"):
        """Created a sell order for amount. If price != "market", then a sell order will be created at price=price"""

        raise NotImplementedError

    def check_balance(self,):
        """Returns portfolio balance"""

        raise NotImplementedError

    def trade_status(self, trade_id: str):
        """Returns the status for trade with id trade_id"""

        raise NotImplementedError

    def cancel_order(self, trade_id: str):
        """Cancel order with trade_id"""

        raise NotImplementedError
   

class BinanceWrapper(APIWrapper):
    """APIWrapper for the Binance Exchange"""

    def __init__(self, client_id, client_secret):
        """Initialize BinanceWrapper"""
        self.client = Client(client_id, client_secret)

    def historical_candles(self, symbol: str, interval:str, start_time: str, end_time: str=None):
        """
        Returns DataFrame with columns = ['time', 'open', 'close', 'high', 'low', 'volume']
        """
        if end_time:
            candles = self.client.get_historical_klines(symbol, interval, start_str=start_time, end_str=end_time)
        else:
            candles = self.client.get_historical_klines(symbol, interval, start_str=start_time)

        data_df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close',
                                   'volume', 'close_time','quote_asset_volume',
                                   'num_trades', 'tkbbav', 'tkqav', 'ign.'])

        data_df = data_df[['time', 'open', 'close', 'high', 'low', 'volume']]

        # Convert all numbers from str to float
        data_df = data_df.applymap(float)

        return data_df

    def last_candles(self, num, symbol, interval):
        """
        Returns DataFrame with columns =['time', 'open', 'close', 'high', 'low', 'volume'] 
        with num rows of latest candles
        NOTE: only goes back 2 days max 
        """

        # TODO replace start_str='2 days ago' with a dynamic alternative
        candles = self.client.get_historical_klines(symbol, interval,
                                                    start_str='2 days ago')

        data_df = pd.DataFrame(candles,
                               columns=['time', 'open', 'high', 'low', 'close',
                                        'volume', 'close_time',
                                        'quote_asset_volume',
                                        'num_trades', 'tkbbav', 'tkqav',
                                        'ign.'])

        data_df = data_df[['time', 'open', 'close', 'high', 'low', 'volume']]
        data_df = data_df.iloc[data_df.shape[0]-num:, :] # take only num last rows

        # Convert all numbers from str to float
        data_df = data_df.applymap(float)

        return data_df


    def buy_order(self, symbol:str, amount: int, price="market"):
        """
        Created a buy order for amount. If price != "market", then a buy order
        will be created at price=price
        """

        if price == "market":
            order = self.client.order_market_buy(symbol=symbol, quantity=amount)
        else:
            order = self.client.order_limit_buy(symbol=symbol, quantity=amount,
                                                price=price)
        return order

    def tickers(self):
        """Return a dictionary of all symbols as key and their current price as value"""
        x = self.client.get_all_tickers()
        ticks = {}
        
        for dct in x:
            symbol = dct['symbol']
            price = dct['price']
            ticks[symbol] = float(price)
        return ticks

    def sell_order(self, symbol: str, amount:int, price="market"):
        """
        Created a sell order for amount. If price != "market", then a sell order
        will be created at price=price
        """

        if price == "market":
            order = self.client.order_market_sell(symbol=symbol, quantity=amount)
        else:
            order = self.client.order_limit_sell(symbol=symbol, quantity=amount,
                                                price=price)
        return order

    def check_balance(self, ):
        """Returns portfolio balance"""

        raise NotImplementedError

    def trade_status(self, symbol:str, trade_id: str):
        """Returns the status for trade with id trade_id"""
        return self.client.get_order(symbol=symbol, orderId=trade_id)

    def cancel_order(self, symbol: str, trade_id: str):
        """Cancel order with trade_id"""
        return self.client.cancel_order(symbol=symbol, orderId=trade_id)

    def account_info(self):
        return self.client.get_account()

    def asset_balance(self, symbol: str):
        return self.client.get_asset_balance(asset=symbol)

    def get_trades(self, symbol: str):
        return self.client.get_my_trades(symbol=symbol)

    def get_trade_fee(self, symbol: str):
        return self.client.get_trade_fee(symbol=symbol)