import pandas as pd
from .data_processing import candles_to_inputs_and_labels

class TradingInterface:

    def latest_candles(self):
        raise NotImplementedError

class LiveTrade(TradingInterface):
    def __init__(self, api_wrapper, symbol, interval):
        self.api_wrapper = api_wrapper
        self.symbol = symbol
        self.interval = interval
    
    def latest_candles(self):
        return self.api_wrapper.latest_candles(30, self.symbol, self.interval)

class BackTest(TradingInterface):
    def __init__(self, candles_path):
        candles_df = pd.read_csv(candles_path)
        self.inputs, _ = candles_to_inputs_and_labels(candles_df, num_rows=30, step=10, return_period=5)
    
    def latest_candles(self):
        return self.inputs.pop(0)