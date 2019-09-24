"""This module contains various strategies used for generating trading signals"""

from .models import CNN, RNN, GRUCNN, save_model, load_model
from .datasets import DFDataset, ChartImageDataset

class Strategy:
    """Abstract class representing a Strategy used by Gekko. The child class must create all NotImplemented methods"""

    def feed_data(self, last_candles):
        """ Feed last_candles to strategy """
        raise NotImplementedError    
    def predict(self):
        """ Make a prediction based on last_candles"""
        raise NotImplementedError
    def update(self):
        """Update model however specified"""
        raise NotImplementedError

class CNNStrat(Strategy):
    """Trading strategy for CNN with input of candles_df"""
    def __init__(self, model):
        self.model = model
        self.input = None

    def feed_data(self, last_candles):       
        self.input = ChartImageDataset.format(last_candles)
        
    def predict(self):
        return self.model(self.input)

    def update(self):
        pass

class RNNStrat(Strategy):
    """Trading strategy for RNN with input of candles_df"""
    def __init__(self, model):
        self.model = model
        self.input = None

    def feed_data(self, last_candles):       
        self.input = DFDataset.format(last_candles)
        
    def predict(self):
        return self.model(self.input)

    def update(self):
        pass
    
