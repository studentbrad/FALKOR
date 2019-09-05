from Gekko import Gekko
from api_wrappers.APIWrapper import APIWrapper
from api_wrappers.BinanceWrapper import BinanceWrapper
from strategies import CNN_Strategy
from Portfolio import Portfolio
from pandas import DataFrame

from strategies.example_strategies import ModelStrat
from models.GRU.GRU import GRUnet

import torch

def save_model(model, path):
    """Save the weights of model to path"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Load weights from path model"""
    try:
        model.load_state_dict(torch.load(path))
    except:
        print("Failed to load model weights from {}.".format(path))

class BackTest:
	"""
	BackTest is used to run FALKOR on historical candlestick data
	and measure the profitability of chosen strategy
	
	Attributes:

	Methods:

	"""
	

	def __init__(self, api_wrapper, strategy):
		"""Initialize BackTest instance"""
		self.api_wrapper = api_wrapper
		self.strategy = strategy

		# Initialize with an empty portfolio
		self.portfolio = Portfolio()
		self.gekko = Gekko(self.portfolio)

	def profit_test(self, model, candles_df):
		"""
		Runs a profitability test of model on historical data of candles_df. 
		By simulating a trade at every model signal, we sum profits/losses made from
		all trade signals generated. Returns string of profit stats
		"""

		model_strat = ModelStrat(model)
		return "{}".format(self.gekko.backtest(model_strat, candles_df))
