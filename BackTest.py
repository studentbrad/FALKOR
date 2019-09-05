from Gekko import Gekko
from api_wrappers.APIWrapper import APIWrapper
from api_wrappers.BinanceWrapper import BinanceWrapper
from strategies import CNN_Strategy
from Portfolio import Portfolio
from .helpers.data_processing import *

from pandas import DataFrame

from strategies.example_strategies import ModelStrat
from .helpers.saving_models import save_model, load_model
from models.GRU.GRU import GRUnet

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


bw = BinanceWrapper('5lJ0uGit9PuUxHka3hBWhPmsi7dWyxEwvEntUZFKmm0xfNz3VjHWi5WSr5W1VBJV','BFWVs8ko7Cd4sjdQ9amGJTnToGWy9TbQWIjeorSCj23FGiwFaknzkgLPcrgWrxsw')

gru = GRUnet(num_features=12, num_rows=30, batch_size=64, hidden_size=500, num_layers=5).float()
load_model(gru, 'gru_weights')

strat = ModelStrat(gru)

b = BackTest(bw, strat)