from Gekko import Gekko
from api_wrappers.APIWrapper import APIWrapper
from api_wrappers.BinanceWrapper import BinanceWrapper
from strategies import CNN_Strategy
from Portfolio import Portfolio
from pandas import DataFrame
from models.GRU.GRU import GRUnet
from helpers.datasets import DFTimeSeriesDataset, OCHLVDataset

import torch

class BackTest:

	def __init__(self):
		p = Portfolio()
		self.gekko = Gekko(p)

	def backtest(self, strategy, input_list, label_list):
		for i, inpoote in enumerate(input_list):
			truth = label_list[i]
			strategy.feed_data(inpoote)
			prediction = strategy.predict()
			strategy.update()


		
		
	
		