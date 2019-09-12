from Gekko import Gekko
from api_wrappers.APIWrapper import APIWrapper
from api_wrappers.BinanceWrapper import BinanceWrapper
from strategies import CNN_Strategy
from Portfolio import Portfolio
from pandas import DataFrame

from models.GRU.GRU import GRUnet
from helpers.datasets import DFTimeSeriesDataset, OCHLVDataset

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
		
def normalize_series(self, ser):
	return (ser-ser.min())/(ser.max()-ser.min())

def create_charts(candles_sliced, save_path):
	"""Create a chart image for each in sliced_candles and return a list of paths to those images"""
	from tqdm import tqdm_notebook as tqdm
	import warnings
	warnings.filterwarnings("ignore")
		
	i = 0
	paths_to_images = []
	for small_df in tqdm(candles_sliced):
		chart = Charting(small_df, 'time', 'close')
		
		path = save_path + 'chart_{}.png'.format(i)
		chart.chart_to_image(path)
		paths_to_images.append(path)	
		i += 1
	return paths_to_images 

class BackTest:

	def __init__():
		p = Portfolio()
		self.gekko = Gekko(p)

	def backtest(strategy, model):
		self.gekko.model_predictions()
		
	
		