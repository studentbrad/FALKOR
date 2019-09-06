from Gekko import Gekko
from api_wrappers.APIWrapper import APIWrapper
from api_wrappers.BinanceWrapper import BinanceWrapper
from strategies import CNN_Strategy
from Portfolio import Portfolio
from pandas import DataFrame

from models.GRU.GRU import GRUnet
from helpers.datasets import DFTimeSeriesDataset, ChartImageDataset

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
		
		df = candles_df.reset_index(drop=True)
		
		for i in range(orig_row_num - split_row_num):
			curr_prices.append( df.loc[df.index[i], 'close'] )
			fut_prices.append( df.loc[df.index[i+split_row_num], 'close'] )

		return curr_prices, fut_prices
	
	def price_returns(self, df, num_rows=30, num_into_fut=5, step=10):
		labels = []
	
		for row_i in range(0, df.shape[0] - num_rows - num_into_fut, step):
			# skip all iterations while row_i < num_rows since nothing yet to create a label for
			if row_i <= num_rows: continue
		
			vf, vi = df['close'][row_i+num_into_fut], df['close'][row_i]
			price_return = (vf - vi) / vi
			labels.append(price_return)
		return labels
	

	   

	def profit_test(self, candles_df, model_type):
		"""
		Runs a profitability test of model on historical data of candles_df. 
		By simulating a trade at every model signal, we sum profits/losses made from
		all trade signals generated. Returns string of profit stats
		"""
		candles_df = candles_df.drop('time', axis=1).reset_index(drop=True)
		price_returnz = self.price_returns(candles_df)
		
		candles_df = candles_df.apply(normalize_series, axis=0)
		split_candles = self._split_into_periods(candles_df, 30)

		split_candles = split_candles[:len(price_returnz)]
		
		if model_type == "gru":
			dataset = DFTimeSeriesDataset(split_candles, price_returnz)
		if model_type == "cnn":
			chart_images = create_charts(split_candles, "images/")
			dataset = ChartImageDataset(chart_images, price_returnz)
		corr_preds, incorr_preds = self.gekko.backtest(self.strategy, dataset)
		return "Correct on {}, Incorrect on {}".format(corr_preds, incorr_preds)
