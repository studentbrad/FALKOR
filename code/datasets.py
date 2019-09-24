"""This module contains all Datasets used in training various PyTorch models"""

import numpy as np
from .data_processing import minmaxnorm
from .charting import chart_to_arr
from torch.utils.data import Dataset

class DFDataset(Dataset):
	"""Dataset with inputs of DataFrame of OCHLV+tech_inds, outputs of normalized np arrays"""

	def __init__(self, inputs, labels):
		self.inputs, self.labels = [], labels

		for item in inputs:
			self.inputs.append( DFDataset.format(item) )

		self.c = 1 # one label
	
	def __len__(self):
		return len(self.inputs)
	
	def __getitem__(self, i):
		return self.inputs[i], self.labels[i]

	@staticmethod
	def df_to_arr(df):
		return np.array(df)

	@staticmethod
	def normalize_df(df, norm_func):
		"""Normalize df with norm_func"""
		# apply norm_func on columns
		df = df.apply(norm_func, axis=0)
		return df

	@staticmethod
	def drop_time_col(df):
		# drop time column
		return df.drop('time', axis=1)
	
	@staticmethod
	def format(df):
		"""Combine all processing steps from dataframe input to output for training"""

		# normalize with minmax
		x = DFDataset.normalize_df(df, minmaxnorm)
		x = DFDataset.drop_time_col(x)
		x = DFDataset.df_to_arr(x)
		return x

class ChartImageDataset(Dataset):
	"""Dataset with input of OCHLV+tech_inds DataFrames, output of ChartImage numpy arrays"""

	def __init__(self, inputs, labels):
		self.inputs, self.labels = [], labels

		for item in inputs:
			self.inputs.append( ChartImageDataset.format(item) )

		self.c = 1 # one label

	def __len__(self):
		return len(self.inputs)
		
	def __getitem__(self, i):
		return self.inputs[i], self.labels[i]

	@staticmethod
	def df_to_chart_arr(df):
		return chart_to_arr(df)

	@staticmethod
	def format(df):
		"""Combine all processing steps from dataframe input to output for training"""
		x = ChartImageDataset.df_to_chart_arr(df)
		return x
