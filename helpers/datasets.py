import math
import pandas as pd
import numpy as np
import torchvision

from PIL import Image
from torch.utils.data import *

from .charting_tools import Charting


def minmaxnorm(ser):
    return (ser-ser.min())/(ser.max()-ser.min())

def normalize_df(df, norm_func):
    # drop time column
    df = df.drop('time', axis=1)

    df = df.apply(norm_func, axis=0)
    df = df.fillna(method='ffill')
    return df

class DFTimeSeriesDataset(Dataset):
    """Dataset for historical timeseries data. 
    self.feature_dfs = [np.Array, np.Array, np.Array, ..., np.Array]
    self.labels = [np.Array, np.Array, np.Array, ..., np.Array]
    """
    def __init__(self, time_series, labels):
        self.time_series, self.labels = time_series, labels
        self.c = 1 # one label
    
    def __len__(self):
        return len(self.time_series)
    
    def __getitem__(self, i):
        df = self.time_series[i]
        df = normalize_df(df, minmaxnorm)
    
        time_series_arr = np.array(df)
        label = np.array(self.labels[i])
       
        # deal with all nans. TODO good code shouldn't have this so handle it elsewhere
        time_series_arr = np.nan_to_num(time_series_arr)
        assert not np.any(np.isnan(time_series_arr))
        return time_series_arr, label

class OCHLVDataset(Dataset):
    """Dataset for a list of dataframes with OCHLV representing a certain period of price"""

    def __init__(self, time_series, labels):
        inputs = []
        for ts in time_series:
            chart = Charting(ts, 'time', 'close')
            arr = chart.chart_to_numpy()
            #image = chart.chart_to_image('chart.png')
            # remove time column
            input_chart = np.delete(arr, 0, axis=1)
            # close chart to save memory
            chart.close_chart()
            inputs.append(input_chart)
        self.time_series, self.labels = inputs, labels
        self.c = 1 # one label

    def __len__(self):
        return len(self.time_series)
        
    def __getitem__(self, i):
        # ignore matplotlib bs
       
        return self.time_series[i], self.labels[i]

class ImagePathDataset(Dataset):
    """ImagePath Dataset"""
    
    def __init__(self, image_paths: list, labels: list):
        """ 
        image_paths: list containing path to image. Order is maintained
        labels: list containing label for each image
        """
        self.image_paths = image_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.labels) 
    
    def __getitem__(self, index):
        """Return Tensor representation of image at images_paths[index]"""
        img = Image.open(self.image_paths[index])
        img.load()
        
        img_tensor = torchvision.transforms.ToTensor()(img)
        
        # remove alpha dimension from png
        img_tensor = img_tensor[:3,:,:]
        return img_tensor, np.array(self.labels[index])
