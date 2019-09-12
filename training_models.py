from helpers.charting_tools import Charting
from helpers.data_processing import add_ti
from helpers.saving_models import load_model, save_model
from helpers.datasets import DFTimeSeriesDataset, OCHLVDataset
from torch.utils.data import DataLoader, Dataset
from BookWorm import BookWorm, BinanceWrapper
from PIL import Image
from tqdm import tqdm_notebook as tqdm
import warnings
import torch
import os
import shutil
import pandas as pd
torch.backends.cudnn.benchmark = True


from models.GRU.GRU import GRUnet, CNN

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 5}

def price_returns(df, num_rows=30, num_into_fut=5, step=10):
    """Get the return percentage of a candlestick further into the future for a list of labels"""
    df = df.reset_index(drop=True)
    labels = []
    
    for row_i in range(0, df.shape[0] - num_rows - num_into_fut, step):
        # skip all iterations while row_i < num_rows since nothing yet to create a label for
        if row_i <= num_rows: continue
        
        vf, vi = df['close'][row_i+num_into_fut], df['close'][row_i]
        price_return = (vf - vi) / vi
        labels.append(price_return)
    return labels

def split_candles(df, num_rows=30, step=10):
    """Split a DataFrame of candlestick data into a list of smaller DataFrames each with num_rows rows"""
    
    slices = []
    
    for row_i in range(0, df.shape[0] - num_rows, step):
        small_df = df.iloc[row_i:row_i+num_rows, :]
        slices.append(small_df)
        
    return slices

def _train(train_dl, model, optim, error_func):
    losses = []
    
    for batch, labels in train_dl:    
        batch, labels = batch.cuda().float(), labels.cuda().float()
        # set model to train mode
        model.train()
        
        # clear gradients
        model.zero_grad()
        
        output = model(batch)
        loss = error_func(output, labels)
        loss.backward()
        optim.step()
        
        
        losses.append(loss)
        
    return round(float(sum(losses) / len(losses)), 6)

def _valid(valid_dl, model, optim, error_func):
    with torch.set_grad_enabled(False):
        losses = []

        for batch, labels in valid_dl:
            batch, labels = batch.cuda().float(), labels.cuda().float()
            
            # set to eval mode
            model.eval()
            
            # clear gradients
            model.zero_grad()

            output = model(batch)
            loss = error_func(output, labels)

            losses.append(loss)
        
    return round(float(sum(losses) / len(losses)), 6)

def _test(test_dl, model, optim, error_func):
    with torch.set_grad_enabled(False):
        losses = []

        for batch, labels in test_dl:
            batch, labels = batch.cuda().float(), labels.cuda().float()
            
            # set to eval mode
            model.eval()
            
            # clear gradients
            model.zero_grad()

            output = model(batch)
            loss = error_func(output, labels)

            losses.append(loss)
        
    return round(float(sum(losses) / len(losses)), 6)

def RMSE(x, y):
            
            #TODO automate this without model_name
            # have to squish x into a rank 1 tensor with batch_size length with the outputs we want
            if len(list(x.size())) == 2:
                 # torch.Size([64, 1])
                x = x.squeeze(1)
            elif len(list(x.size())) == 3:
                # torch.Size([64, 30, 1])
                x = x[:, 29, :] # take only the last prediction from the 30 time periods in our matrix
                x = x.squeeze(1)
    
            mse = torch.nn.MSELoss()
            return torch.sqrt(mse(x, y))

def train(model, optim, error_func, num_epochs, train_dl, valid_dl, output_name, test_dl=None):
    """Train a PyTorch model with optim as optimizer strategy"""
    
    for epoch_i in range(num_epochs):     
        # forward and backward passes of all batches inside train_gen
        train_loss = _train(train_dl, model, optim, error_func)
        valid_loss = _valid(valid_dl, model, optim, error_func)
        
        # run on test set if provided
        if test_dl: test_output = _test(test_dl, model, optim, error_func)
        else: test_output = "no test selected"
        print("train loss: {}, valid loss: {}, test output: {}".format(train_loss, valid_loss, test_output))

def train_on_df(model, candles_df, lr, num_epochs, output_name, needs_image):
    torch.backends.cudnn.benchmark = True

    # simple data cleaning 
    candles = candles_df.reset_index(drop=True)
    candles = candles.ffill()
    candles = candles.astype(float)

    candles = add_ti(candles)

    labels = price_returns(candles)
    inputs = split_candles(candles)
    # remove all inputs without a label
    inputs = inputs[len(inputs)-len(labels):]

    # calculate s - index of train/valid split
    s = int(len(inputs) * 0.7)

    if needs_image:
        train_ds = OCHLVDataset(inputs[:s], labels[:s])
        valid_ds = OCHLVDataset(inputs[s:], labels[s:])
    else:
        train_ds = DFTimeSeriesDataset(inputs[:s], labels[:s])
        valid_ds = DFTimeSeriesDataset(inputs[s:], labels[s:])

    train_dl = DataLoader(train_ds, **params)
    valid_dl = DataLoader(valid_ds, **params)

    optim = torch.optim.Adam(model.parameters(), lr)

    train(model, optim, RMSE, num_epochs, train_dl, valid_dl, output_name)

gru = GRUnet(11, 30, 64, 500, 3).cuda()
candles = pd.read_csv('bitcoin1m.csv')

candles = candles[len(candles)-50000:]

train_on_df(gru, candles, 1e-3, 5, 'gru_w', needs_image=False)