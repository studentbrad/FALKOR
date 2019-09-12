from helpers.charting_tools import Charting
from helpers.data_processing import add_ti, clean_candles_df, split_candles, price_returns
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

from models.GRU.GRU import GRUnet
from models.CNN.CNN import CNN

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 5}

def _train(train_dl, model, optim, error_func, debug=False):
    batch_loss = 0
    
    for batch, labels in train_dl:    
        batch, labels = batch.cuda().float(), labels.cuda().float()
        
        if debug: print("batch[0] __str__: {} labels[0] __str__: {}".format(batch[0], labels[0]))
        # set model to train mode
        model.train()
        
        # clear gradients
        model.zero_grad()
        
        output = model(batch)
        if debug: print("OUTPUT: shape: {} __str__ {}".format(output.shape, output))

        loss = error_func(output, labels)
        if debug: print("LOSS: {}".format(loss.item()))

        loss.backward()
        optim.step()
        
        batch_loss += loss
    print(batch_loss) 
    return round(float(batch_loss), 6)

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

def train(model, optim, error_func, num_epochs, train_dl, valid_dl, test_dl=None, debug=False):
    """Train a PyTorch model with optim as optimizer strategy"""
    
    for epoch_i in range(num_epochs):     
        # forward and backward passes of all batches inside train_gen
        train_loss = _train(train_dl, model, optim, error_func, debug)
        valid_loss = _valid(valid_dl, model, optim, error_func)
        
        # run on test set if provided
        if test_dl is not None: test_output = _test(test_dl, model, optim, error_func)
        else: test_output = "no test selected"
        print("train loss: {}, valid loss: {}, test output: {}".format(train_loss, valid_loss, test_output))

def train_on_df(model, candles_df, lr, num_epochs, needs_image, debug):
    torch.backends.cudnn.benchmark = True
    
    print('cleaning data')
    # simple data cleaning 
    candles = clean_candles_df(candles_df)
    
    print('adding technical indicators')
    candles = add_ti(candles)
    
    # remove time column
    candles = candles.drop('time', axis=1).reset_index(drop=True)
    
    print('creating input and label lists')
    labels = price_returns(candles)
    inputs = split_candles(candles)
    # remove all inputs without a label
    inputs = inputs[len(inputs)-len(labels):]

    # calculate s - index of train/valid split
    s = int(len(inputs) * 0.7)
    
    print('creating Datasets and DataLoaders')

    if needs_image:
            train_ds = OCHLVDataset(inputs[:s], labels[:s])
            valid_ds = OCHLVDataset(inputs[s:], labels[s:])
    else:
        train_ds = DFTimeSeriesDataset(inputs[:s], labels[:s])
        valid_ds = DFTimeSeriesDataset(inputs[s:], labels[s:])

    train_dl = DataLoader(train_ds, drop_last=True, **params)
    valid_dl = DataLoader(valid_ds, drop_last=True, **params)

    optim = torch.optim.Adam(model.parameters(), lr)
    
    print('commencing training')
    train(model, optim, RMSE, num_epochs, train_dl, valid_dl, debug)


def run_train():
    model = GRUnet(11, 30, 64, 500, 3).cuda()
    candles = pd.read_csv('bitcoin1m.csv')

    candles = candles[len(candles)-500000:]

    train_on_df(model, candles, 1e-3, 5, needs_image=False, debug=True)

    save_model(model, 'gru_w')
