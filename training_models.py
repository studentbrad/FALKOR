from helpers.charting_tools import Charting
from helpers.data_processing import add_ti
from helpers.saving_models import load_model, save_model
from helpers.datasets import DFTimeSeriesDataset, ChartImageDataset
from torch.utils.data import *
from BookWorm import BookWorm, BinanceWrapper
from PIL import Image
from tqdm import tqdm_notebook as tqdm
import warnings
import torch
import os
import shutil
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 5}

def create_charts(candles_sliced, save_path):
    """Create a chart image for each in sliced_candles and return a list of paths to those images"""
    
    # turn off stupid warnings
    warnings.filterwarnings("ignore")

    try: os.mkdir(save_path)
    except: pass

    i = 0
    paths_to_images = []
    for small_df in tqdm(candles_sliced):
        chart = Charting(small_df, 'time', 'close')
        
        path = save_path + 'chart_{}.png'.format(i)
        chart.chart_to_image(path)
        paths_to_images.append(path)
        i += 1

    warnings.filterwarnings("default")
    return paths_to_images        

def split_candles(df, num_rows=30, step=10):
    """Split a DataFrame of candlestick data into a list of smaller DataFrames each with num_rows rows"""
    
    slices = []
    
    for row_i in range(0, df.shape[0] - num_rows, step):
        small_df = df.iloc[row_i:row_i+num_rows, :]
        slices.append(small_df)
        
    return slices

def get_price_returns(df, num_rows=30, num_into_fut=5, step=10):
    labels = []
    
    for row_i in range(0, df.shape[0] - num_rows - num_into_fut, step):
        # skip all iterations while row_i < num_rows since nothing yet to create a label for
        if row_i <= num_rows: continue
        
        vf, vi = df['close'][row_i+num_into_fut], df['close'][row_i]
        price_return = (vf - vi) / vi
        labels.append(price_return)
    return labels

def normalize_series(ser):
    return (ser-ser.min())/(ser.max()-ser.min())

def normalize_df(df):
    df = df.drop('time', axis=1).reset_index(drop=True)
    return df.apply(normalize_series, axis=0)

def _train(train_gen, model, optim, error_func):
    losses = []
    
    for batch, labels in train_gen:    
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

def _valid(valid_gen, model, optim, error_func):
    with torch.set_grad_enabled(False):
        losses = []

        for batch, labels in valid_gen:
            batch, labels = batch.cuda().float(), labels.cuda().float()
            
            # set to eval mode
            model.eval()
            
            # clear gradients
            model.zero_grad()

            output = model(batch)
            loss = error_func(output, labels)

            losses.append(loss)
        
    return round(float(sum(losses) / len(losses)), 6)

def _test(test_gen, model, optim, error_func):
    with torch.set_grad_enabled(False):
        losses = []

        for batch, labels in valid_gen:
            batch, labels = batch.cuda().float(), labels.cuda().float()
            
            # set to eval mode
            model.eval()
            
            # clear gradients
            model.zero_grad()

            output = model(batch)
            loss = error_func(output, labels)

            losses.append(loss)
        
    return round(float(sum(losses) / len(losses)), 6)

def train(model, model_name, optim, num_epochs, train_gen, valid_gen, test_gen=None):
    """Train a PyTorch model with optim as optimizer strategy"""
    
    for epoch_i in range(num_epochs):
        
        
        def RMSE(x, y):
            
            # have to squish x into a rank 1 tensor with batch_size length with the outputs we want
            if model_name == 'resnet':
                 # torch.Size([64, 1])
                x = x.squeeze(1)
            elif model_name == 'gru':
                # torch.Size([64, 30, 1])
                x = x[:, 29, :] # take only the last prediction from the 30 time periods in our matrix
                x = x.squeeze(1)
    
            mse = torch.nn.MSELoss()
            return torch.sqrt(mse(x, y))
        
        
        # forward and backward passes of all batches inside train_gen
        train_loss = _train(train_gen, model, optim, RMSE)
        valid_loss = _valid(valid_gen, model, optim, RMSE)
        
        # run on test set if provided
        if test_gen: test_output = _test(test_gen, model, optim)
        else: test_output = "no test selected"
        print("train loss: {}, valid loss: {}, test output: {}".format(train_loss, valid_loss, test_output))


def train_cnn(model, candles, lr=1e-3, epochs=1):
    candles = add_ti(candles)
    price_returns = get_price_returns(candles)
    # split candles into 30 period and a label
    candles_sliced = split_candles(candles)
    # we need to remove candle slices without a label from candles_sliced
    candles_sliced = candles_sliced[len(candles_sliced)-len(price_returns):]

    paths_to_images = create_charts(candles_sliced, "images/")
    
    # get index of split between validation and test set
    split = 0.7
    s = int(len(candles_sliced) * 0.7)
    while s % params['batch_size'] != 0:
        s += 1

    # create two ChartImageDatasets, split by split, for the purpose of creating a DataLoader for the specific model
    train_ds= ChartImageDataset(paths_to_images[:s], price_returns[:s])
    valid_ds = ChartImageDataset(paths_to_images[s:], price_returns[s:])
    train_dl = DataLoader(train_dl, **params)
    valid_dl= DataLoader(valid_dl, **params)

    train(model, 'resnet', torch.optim.Adam(model.parameters(), lr), epochs, train_dl, valid_dl)

    save_model(model, 'cnn_weights')

    # delete images/ folder
    shutil.rmtree('images/')

def train_gru(model, candles, lr=1e-3, epochs=1):
    candles = add_ti(candles)
    price_returns = get_price_returns(candles)
    candles = normalize_df(candles)
    # split candles into 30 period and a label
    candles_sliced = split_candles(candles)
    # we need to remove candle slices without a label from candles_sliced
    candles_sliced = candles_sliced[len(candles_sliced)-len(price_returns):]

    # get index of split between validation and test set
    split = 0.7
    s = int(len(candles_sliced) * 0.7)
    while s % params['batch_size'] != 0:
        s += 1

    # create two ChartImageDatasets, split by split, for the purpose of creating a DataLoader for the specific model
    train_ds = DFTimeSeriesDataset(candles_sliced[:s], price_returns[:s])
    valid_ds = DFTimeSeriesDataset(candles_sliced[s:], price_returns[s:])
    train_dl = DataLoader(train_dl, **params)
    valid_dl= DataLoader(valid_dl, **params)

    train(model, 'gru', torch.optim.Adam(model.parameters(), lr), epochs, train_dl, valid_dl)
    save_model(model, 'gru_weights')