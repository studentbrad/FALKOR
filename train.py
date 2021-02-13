"""
This module contains all training for network models.
"""

import os
import random

import pandas as pd
import torch

from dataprocessing import \
    create_cdfs_and_fdfs, \
    create_rnn_input, \
    create_cnn_input
from datasets import \
    ServeDataset
from models import \
    CNN, \
    RNN, \
    load_model, \
    save_model


def _train(dataloader, model, optimizer, error_func):
    """
    Internal function for training a model.
    :param dataloader: dataloader
    :param model: model
    :param optimizer: optimizer
    :param error_func: error function
    :return: average loss
    """
    losses = []
    for inputs, labels in dataloader:
        inputs, labels = inputs.float(), labels.float()
        inputs[inputs != inputs] = 0.
        model.train()
        model.zero_grad()
        outputs = model(inputs)
        loss = error_func(outputs, labels)
        losses.append(loss)
        loss.backward()
        optimizer.step()
    return round(float(sum(losses)) / len(losses), 6)


def _valid(dataloader, model, error_func):
    """
    Internal function for validating a model.
    :param dataloader: dataloader
    :param model: model
    :param error_func: error function
    :return: average loss
    """
    with torch.no_grad():
        losses = []
        for inputs, labels in dataloader:
            inputs, labels = inputs.float(), labels.float()
            inputs[inputs != inputs] = 0.
            model.eval()
            model.zero_grad()
            outputs = model(inputs)
            loss = error_func(outputs, labels)
            losses.append(loss)
    return round(float(sum(losses) / len(losses)), 6)


def train(model, optimizer, error_func, num_epochs, train_dl, valid_dl):
    """
    Train a model.
    :param model: model
    :param optimizer: optimizer
    :param error_func: error function
    :param num_epochs: number of epochs
    :param train_dl: train dataloader
    :param valid_dl: validate dataloader
    :return: None
    """
    for epoch_i in range(num_epochs):
        train_loss = _train(train_dl, model, optimizer, error_func)
        valid_loss = _valid(valid_dl, model, error_func)
        print(f'train loss: {train_loss}, valid loss: {valid_loss}')


def rmse(x, y):
    """
    Calculate the root mean squared error (RMSE).
    :param x: predicted value
    :param y: value
    :return: rmse
    """
    mse = torch.nn.MSELoss()
    return torch.sqrt(mse(x, y))


def train_on_directory(directory, model, model_type, num_epochs, lr):
    """
    Train a model on a directory of csv files.
    :param directory: directory of csv files
    :param model: model
    :param model_type: model type ('CNN', 'RNN', ...)
    :param num_epochs: number of epochs
    :param lr: learning rate
    :return: None
    """
    torch.backends.cudnn.benchmark = True
    files = [os.path.join(directory, file)
             for file in os.listdir(directory)
             if os.path.isfile(os.path.join(directory, file))]
    candles = [pd.read_csv(file)
               for file in files]
    if model_type == 'RNN':
        create_input = create_rnn_input
    elif model_type == 'CNN':
        create_input = create_cnn_input
    else:
        raise ValueError
    inputs = []
    labels = []
    for df in candles:
        cdfs, fdfs = create_cdfs_and_fdfs(df,
                                          window=100,
                                          step=100,
                                          return_period=1)
        for cdf, fdf in zip(cdfs, fdfs):
            temp_input = create_input(cdf)
            temp_label = fdfs[-1]
            inputs.extend(temp_input)
            labels.extend(temp_label)
    # only take a portion of inputs for training
    inputs_labels = list(zip(inputs, labels))
    inputs_labels = random.shuffle(inputs_labels)
    inputs, labels = zip(*inputs_labels)
    portion = .7
    i = int(portion * len(inputs))
    train_ds = ServeDataset(inputs[:i], labels[:i])
    valid_ds = ServeDataset(inputs[i:], labels[i:])
    train_dl = torch.utils.data.DataLoader(train_ds,
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=5)
    valid_dl = torch.utils.data.DataLoader(valid_ds,
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=5)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    train(model, optimizer, rmse, num_epochs, train_dl, valid_dl)


def train_rnn(directory, model_path, num_epochs, lr):
    """
    Train a Recurrent Neural Network (RNN).
    :param directory: directory
    :param model_path: model path
    :param num_epochs: number of epochs
    :param lr: learning rate
    """
    model = RNN(13, 100, 1, 100, 3)
    load_model(model, model_path)
    train_on_directory(directory, model, 'RNN', num_epochs, lr)
    save_model(model, model_path)


def train_cnn(directory, model_path, num_epochs, lr):
    """
    Train a Convolutional Neural Network (CNN).
    :param directory: directory
    :param model_path: model_path
    :param num_epochs: number of epochs
    :param lr: learning rate
    """
    model = CNN()
    load_model(model, model_path)
    train_on_directory(directory, model, 'CNN', num_epochs, lr)
    save_model(model, model_path)


def main():
    directory = 'data'
    model = 'RNN'
    model_path = 'model.pth'
    num_epochs = 20
    lr = 1e-3
    if model == 'RNN':
        train_rnn(directory, model_path, num_epochs, lr)
    elif model == 'CNN':
        train_cnn(directory, model_path, num_epochs, lr)


if __name__ == '__main__':
    main()
