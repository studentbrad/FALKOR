"""
This module contains all training for network models.
"""

import os

import pandas as pd
import torch
from sklearn.utils import shuffle

from dataprocessing import \
    create_smaller_dataframes, \
    create_rnn_input_label, \
    create_cnn_input_label
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
        model.train()
        model.zero_grad()
        outputs = model(inputs)
        loss = error_func(outputs, labels)
        losses.append(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    loss = round(float(sum(losses)) / len(losses), 6)
    return loss


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
            model.eval()
            model.zero_grad()
            outputs = model(inputs)
            loss = error_func(outputs, labels)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            losses.append(loss)
    loss = round(float(sum(losses)) / len(losses), 6)
    return loss


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
        create_input_label = create_rnn_input_label
    elif model_type == 'CNN':
        create_input_label = create_cnn_input_label
    else:
        raise ValueError
    nn_inputs = []
    nn_labels = []
    for candle in candles:
        dfs = create_smaller_dataframes(candle,
                                        window=100,
                                        step=100)
        for df in dfs:
            nn_input, nn_label = create_input_label(df, drop_nan=True)
            nn_inputs.append(nn_input)
            nn_labels.append(nn_label)
    # only take a portion of inputs for training
    nn_inputs, nn_labels = shuffle(nn_inputs, nn_labels)
    portion = .7
    i = int(portion * len(nn_inputs))
    # train_ds = ServeDataset(nn_inputs[:i], nn_labels[:i])
    # valid_ds = ServeDataset(nn_inputs[i:], nn_labels[i:])
    # train_dl = torch.utils.data.DataLoader(train_ds,
    #                                        batch_size=1,
    #                                        shuffle=True,
    #                                        num_workers=1)
    # valid_dl = torch.utils.data.DataLoader(valid_ds,
    #                                        batch_size=1,
    #                                        shuffle=True,
    #                                        num_workers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # train(model, optimizer, torch.nn.MSELoss(), num_epochs, train_dl, valid_dl)
    for i in range(len(nn_inputs)):
        nn_inputs[i] = torch.Tensor(nn_inputs[i])
    for i in range(len(nn_labels)):
        nn_labels[i] = torch.Tensor(nn_labels[i])
    for epoch_i in range(num_epochs):
        losses = []
        for nn_input, nn_label in zip(nn_inputs, nn_labels):
            nn_input, nn_label = nn_input.float(), nn_label.float()
            nn_input = torch.clamp(nn_input, 0., 1000.)
            nn_label = torch.clamp(nn_input, 0., 1000.)
            model.train()
            model.zero_grad()
            nn_output = model(nn_input.unsqueeze(0))
            loss = torch.nn.MSELoss()(nn_output, nn_label.unsqueeze(0))
            losses.append(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        loss = round(float(sum(losses)) / len(losses), 6)
        print(f'loss: {loss}')


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
    num_epochs = 40
    lr = 1e-3
    if model == 'RNN':
        train_rnn(directory, model_path, num_epochs, lr)
    elif model == 'CNN':
        train_cnn(directory, model_path, num_epochs, lr)


if __name__ == '__main__':
    main()
