"""
This module contains all training for network models.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.utils import shuffle

matplotlib.use('agg')

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


def draw_curve(epochs, train_losses, valid_losses):
    """
    Draw the training and validation curves.
    :param epochs: epochs
    :param train_losses: training losses
    :param valid_losses: validation losses
    :return: None
    """
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax0.plot(epochs, train_losses, 'bo-', label='train')
    ax0.plot(epochs, valid_losses, 'ro-', label='valid')
    fig.savefig('train.jpg')


def train(dataloader, model, optimizer, criterion):
    """
    Train a model.
    :param dataloader: dataloader
    :param model: model
    :param optimizer: optimizer
    :param criterion: criterion
    :return: average loss
    """
    losses = []
    for inputs, labels in dataloader:
        inputs, labels = inputs.float(), labels.float()
        model.train()
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-1)
        optimizer.step()
    loss = round(float(sum(losses)) / len(losses), 6)
    return loss


def valid(dataloader, model, criterion):
    """
    Validate a model.
    :param dataloader: dataloader
    :param model: model
    :param criterion: criterion
    :return: average loss
    """
    with torch.no_grad():
        losses = []
        for inputs, labels in dataloader:
            inputs, labels = inputs.float(), labels.float()
            model.eval()
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss)
    loss = round(float(sum(losses)) / len(losses), 6)
    return loss


def train_and_valid(model, optimizer, criterion, num_epochs, train_dl, valid_dl):
    """
    Train and validate a model.
    :param model: model
    :param optimizer: optimizer
    :param criterion: criterion
    :param num_epochs: number of epochs
    :param train_dl: train dataloader
    :param valid_dl: validate dataloader
    :return: None
    """
    epochs = range(1, num_epochs + 1)
    train_losses = []
    valid_losses = []
    for epoch in epochs:
        train_loss = train(train_dl, model, optimizer, criterion)
        valid_loss = valid(valid_dl, model, criterion)
        print(f'Epoch: {epoch:03}... Training Loss: {train_loss:.6f}... Validation Loss: {valid_loss:.6f}')
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
    draw_curve(epochs, train_losses, valid_losses)


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
            nn_input, nn_label = create_input_label(df,
                                                    drop_nan=True,
                                                    fill_nan=True,
                                                    lower=0.,
                                                    upper=1000.)
            nn_inputs.append(nn_input)
            nn_labels.append(nn_label)
    # only take a portion of inputs for training
    nn_inputs, nn_labels = shuffle(nn_inputs, nn_labels)
    portion = .7
    i = int(portion * len(nn_inputs))
    train_ds = ServeDataset(nn_inputs[:i], nn_labels[:i])
    valid_ds = ServeDataset(nn_inputs[i:], nn_labels[i:])
    train_dl = torch.utils.data.DataLoader(train_ds,
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=1)
    valid_dl = torch.utils.data.DataLoader(valid_ds,
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = torch.nn.MSELoss()
    print(f'Training and validating the {model_type} model...')
    train_and_valid(model, optimizer, criterion, num_epochs, train_dl, valid_dl)
    # for i in range(len(nn_inputs)):
    #     nn_inputs[i] = torch.Tensor(nn_inputs[i])
    # for i in range(len(nn_labels)):
    #     nn_labels[i] = torch.Tensor(nn_labels[i])
    # for epoch in range(num_epochs):
    #     train_losses = []
    #     for nn_input, nn_label in zip(nn_inputs[:i], nn_labels[:i]):
    #         nn_input, nn_label = nn_input.float(), nn_label.float()
    #         model.train()
    #         model.zero_grad()
    #         nn_output = model(nn_input.unsqueeze(0))
    #         train_loss = criterion(nn_output, nn_label.unsqueeze(0))
    #         train_losses.append(train_loss)
    #         train_loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-1)
    #         optimizer.step()
    #     train_loss = round(float(sum(train_losses)) / len(train_losses), 6)
    #     valid_losses = []
    #     with torch.no_grad():
    #         for nn_input, nn_label in zip(nn_inputs[i:], nn_labels[i:]):
    #             nn_input, nn_label = nn_input.float(), nn_label.float()
    #             model.eval()
    #             model.zero_grad()
    #             nn_output = model(nn_input.unsqueeze(0))
    #             valid_loss = criterion(nn_output, nn_label.unsqueeze(0))
    #             valid_losses.append(valid_loss)
    #     valid_loss = round(float(sum(valid_losses)) / len(valid_losses), 6)
    #     print(f'Epoch: {epoch + 1:03}... Training Loss: {train_loss:.6f}... Validation Loss: {valid_loss:.6f}')


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
    model = 'CNN'
    model_path = 'models/model.pth'
    num_epochs = 40
    lr = 1e-3
    if model == 'RNN':
        train_rnn(directory, model_path, num_epochs, lr)
    elif model == 'CNN':
        train_cnn(directory, model_path, num_epochs, lr)


if __name__ == '__main__':
    main()
