"""
This module contains various PyTorch models.
"""

import torch
import torchvision.models as models


def save_model(model, path):
    """
    Save the weights of the model to the path.
    :param model: model
    :param path: path
    :return: None
    """
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """
    Load weights of the model from the path.
    :param model: model
    :param path: path
    :return: None
    """
    try:
        model.load_state_dict(torch.load(path))
    except:
        print(f'Failed to load model weights from {path}.')


class RNN(torch.nn.Module):
    """
    Recurrent Neural Network (RNN) model.
    """

    def __init__(self,
                 num_features,
                 num_rows,
                 batch_size,
                 hidden_size,
                 num_layers):
        """
        Initialize the model by setting up the layers.
        :param num_features: number of features
        :param num_rows: number of rows
        :param batch_size: size of the batch
        :param hidden_size: size of the hidden state
        :param num_layers: number of layers
        :return: None
        """
        super(RNN, self).__init__()
        # the number of features i.e. columns in the tensor
        self.num_features = num_features
        # the number of rows in the tensor
        self.num_rows = num_rows
        # the batch size in the tensor
        self.batch_size = batch_size
        # the hidden size
        self.hidden_size = hidden_size
        # the number of hidden layers
        self.num_layers = num_layers
        # create the recurrent neural network
        self.rnn = torch.nn.GRU(input_size=self.num_features,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                batch_first=True)
        # create the initial hidden state
        # hidden has shape num_layers, batch_size, hidden_size
        self.hidden = torch.zeros(self.num_layers,
                                  self.batch_size,
                                  self.hidden_size)
        # create the link layer i.e. the conversion from hidden_size to size_1
        self.link_layer = torch.nn.Linear(self.hidden_size, 1000)
        # create the dense layer i.e. the conversion from size_1 to size_2
        self.dense1 = torch.nn.Linear(1000, 500)
        # create the dense layer i.e. the conversion from size_2 to size_3
        self.dense2 = torch.nn.Linear(500, 100)
        # create the dense layer i.e. the conversion from size_3 to size_4
        self.dense3 = torch.nn.Linear(100, 13)
        # # create the dense layer i.e. the conversion from size_4 to size_5
        # self.dense4 = torch.nn.Linear(13, 5)

    def forward(self, x):
        """
        Perform a forward pass of our model on some input and
        the hidden state.
        :param x: model input
        :return: model output
        """
        # apply the recurrent neural network
        # x has shape batch_size, num_rows, num_features
        x, self.hidden = self.rnn(x, self.hidden)
        # take only the last row of the output
        x = x[:, -1]
        # detach the hidden layer to prevent further back-propagating
        # i.e. fix the vanishing gradient problem
        self.hidden = self.hidden.detach()
        # apply the link layer
        # x has shape batch_size, num_rows, hidden_size
        x = self.link_layer(x)
        x = torch.nn.functional.relu(x)
        # apply four fully-connected linear layers with relu activation function
        # x has shape batch_size, num_rows, size_1
        x = self.dense1(x)
        x = torch.nn.functional.relu(x)
        # x has shape batch_size, num_rows, size_2
        x = self.dense2(x)
        x = torch.nn.functional.relu(x)
        # x has shape batch_size, num_rows, size_3
        x = self.dense3(x)
        # x = torch.nn.functional.relu(x)
        # # x has shape batch_size, num_rows, size_4
        # x = self.dense4(x)
        # # x has shape batch_size, num_rows, size_5
        return x


class CNN(torch.nn.Module):
    """
    Convolution Neural Network (CNN) model.
    """

    def __init__(self):
        """
        Initialize the model by setting up the layers.
        :return: None
        """
        super(CNN, self).__init__()
        # create the convolutional neural network
        self.cnn = models.resnet18(pretrained=True, progress=False)
        # create the dense layer i.e. the conversion from size_1 to size_2
        self.dense1 = torch.nn.Linear(1000, 500)
        # create the dense layer i.e. the conversion from size_2 to size_3
        self.dense2 = torch.nn.Linear(500, 100)
        # create the dense layer i.e. the conversion from size_3 to size_4
        self.dense3 = torch.nn.Linear(100, 13)
        # # create the dense layer i.e. the conversion from size_4 to size_5
        # self.dense4 = torch.nn.Linear(13, 5)

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        :param x: model input
        :return: model output
        """
        # apply the convolutional neural network
        # x has shape batch_size, channels, rows, columns
        x = self.cnn(x)
        # apply four fully-connected linear layers with relu activation function
        # x has shape batch_size, size_1
        x = self.dense1(x)
        x = torch.nn.functional.relu(x)
        # x has shape batch_size, size_2
        x = self.dense2(x)
        x = torch.nn.functional.relu(x)
        # x has shape batch_size, size_3
        x = self.dense3(x)
        # x = torch.nn.functional.relu(x)
        # # x has shape batch_size, size_4
        # x = self.dense4(x)
        # # x has shape batch_size, size_5
        return x


class RNNCNN(torch.nn.Module):
    """
    Recurrent Neural Network/Convolution Neural Network (RNNCNN) model.
    """

    def __init__(self,
                 num_features,
                 num_rows,
                 batch_size,
                 hidden_size,
                 num_layers):
        """
        Initialize the model by setting up the layers.
        :param num_features: number of features
        :param num_rows: number of rows
        :param batch_size: size of the batch
        :param hidden_size: size of hidden state
        :param num_layers: number of layers
        :return: None
        """
        super(RNNCNN, self).__init__()
        # the number of features i.e. columns in the rnn_x tensor
        self.num_features = num_features
        # the number of rows in the rnn_x tensor
        self.num_rows = num_rows
        # the batch size in the rnn_x tensor
        self.batch_size = batch_size
        # the hidden size
        self.hidden_size = hidden_size
        # the number of hidden layers
        self.num_layers = num_layers
        # create the recurrent neural network
        self.rnn = torch.nn.GRU(input_size=self.num_features,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                batch_first=True)
        # create the initial hidden state
        # hidden has shape num_layers, batch_size, hidden_size
        self.hidden = torch.zeros(self.num_layers,
                                  self.batch_size,
                                  self.hidden_size)
        # create the link layer i.e. the conversion from hidden_size to size_1
        self.link_layer = torch.nn.Linear(self.hidden_size, 1000)
        # create the convolutional neural network
        self.cnn = models.resnet18(pretrained=True, progress=False)
        # create the dense layer i.e. the conversion from size_1 to size_2
        self.dense1 = torch.nn.Linear(1000, 500)
        # create the dense layer i.e. the conversion from size_2 to size_3
        self.dense2 = torch.nn.Linear(500, 100)
        # create the dense layer i.e. the conversion from size_3 to size_4
        self.dense3 = torch.nn.Linear(100, 13)
        # # create the dense layer i.e. the conversion from size_4 to size_5
        # self.dense4 = torch.nn.Linear(13, 5)

    def forward(self, x, weight=.5):
        """
        Perform a forward pass of our model on some input and
        the hidden state.
        :param x: model input
        :param weight: weight
        :return: model output
        """
        rnn_x, cnn_x = x
        # apply the recurrent neural network
        # rnn_x has shape batch_size, num_rows, num_features
        rnn_x, self.hidden = self.rnn(rnn_x, self.hidden)
        # take only the last row of the output
        rnn_x = rnn_x[:, -1]
        # detach the hidden layer to prevent further back-propagating
        # i.e. fix the vanishing gradient problem
        self.hidden = self.hidden.detach()
        # apply the link layer
        # rnn_x has shape batch_size, num_rows, hidden_size
        rnn_x = self.link_layer(rnn_x)
        rnn_x = torch.nn.functional.relu(rnn_x)
        # apply the convolutional neural network
        # cnn_x has shape batch_size, channels, rows, columns
        cnn_x = self.cnn(cnn_x)
        # apply a weighted average
        # rnn_x has shape batch_size, num_rows, size_1
        # cnn_x has shape batch_size, size_1
        x = (rnn_x * weight).add(cnn_x * (1. - weight))
        # apply four fully-connected linear layers with relu activation function
        # x has shape batch_size, num_rows, size_1
        x = self.dense1(x)
        x = torch.nn.functional.relu(x)
        # x has shape batch_size, num_rows, size_2
        x = self.dense2(x)
        x = torch.nn.functional.relu(x)
        # x has shape batch_size, num_rows, size_3
        x = self.dense3(x)
        # x = torch.nn.functional.relu(x)
        # # x has shape batch_size, num_rows, size_4
        # x = self.dense4(x)
        # # x has shape batch_size, num_rows, size_5
        return x

    def load_cnn_weights(self, cnn):
        """
        Load the convolutional neural network weights.
        :param cnn: convolutional neural network
        :return: None
        """
        cnn_params = cnn.named_parameters()
        gru_cnn_params = dict(self.cnn.named_parameters())
        for name, cnn_param in cnn_params:
            if name in gru_cnn_params:
                gru_cnn_params[name].data.copy_(cnn_param.data)

    def load_rnn_weights(self, rnn):
        """
        Load the recurrent neural network weights.
        :param rnn: recurrent neural network
        :return: None
        """
        rnn_params = rnn.named_parameters()
        gru_rnn_params = dict(self.rnn.named_parameters())
        for name, rnn_param in rnn_params:
            if name in gru_rnn_params:
                gru_rnn_params[name].data.copy_(rnn_param.data)
