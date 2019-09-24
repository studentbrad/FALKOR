"""This module contains PyTorch models and model operation functions"""

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torchvision.models as models


def save_model(model, path):
    """Save the weights of model to path"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Load weights from path model"""
    try:
        model.load_state_dict(torch.load(path))
    except:
        print("Failed to load model weights from {}.".format(path))

class CNN(nn.Module):
    def __init__(self):
        """Initialize the model by setting up the layers"""
        super(CNN, self).__init__()
        
        # initial layer is resnet
        self.resnet = models.resnet18(pretrained=True, progress=False)
        
        # final fully connected layers
        self.dense1 = nn.Linear(1000, 100)
    
        # output layer
        self.dense2 = nn.Linear(100, 1)
    
    def forward(self, x):
        """Perform a forward pass of our model on some input and hidden state"""
        
        x = self.resnet(x)
        
         # apply three fully-connected Linear layers with ReLU activation function
        x = self.dense1(x)
        x = relu(x)
        
        # output is a size 1 Tensor
        x = self.dense2(x)
                
        return x

class RNN(nn.Module):

    def __init__(self, num_features, num_rows, batch_size, hidden_size, num_layers, eval_mode=False):
        """Initialize the model by setting up the layers"""
        super(RNN, self).__init__()
        
        # initialize information about model
        self.num_features = num_features
        self.num_rows = num_rows
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.eval_mode = eval_mode  
        # RNN-GRU Layer
        self.rnn = nn.GRU(batch_first=True, input_size=self.num_features,
                          hidden_size=self.hidden_size, num_layers = self.num_layers)
        
        # init GRU hidden layer
        self.hidden = self.init_hidden(batch_size=self.batch_size, hidden_size=hidden_size)
        
        # dropout layer
        #self.dropout = nn.Dropout(0.3)
        
        # 3 fully-connected hidden layers - with an output of dim 1
        self.link_layer = nn.Linear(self.hidden_size, 100)
        self.dense1 = nn.Linear(100, 10)
        self.dense2 = nn.Linear(10, 1)
        
    def forward(self, x):
        """Perform a forward pass of our model on some input and hidden state"""
        # GRU layer
        x, self.hidden = self.rnn(x, self.hidden)
        
        # detatch the hidden layer to prevent further backpropagating. i.e. fix the vanishing gradient problem
        self.hidden = self.hidden.detach()
                
        # apply a Dropout layer 
        #x = self.dropout(x)
        
        # pass through the link_layer
        x = self.link_layer(x)
        x = relu(x)
        
        # apply three fully-connected Linear layers with ReLU activation function
        x = self.dense1(x)
        x = relu(x)
        
        x = self.dense2(x)
        
        # output is a size 1 Tensor
        return x
    
    def init_hidden(self, batch_size, hidden_size):
        """Initializes hidden state"""
        
        # Creates initial hidden state for GRU of zeroes
        if self.eval_mode:
            hidden = torch.ones(self.num_layers, self.batch_size, hidden_size).cpu()
        else:
            hidden = torch.ones(self.num_layers, self.batch_size, hidden_size).cuda()
            
        return hidden

class GRUCNN(nn.Module):

    def __init__(self, num_features, num_rows, batch_size, hidden_size, num_layers):
        """Initialize the model by setting up the layers"""
        super(GRUCNN, self).__init__()
        
        # initialize gru and cnn - the full models
        
        # gru model params
        self.num_features = num_features
        self.num_rows = num_rows
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # resnet model
        self.cnn = models.resnet18(pretrained=True, progress=False)
              
        # RNN-GRU model
        self.rnn = nn.GRU(batch_first=True, input_size=self.num_features,
                          hidden_size=self.hidden_size, num_layers=self.num_layers)
        
        # init GRU hidden layer
        self.hidden = self.init_hidden(batch_size=self.batch_size, hidden_size=hidden_size)
        self.gru_output = nn.Linear(self.hidden_size, 100)
        
        # final fully connected layers
        self.link_layer = nn.Linear(self.hidden_size, 100)
        self.dense1 = nn.Linear(100, 10)
        self.dense2 = nn.Linear(10, 1)
        
    
    
    def forward(self, m_input):
        """Perform a forward pass of our model on some input and hidden state"""

        # input is in a tuple (gru_input, cnn_input)
        gru_input, cnn_input = m_input

        # gru
        gru_out, self.hidden = self.rnn(gru_input, self.hidden)
        
        # detatch the hidden layer to prevent further backpropagating. i.e. fix the vanishing gradient problem
        self.hidden = self.hidden.detach()
        
        # pass through linear layer
        gru_out = torch.squeeze(self.gru_output(gru_out))
                
        # cnn
        cnn_out = self.cnn(cnn_input)
        
        # add the outputs of grunet and cnn
        x = gru_out.add(cnn_out)
        
        # feed through final layers

        # apply three fully-connected Linear layers with ReLU activation function
        x = self.dense1(x)
        x = relu(x)
        
        x = self.dense2(x)
        
        # output is a size 1 Tensor     
        return x
    
    def init_hidden(self, batch_size, hidden_size):
        """Initializes hidden state"""
        
        # Creates initial hidden state for GRU of zeroes
        hidden = torch.ones(self.num_layers, self.batch_size, hidden_size).cuda()
            
        return hidden

    def load_cnn_weights(self, cnn):
        cnn_params = cnn.named_parameters()
        gru_cnn_params = dict(self.cnn.named_parameters())
        
        for name, cnn_param in cnn_params:
            if name in gru_cnn_params:
                gru_cnn_params[name].data.copy_(cnn_param.data)
    
    def load_gru_weights(self, gru):
        gru_params = gru.named_parameters()
        gru_cnn_params = dict(self.rnn.named_parameters())
        
        for name, gru_param in gru_params:
            if name in gru_cnn_params:
                gru_cnn_params[name].data.copy_(gru_param.data)
