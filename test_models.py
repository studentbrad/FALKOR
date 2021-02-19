"""
This module contains testing for various PyTorch models.
"""

import torch

from .models import \
    RNN, \
    CNN, \
    RNNCNN


class TestRNN:
    """
    Tests RNN.
    """

    def test_model(self):
        """
        Tests the model inputs/outputs.
        """
        model = RNN(13, 100, 1, 100, 3)
        x = torch.FloatTensor(1, 100, 13)
        output = model(x)
        batch_size, rows, columns = output.size()
        assert batch_size == 1
        assert rows == 100
        assert columns == 13


class TestCNN:
    """
    Tests CNN.
    """

    def test_model(self):
        """
        Tests the model inputs/outputs.
        """
        model = CNN()
        x = torch.FloatTensor(1, 3, 214, 214)
        output = model(x)
        batch_size, size = output.size()
        assert batch_size == 1
        assert size == 13


class TestRNNCNN:
    """
    Tests the RNNCNN.
    """

    def test_model(self):
        """
        Tests the model inputs/outputs.
        """
        model = RNNCNN(13, 100, 1, 100, 3)
        rnn_input = torch.FloatTensor(1, 100, 13)
        cnn_input = torch.FloatTensor(1, 3, 214, 214)
        x = (rnn_input, cnn_input)
        output = model(x)
        batch_size, rows, columns = output.size()
        assert batch_size == 1
        assert rows == 100
        assert columns == 13
