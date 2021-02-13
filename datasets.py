"""
This module contains all datasets used in training various PyTorch models.
"""

import torch


class ServeDataset(torch.utils.data.Dataset):
    """
    Dataset for serving inputs and labels.
    """

    def __init__(self, inputs, labels):
        """
        Initialization function for the serve dataset.
        :param inputs: dataframe inputs
        :param labels: dataframe labels
        :return: None
        """
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        """
        Length operator for the serve dataset.
        :return: length of the dataset
        """
        return len(self.inputs)

    def __getitem__(self, i):
        """
        Get item operator for the serve dataset.
        :param i: index
        :return: item at the index
        """
        return self.inputs[i], self.labels[i]
