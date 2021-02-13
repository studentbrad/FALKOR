"""
This module contains testing for datasets.
"""

import numpy as np
import torch

from .datasets import \
    ServeDataset


class TestServeDataset:
    """
    Tests DFDataset.
    """

    @staticmethod
    def test_dataloader():
        """
        Tests the dataloader.
        """
        inputs = np.arange(10)
        labels = np.arange(10)
        dataset = ServeDataset(inputs, labels)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=1,
                                                 shuffle=True,
                                                 num_workers=0)
        for inputs, labels in dataloader:
            assert inputs == labels
