"""
This module contains testing for dataprocessing.
"""

import datetime
import os
import random

import numpy as np
import pandas as pd

from .dataprocessing import \
    columns_remap, \
    perform_normalization, \
    format_date_column, \
    add_technical_indicators, \
    create_cdfs_and_fdfs, \
    create_rnn_input, \
    create_cnn_input, \
    create_label


def test_perform_normalization():
    """
    Tests perform_normalization.
    """
    df = pd.DataFrame([[1, 2, 3, 4]], columns=['a', 'b', 'c', 'd'])
    df = perform_normalization(df, ['a', 'd'])
    actual = np.array(df).flatten()
    expected = np.array([0, 2, 3, 1])
    assert actual.size == expected.size
    assert all([a == b for a, b in zip(actual, expected)])


def test_format_date_column():
    """
    Tests format_date_column.
    """
    df = pd.DataFrame([[20210101, 0]], columns=['Date', 'Close'])
    df = format_date_column(df)
    actual = df['Date'][0]
    expected = datetime.datetime(2021, 1, 1)
    assert actual == expected


def test_add_techincal_indicators():
    """
    Tests add_technical_indicators.
    """
    candles = [pd.read_csv(os.path.join(root, file))
               for root, _, files in os.walk('data')
               for file in files]
    for df in candles:
        df = df.rename(columns_remap, axis=1)
        df = add_technical_indicators(df)
        _, columns = df.shape
        assert columns == 18


def test_create_cdfs_and_fdfs():
    """
    Tests create_cdfs_and_fdfs.
    """
    df = pd.DataFrame([[55.55], [92.57]], columns=['Close'])
    cdfs, fdfs = create_cdfs_and_fdfs(df, 1, 1, 1)
    assert cdfs[0]['Close'][0] == 55.55
    assert fdfs[0]['Close'][1] == 92.57


def test_create_rnn_input():
    """
    Tests create_rnn_input.
    """
    candles = [pd.read_csv(os.path.join(root, file))
               for root, _, files in os.walk('data')
               for file in files]
    for df in candles:
        i = random.randint(0, df.shape[0])
        window = random.randint(50, 100)
        df = df.iloc[i:i + window, :]
        array = create_rnn_input(df)
        rows, columns = array.shape
        assert rows == window
        assert columns == 13


def test_create_cnn_input():
    """
    Tests create_cnn_input.
    """
    candles = [pd.read_csv(os.path.join(root, file))
               for root, _, files in os.walk('data')
               for file in files]
    for df in candles:
        i = random.randint(0, df.shape[0])
        window = random.randint(50, 100)
        df = df.iloc[i:i + window, :]
        array = create_cnn_input(df)
        channels, rows, columns = array.shape
        assert channels == 3
        assert rows == 224
        assert columns == 224


def test_create_label():
    """
    Tests create_label.
    """
    candles = [pd.read_csv(os.path.join(root, file))
               for root, _, files in os.walk('data')
               for file in files]
    for df in candles:
        i = random.randint(0, df.shape[0])
        window = random.randint(50, 100)
        df = df.iloc[i:i + window, :]
        array = create_label(df)
        columns, = array.shape
        assert columns == 5
