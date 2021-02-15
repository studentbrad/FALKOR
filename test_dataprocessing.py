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
    format_date_column, \
    add_technical_indicators, \
    split_dataframe, \
    stack_dataframe, \
    create_rnn_input, \
    create_cnn_input


def test_format_date_column():
    """
    Tests format_date_column.
    """
    df = pd.DataFrame([[20210101, 0]], columns=['Date', 'Close'])
    df = format_date_column(df)
    actual = df['Date'][0]
    expected = datetime.datetime(2021, 1, 1)
    assert actual == expected


def test_add_technical_indicators():
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


def test_split_dataframe():
    """
    Tests split_dataframe.
    """
    df = pd.DataFrame([[55.55], [92.57]], columns=['Close'])
    dfs = split_dataframe(df, 1, 1)
    assert dfs[0].iloc[0]['Close'] == 55.55
    assert dfs[1].iloc[0]['Close'] == 92.57


def test_stack_dataframe():
    """
    Tests stack_dataframe.
    """
    df = pd.DataFrame([[55.55], [92.57], [99.52]], columns=['Close'])
    df = stack_dataframe(df)
    assert df.iloc[0]['Close'] == 55.55
    assert df.iloc[1]['Close'] == np.log(92.57 / 55.55)
    assert df.iloc[2]['Close'] == np.log(99.52 / 92.57)


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
