"""
This module contains testing for dataprocessing.
"""

import datetime
import os
import random

import numpy as np
import pandas as pd

from .dataprocessing import \
    rename_columns, \
    filter_columns, \
    format_date_column, \
    format_dataframe, \
    add_technical_indicators, \
    create_smaller_dataframes, \
    create_relative_dataframe, \
    create_logarithm_dataframe, \
    create_rnn_input, \
    create_cnn_input


def test_rename_columns():
    """
    Tests rename_columns.
    """
    df = pd.DataFrame([[20210101, 0, 0, 0, 0, 0]],
                      columns=['<DATE>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>'])
    df = rename_columns(df)
    actual = df.columns
    expected = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for column in actual:
        assert column in expected


def test_filter_columns():
    """
    Tests filter_columns.
    """
    df = pd.DataFrame([[20210101, 0, 0, 0, 0, 0, 0]],
                      columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'N/A'])
    df = filter_columns(df)
    actual = df.columns
    expected = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for column in actual:
        assert column in expected


def test_format_date_column():
    """
    Tests format_date_column.
    """
    df = pd.DataFrame([[20210101]],
                      columns=['Date'])
    df = format_date_column(df)
    actual = df['Date'][0]
    expected = datetime.datetime(2021, 1, 1)
    assert actual == expected


def test_format_dataframe():
    """
    Tests format_dataframe.
    """
    df = pd.DataFrame([[20210101, 0, 0, 0, 0, 0]],
                      columns=['<DATE>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>'])
    df = format_dataframe(df)
    assert df.index[0] == datetime.datetime(2021, 1, 1)
    actual = df.columns
    expected = ['Open', 'High', 'Low', 'Close', 'Volume']
    for column in actual:
        assert df[column][0] == 0
        assert column in expected


def test_add_technical_indicators():
    """
    Tests add_technical_indicators.
    """
    candles = [pd.read_csv(os.path.join(root, file))
               for root, _, files in os.walk('data')
               for file in files]
    for df in candles:
        df = rename_columns(df)
        df = add_technical_indicators(df)
        _, columns = df.shape
        assert columns == 18


def test_create_smaller_dataframes():
    """
    Tests create_smaller_dataframes.
    """
    df = pd.DataFrame([[55.55], [92.57]],
                      columns=['Close'])
    dfs = create_smaller_dataframes(df, 1, 1)
    assert dfs[0]['Close'][0] == 55.55
    assert dfs[1]['Close'][0] == 92.57


def test_create_relative_dataframe():
    """
    Tests create_relative_dataframe.
    """
    df = pd.DataFrame([[55.55], [92.57], [99.52]],
                      columns=['Close'])
    df = create_relative_dataframe(df)
    assert df['Close'][0] == 55.55
    assert df['Close'][1] == 92.57 / 55.55
    assert df['Close'][2] == 99.52 / 92.57


def test_create_logarithm_dataframe():
    """
    Tests create_logarithm_dataframe.
    """
    df = pd.DataFrame([[55.55], [92.57], [99.52]],
                      columns=['Close'])
    df = create_logarithm_dataframe(df)
    assert df['Close'][0] == 55.55
    assert df['Close'][1] == np.log(92.57)
    assert df['Close'][2] == np.log(99.52)


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
