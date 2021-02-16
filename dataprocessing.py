"""
This module contains all dataprocessing used in training various PyTorch models.
"""

import datetime

import matplotlib.colors as clr
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np

# default dictionary of columns remapping
default_columns_remap = {
    '<DATE>': 'Date',
    '<OPEN>': 'Open',
    '<HIGH>': 'High',
    '<LOW>': 'Low',
    '<CLOSE>': 'Close',
    '<VOL>': 'Volume'
}

# default list of columns
default_columns = list(default_columns_remap.values())


def rename_columns(df, columns_remap=None):
    """
    Takes a dataframe and renames the columns.
    :param df: dataframe
    :param columns_remap: dictionary of columns remapping
    :return: dataframe
    """
    # if the dictionary of columns remapping is not given, use the default
    if columns_remap is None:
        columns_remap = default_columns_remap
    # rename the columns
    df = df.rename(columns_remap, axis=1)
    return df


def filter_columns(df, columns=None):
    """
    Takes a dataframe and filters the columns.
    :param df: dataframe
    :param columns: list of columns
    :return: dataframe
    """
    # if the list of columns is not given, use the default
    if columns is None:
        columns = default_columns
    # filter the columns
    df = df.filter(items=columns)
    return df


def format_date_column(df, column='Date', datetime_format='%Y%m%d'):
    """
    Takes a dataframe and formats the date column.
    :param df: dataframe
    :param column: column
    :param datetime_format: format
    :return: dataframe
    """
    # format the date column as a string
    df[column] = df[column].astype(str)
    # format the date column as a datetime object
    df[column] = df[column].apply(lambda x: datetime.datetime.strptime(x, datetime_format))
    return df


def format_dataframe(df):
    """
    Takes a dataframe and formats it.
    :param df: dataframe
    :return: dataframe
    """
    # rename the columns
    df = rename_columns(df)
    # filter the columns
    df = filter_columns(df)
    # format the date column
    df = format_date_column(df)
    # set the date column as the index
    df = df.set_index('Date')
    return df


def add_technical_indicators(df):
    """
    Takes a dataframe and adds technical indicators.
    :param df: dataframe
    :return: dataframe
    """
    # calculate the simple moving average over a 20 day windows
    rolling20 = df['Close'].rolling(window=20)
    mean20 = rolling20.mean()
    df['SMA20'] = mean20
    # calculate the simple moving average over a 50 day window
    rolling50 = df['Close'].rolling(window=50)
    mean50 = rolling50.mean()
    df['SMA50'] = mean50
    # calculate the exponential moving average for a span of 13 days
    df['EMA13'] = df['Close'].ewm(span=13, adjust=False).mean()
    # calculate the on-balance volume
    df['OBV'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'],
                         np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0)).cumsum()
    # calculate the short term bollinger bands
    std20 = rolling20.std()
    df['BBU20'] = mean20 + std20 * 2
    df['BBL20'] = mean20 - std20 * 2
    # calculate the long term bollinger bands
    std50 = rolling50.std()
    df['BBU50'] = mean50 + std50 * 2.5
    df['BBL50'] = mean50 - std50 * 2.5
    return df


def split_dataframe(df, window, step):
    """
    Takes a dataframe of size n x m
    and creates dataframes of size window x m
    with a moving start index index_i, index_(i + 1), ..., index_n where index_(i + 1) = index_i + step.
    :param df: dataframe
    :param window: size of the window
    :param step: size of the step
    :return: dataframes
    """
    rows = df.shape[0]  # total number of rows in the dataframe
    dfs = [df.iloc[i: i + window, :].reset_index(drop=True) for i in range(0, rows, step) if i + window <= rows]
    return dfs


def stack_dataframe(df):
    """
    Takes a dataframe of size n x m with rows r_i, r_(i + 1), ..., r_n
    and creates a dataframe of size (n - 1) x m
    with relative log values where r_(i + 1) = log(r_(i + 1) / r_i).
    :param df: dataframe
    :return: dataframe, first row
    """
    rows = df.shape[0]  # total number of rows in the dataframe
    for i in range(rows - 1, 0, -1):
        df.iloc[i] = df.iloc[i] / df.iloc[i - 1]
    df.iloc[1:] = df.iloc[1:].apply(np.log)
    return df


def create_rnn_input(df):
    """
    Takes a dataframe and creates an Recurrent Neural Network (RNN) input.
    :param df: dataframe
    :return: rnn input
    """
    # format the dataframe
    df = format_dataframe(df)
    # add technical indicators
    df = add_technical_indicators(df)
    # stack the dataframe
    df = stack_dataframe(df)
    # move the dataframe to a numpy array
    array = np.array(df)
    return array


def create_cnn_input(df):
    """
    Takes a dataframe and creates a Convolutional Neural Network (CNN) input.
    :param df: dataframe
    :return: cnn input
    """
    # format the dataframe
    df = format_dataframe(df)
    # add technical indicators
    df = add_technical_indicators(df)
    # create the marketcolors
    marketcolors = mpf.make_marketcolors(up='green',
                                         down='red',
                                         volume='blue')
    # create the style
    style = mpf.make_mpf_style(marketcolors=marketcolors)
    # create the addplot for technical indicators
    columns = {'SMA20': '#dfff10',  # lime
               'SMA50': '#ffbf00',  # tangerine
               'EMA13': '#ff7f50',  # dark tangerine
               'BBU20': '#de3163',  # dark pink
               'BBL20': '#9fe2bf',  # cyan
               'BBU50': '#40e0d0',  # dark cyan
               'BBL50': '#6495ed'}  # sky blue
    # OTHER: '#ccccff',  # light purple
    addplot = [mpf.make_addplot(df[column],
                                color=clr.to_rgb(color),
                                width=1)
               for column, color in columns.items()]
    # create the figure
    fig, _ = mpf.plot(df,
                      type='candle',
                      style=style,
                      volume=True,
                      addplot=addplot,
                      returnfig=True,
                      axisoff=True)
    # set the figure size in pixels
    dpi = fig.get_dpi()
    fig.set_size_inches(224 / dpi, 224 / dpi)
    # draw the figure
    fig.canvas.draw()
    # move the figure to a numpy array
    array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # reshape the numpy array to (height, width, channel)
    array = array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # move the rgb dimension to the start for pytorch compatibility
    array = np.moveaxis(array, 2, 0)
    # show the plot
    # plt.show()
    # close the figure
    plt.close(fig)
    return array
