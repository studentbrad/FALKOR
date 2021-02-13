"""
This module contains all dataprocessing used in training various PyTorch models.
"""

import datetime

import matplotlib.colors as clr
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np

# columns in the dataframe with renaming
columns_remap = {'<DATE>': 'Date',
                 '<OPEN>': 'Open',
                 '<HIGH>': 'High',
                 '<LOW>': 'Low',
                 '<CLOSE>': 'Close',
                 '<VOL>': 'Volume'}


def perform_normalization(df, columns):
    """
    Takes a dataframe and performs normalization on the dataframe.
    :param df: dataframe
    :param columns: columns
    :return: dataframe
    """
    df_filtered = df.filter(items=columns)
    maximum = df_filtered.max().max()
    minimum = df_filtered.min().min()
    for column in columns:
        df[column] = (df[column] - minimum) / (maximum - minimum)
    return df


def format_date_column(df):
    """
    Takes a dataframe and formats the date column ('Date').
    :param df: dataframe
    :return: dataframe
    """
    # format the date column as a string
    df['Date'] = df['Date'].astype(str)
    # format the date column as a datetime object
    df['Date'] = df['Date'].apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d'))
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


def create_cdfs_and_fdfs(df, window, step, return_period):
    """
    Takes a dataframe and creates current dataframes and future dataframes.
    :param df: dataframe
    :param window: size of the window
    :param step: size of the step
    :param return_period: return period
    :return: current dataframes and future dataframes
    """
    df = df.reset_index(drop=True)
    rows = df.shape[0]  # total number of rows in the dataframe
    cdfs = []
    fdfs = []
    for i in range(0, rows - window, step):
        j = i + return_period
        if j + window <= rows:  # future dataframe can be formed
            cdf = df.iloc[i: i + window, :]  # current dataframe
            fdf = df.iloc[j: j + window, :]  # future dataframe
            cdfs.append(cdf)
            fdfs.append(fdf)
    return cdfs, fdfs


def create_rnn_input(df):
    """
    Takes a dataframe and creates an Recurrent Neural Network (RNN) input.
    :param df: dataframe
    :return: rnn input
    """
    # rename columns
    df = df.rename(columns_remap, axis=1)
    # filter the columns
    df = df.filter(items=list(columns_remap.values()))
    # format the date column
    df = format_date_column(df)
    # set the date column as the index
    df = df.set_index('Date')
    # add technical indicators
    df = add_technical_indicators(df)
    # drop all columns with nan
    df = df.dropna(axis=1, how='all')
    # perform group normalization
    columns = ['Open',
               'High',
               'Low',
               'Close',
               'SMA20',
               'SMA50',
               'EMA13',
               'BBU20',
               'BBL20',
               'BBU50',
               'BBL50']
    df = perform_normalization(df, columns)
    # perform individual normalization
    columns = ['Volume',
               'OBV']
    for column in columns:
        df = perform_normalization(df, [column])
    # move the dataframe to a numpy array
    array = np.array(df)
    return array


def create_cnn_input(df):
    """
    Takes a dataframe and creates a Convolutional Neural Network (CNN) input.
    :param df: dataframe
    :return: cnn input
    """
    # rename columns
    df = df.rename(columns_remap, axis=1)
    # filter the columns
    df = df.filter(items=list(columns_remap.values()))
    # format the date column
    df = format_date_column(df)
    # set the date column as the index
    df = df.set_index('Date')
    # add technical indicators
    df = add_technical_indicators(df)
    # drop all columns with nan
    df = df.dropna(axis=1, how='all')
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


def create_label(df):
    """
    Takes a dataframe and creates a Recurrent Neural Network (RNN) label.
    :param df: dataframe
    :return: rnn label
    """
    # rename columns
    df = df.rename(columns_remap, axis=1)
    # filter the columns
    df = df.filter(items=list(columns_remap.values()))
    # format the date column
    df = format_date_column(df)
    # set the date column as the index
    df = df.set_index('Date')
    # perform group normalization
    columns = ['Open',
               'High',
               'Low',
               'Close']
    df = perform_normalization(df, columns)
    # perform individual normalization
    columns = ['Volume']
    for column in columns:
        df = perform_normalization(df, [column])
    # only consider the final row
    df = df.iloc[-1]
    # move the dataframe to a numpy array
    array = np.array(df)
    return array
