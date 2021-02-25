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

# default list of technical indicators
default_technical_indicators = [
    'SMA20',
    'SMA50',
    'EMA13',
    'OBV',
    'BBU20',
    'BBL20',
    'BBU50',
    'BBL50'
]


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


def format_date_column(df, date='Date', datetime_format='%Y%m%d'):
    """
    Takes a dataframe and formats the date column.
    :param df: dataframe
    :param date: date column name
    :param datetime_format: format
    :return: dataframe
    """
    # format the date column as a string
    df[date] = df[date].astype(str)
    # format the date column as a datetime object
    df[date] = df[date].apply(lambda x: datetime.datetime.strptime(x, datetime_format))
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


def add_technical_indicators(df, price='Close', volume='Volume', technical_indicators=None):
    """
    Takes a dataframe and adds technical indicators.
    :param df: dataframe
    :param price: price column name
    :param volume: volume column name
    :param technical_indicators: list of technical indicators
    :return: dataframe
    """
    if technical_indicators is None:
        technical_indicators = default_technical_indicators
    if 'SMA20' in technical_indicators or \
            'BBU20' in technical_indicators or \
            'BBL20' in technical_indicators:
        rolling20 = df[price].rolling(window=20)
        mean20 = rolling20.mean()
        std20 = rolling20.std()
    else:
        mean20 = None
        std20 = None
    if 'SMA50' in technical_indicators or \
            'BBU50' in technical_indicators or \
            'BBL50' in technical_indicators:
        rolling50 = df[price].rolling(window=50)
        mean50 = rolling50.mean()
        std50 = rolling50.std()
    else:
        mean50 = None
        std50 = None
    # calculate the simple moving average over a 20 day windows
    if 'SMA20' in technical_indicators:
        df['SMA20'] = mean20
    # calculate the simple moving average over a 50 day window
    if 'SMA50' in technical_indicators:
        df['SMA50'] = mean50
    # calculate the exponential moving average for a span of 13 days
    if 'EMA13' in technical_indicators:
        df['EMA13'] = df[price].ewm(span=13, adjust=False).mean()
    # calculate the on-balance volume
    if 'OBV' in technical_indicators:
        df['OBV'] = np.where(df[price] > df[price].shift(1), df[volume],
                             np.where(df[price] < df[price].shift(1), -df[volume], 0)).cumsum()
    # calculate the short term bollinger bands
    if 'BBU20' in technical_indicators:
        df['BBU20'] = mean20 + std20 * 2
    if 'BBL20' in technical_indicators:
        df['BBL20'] = mean20 - std20 * 2
    # calculate the long term bollinger bands
    if 'BBU50' in technical_indicators:
        df['BBU50'] = mean50 + std50 * 2.5
    if 'BBL50' in technical_indicators:
        df['BBL50'] = mean50 - std50 * 2.5
    return df


def create_smaller_dataframes(df, window, step):
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


def create_relative_dataframe(df, start_index=1, end_index=None):
    """
    Takes a dataframe of size n x m with rows r_i, r_(i + 1), ..., r_n
    and creates a dataframe of size n x m
    with relative values where r_i' = r_i, r_(i + 1)' = r_(i + 1) / r_i, ..., r_n' = r_n / r_(n - 1).
    :param df: dataframe
    :param start_index: start index
    :param end_index: end index
    :return: dataframe
    """
    if end_index is None:
        end_index = df.shape[0]
    for i in range(end_index - 1, start_index - 1, -1):
        df.iloc[i] = df.iloc[i] / df.iloc[i - 1]
    return df


def create_logarithm_dataframe(df, start_index=1, end_index=None):
    """
    Takes a dataframe of size n x m with rows r_i, r_(i + 1), ..., r_n
    and creates a dataframe of size n x m
    with logarithm values where r_i' = r_i, r_(i + 1)' = log(r_(i + 1)), ..., r_n' = log(r_n).
    :param df: dataframe
    :param start_index: start index
    :param end_index: end index
    :return: dataframe
    """
    if end_index is None:
        end_index = df.shape[0]
    df.iloc[start_index:end_index] = df.iloc[start_index:end_index].apply(np.log)
    return df


def create_rnn_input_label(df, drop_nan=False, fill_nan=False, lower=None, upper=None, return_df=False):
    """
    Takes a dataframe and creates an Recurrent Neural Network (RNN) input and label.
    :param df: dataframe
    :param drop_nan: drop rows with nan
    :param fill_nan: fill nans with zeros
    :param lower: lower bound
    :param upper: upper bound
    :param return_df: return the formatted dataframe
    :return: input, label, dataframe (optional)
    """
    # format the dataframe
    df = format_dataframe(df)
    # add technical indicators
    df = add_technical_indicators(df)
    # drop rows with nan
    df = df.dropna() if drop_nan else df
    # create a relative dataframe
    df = create_relative_dataframe(df)
    # create a logarithm dataframe
    df = create_logarithm_dataframe(df)
    # clip values outside the bounds
    df = df.clip(lower, upper)
    # fill nans with zeros
    df = df.fillna(0.) if fill_nan else df
    # move the dataframe to a numpy array
    array = np.array(df)
    # remove the first row and the last row from the array
    rnn_input = array[1:-1]
    # remove the first two rows from the array
    rnn_label = array[2:]
    if return_df:
        return rnn_input, rnn_label, df
    else:
        return rnn_input, rnn_label


def create_cnn_input_label(df, drop_nan=False, fill_nan=False, lower=None, upper=None, return_df=False):
    """
    Takes a dataframe and creates a Convolutional Neural Network (CNN) input and label.
    :param df: dataframe
    :param drop_nan: drop rows with nan
    :param fill_nan: fill nans with zeros
    :param lower: lower bound
    :param upper: lower bound
    :param return_df: return the formatted dataframe
    :return: input, label, dataframe (optional)
    """
    # format the dataframe
    df = format_dataframe(df)
    # add technical indicators
    df = add_technical_indicators(df)
    # drop rows with nan
    df = df.dropna() if drop_nan else df
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
    cnn_input = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # reshape the numpy array to (height, width, channel)
    cnn_input = cnn_input.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # move the rgb dimension to the start for pytorch compatibility
    cnn_input = np.moveaxis(cnn_input, 2, 0)
    # show the plot
    # plt.show()
    # close the figure
    plt.close(fig)
    # create a relative dataframe
    df = create_relative_dataframe(df)
    # create a logarithm dataframe
    df = create_logarithm_dataframe(df)
    # clip values outside the bounds
    df = df.clip(lower, upper)
    # fill nans with zeros
    df = df.fillna(0.) if fill_nan else df
    # move the dataframe to a numpy array
    array = np.array(df)
    # remove all rows from the array, except for the last row
    cnn_label = array[-1]
    if return_df:
        return cnn_input, cnn_label, df
    else:
        return cnn_input, cnn_label
