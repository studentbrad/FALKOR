"""This module contains methods relevant to creating a chart image for securities"""
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

import pandas as pd
from itertools import compress
from PIL import Image

import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import TICKLEFT, TICKRIGHT, Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

from six.moves import xrange, zip

import warnings

############### The following are MATPLOTLIB retired methods #######################

def _check_input(opens, closes, highs, lows, miss=-1):
    """Checks that *opens*, *highs*, *lows* and *closes* have the same length.
    NOTE: this code assumes if any value open, high, low, close is
    missing (*-1*) they all are missing
    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    opens : sequence
        sequence of opening values
    highs : sequence
        sequence of high values
    lows : sequence
        sequence of low values
    closes : sequence
        sequence of closing values
    miss : int
        identifier of the missing data
    Raises
    ------
    ValueError
        if the input sequences don't have the same length
    """

    def _missing(sequence, miss=-1):
        """Returns the index in *sequence* of the missing data, identified by
        *miss*
        Parameters
        ----------
        sequence :
            sequence to evaluate
        miss :
            identifier of the missing data
        Returns
        -------
        where_miss: numpy.ndarray
            indices of the missing data
        """
        return np.where(np.array(sequence) == miss)[0]

    same_length = len(opens) == len(highs) == len(lows) == len(closes)
    _missopens = _missing(opens)
    same_missing = ((_missopens == _missing(highs)).all() and
                    (_missopens == _missing(lows)).all() and
                    (_missopens == _missing(closes)).all())

    if not (same_length and same_missing):
        msg = ("*opens*, *highs*, *lows* and *closes* must have the same"
                " length. NOTE: this code assumes if any value open, high,"
                " low, close is missing (*-1*) they all must be missing.")
        raise ValueError(msg)

def candlestick2_ochl(ax, opens, closes, highs, lows, width=4,
                      colorup='k', colordown='r',
                      alpha=0.75):
    """Represent the open, close as a bar line and high low range as a
    vertical line.
    Preserves the original argument order.
    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    opens : sequence
        sequence of opening values
    closes : sequence
        sequence of closing values
    highs : sequence
        sequence of high values
    lows : sequence
        sequence of low values
    width : int
        size of open and close ticks in points
    colorup : color
        the color of the lines where close >= open
    colordown : color
        the color of the lines where close <  open
    alpha : float
        bar transparency
    Returns
    -------
    ret : tuple
        (lineCollection, barCollection)
    """

    return candlestick2_ohlc(ax, opens, highs, lows, closes, width=width,
                             colorup=colorup, colordown=colordown,
                             alpha=alpha)


def candlestick2_ohlc(ax, opens, highs, lows, closes, width=4,
                      colorup='k', colordown='r',
                      alpha=0.75):
    """Represent the open, close as a bar line and high low range as a
    vertical line.
    NOTE: this code assumes if any value open, low, high, close is
    missing they all are missing
    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    opens : sequence
        sequence of opening values
    highs : sequence
        sequence of high values
    lows : sequence
        sequence of low values
    closes : sequence
        sequence of closing values
    width : int
        size of open and close ticks in points
    colorup : color
        the color of the lines where close >= open
    colordown : color
        the color of the lines where close <  open
    alpha : float
        bar transparency
    Returns
    -------
    ret : tuple
        (lineCollection, barCollection)
    """

    _check_input(opens, highs, lows, closes)

    delta = width / 2.
    barVerts = [((i - delta, open),
                 (i - delta, close),
                 (i + delta, close),
                 (i + delta, open))
                for i, open, close in zip(xrange(len(opens)), opens, closes)
                if open != -1 and close != -1]

    rangeSegments = [((i, low), (i, high))
                     for i, low, high in zip(xrange(len(lows)), lows, highs)
                     if low != -1]

    colorup = mcolors.to_rgba(colorup, alpha)
    colordown = mcolors.to_rgba(colordown, alpha)
    colord = {True: colorup, False: colordown}
    colors = [colord[open < close]
              for open, close in zip(opens, closes)
              if open != -1 and close != -1]

    useAA = 0,  # use tuple here
    lw = 0.5,   # and here
    rangeCollection = LineCollection(rangeSegments,
                                     colors=colors,
                                     linewidths=lw,
                                     antialiaseds=useAA,
                                     )

    barCollection = PolyCollection(barVerts,
                                   facecolors=colors,
                                   edgecolors=colors,
                                   antialiaseds=useAA,
                                   linewidths=lw,
                                   )

    minx, maxx = 0, len(rangeSegments)
    miny = min([low for low in lows if low != -1])
    maxy = max([high for high in highs if high != -1])

    corners = (minx, miny), (maxx, maxy)
    ax.update_datalim(corners)
    ax.autoscale_view()

    # add these last
    ax.add_collection(rangeCollection)
    ax.add_collection(barCollection)
    return rangeCollection, barCollection

################################### END OF MATPLOTLIB METHODS ####################################

def _normalize_by_dataset(list1, list2, from_origin=False):
    """Normalize list1 to be in the same interval as list2, i.e. [0, max(list2)]"""

    if not from_origin:
        # To scale variable x from dataset X into range [a,b] we use:
        # x_norm = ( (b-a) * ( (x-min(X)) / (max(X)-min(X)) ) + a
        normalized_list1 = []
        for i in range(len(list1)):
            x_norm = ((max(list2) - min(list2)) * (
                        (list1[i] - min(list1)) / (
                            max(list1) - min(
                        list1))) + min(list2))
            normalized_list1.append(x_norm)

        return normalized_list1

    else:
        normalized_list1 = []
        for i in range(len(list1)):
            x_norm = ((max(list2)) * ((list1[i] - min(list1)) / (max(list1) - min(list1) + 0.0000001)))
            normalized_list1.append(x_norm)

        return normalized_list1

def remove_labels():
    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')


def create_chart(candles_df, tech_inds_df=None):
    """Creates a matplotlib figure containing the chart"""

    # turn off matplotlib warning spam
    warnings.filterwarnings('ignore')
    
    # Create matplotlib figure
    fig = plt.figure()
    
    # Create axis for the price and technical indicator graph
    ax0 = fig.add_subplot(411)

    # Plot Price
    candles_df.plot(x='time', y='close', ax=ax0, color='black', label='_nolegend_', linewidth=2)
    
    remove_labels()

    # Plot Technical Indicators
    if tech_inds_df:
        for col_name in tech_inds_df:
            ti_df = candles_df[['time', col_name]].copy()
            ti_df.plot(x='time', y=col_name, ax=ax0, label='_nolegend_')
        remove_labels()
    
    # Plot candlesticks
    ax1 = fig.add_subplot(412)

    candlestick2_ochl(
        width=0.4, colorup='g', colordown='r', ax=ax1, opens=candles_df['open'],
        closes=candles_df['close'], highs=candles_df['high'], lows=candles_df['low']
        )

    remove_labels()

    # Plot Volume as Bar Chart on the bottom

    time_list = candles_df['time'].tolist()
    volume_list = candles_df['volume'].tolist()
    norm_volume_list = _normalize_by_dataset(volume_list, candles_df['close'], from_origin=True)

    ax2 = fig.add_subplot(413)

    vol_df = pd.DataFrame(list(zip(time_list, norm_volume_list)), columns=['time', 'volume'])
    vol_df.plot.bar(x='time', y='volume', ax=ax2, label='_nolegend_')
    remove_labels()
    return fig

def chart_to_image(candles_df, file_name, tech_inds_df=None):
    """Creates the specified chart and saves it to an image at file_name location"""
    fig = create_chart(candles_df, tech_inds_df)

    # change resolution of image to 224x224
    resize_plot(fig, 224.0, 224.0)

    # remove labels and titles 
    remove_labels()

    plt.savefig(file_name, legend=False, bbox_inches='tight', dpi=85)

    # close all open plots to save memory
    plt.close('all')

    # restore warnings
    warnings.filterwarnings('default')

def chart_to_arr(candles_df, tech_inds_df=None):
    """Creates the specified chart and returns its numpy arr representation"""

    fig = create_chart(candles_df, tech_inds_df)

    # change resolution of image to 224x224
    resize_plot(fig, 224.0, 224.0)
    
    # remove labels and titles 
    remove_labels()

    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = np.moveaxis(data, 2, 0) # move rbg dimension to the start for pytorch compability
    
    warnings.filterwarnings('default')
    
    return data

def resize_plot(fig, width_px, height_px):
    DPI = fig.get_dpi()
    fig.set_size_inches(width_px/float(DPI),height_px/float(DPI)) 