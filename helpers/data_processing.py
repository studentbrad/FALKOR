from helpers.technical_indicators import sma, macd, obv, bollinger_bands, ema
import warnings

def add_ti(df):
    """Take an input df of ochlv data and return a df ready for creating a chart image with"""
    warnings.filterwarnings("ignore")
    # Create Technical Indicators for df

    # price and volume as lists of floats
    price_list = [float(x) for x in df.close.tolist()]
    volume_list = [float(x) for x in df.volume.tolist()]

    sma20_list = sma(price_list, n=20)
    macd_list = macd(price_list)
    obv_list = obv(volume_list, price_list)

    bb20 = bollinger_bands(price_list, 20, mult=2)
    bb20_low = [x[0] for x in bb20]
    bb20_mid = [x[1] for x in bb20]
    bb20_up = [x[2] for x in bb20]

    ti_dict = {'sma20': sma20_list, 'macd': macd_list, 'obv': obv_list,
             'bb20_low': bb20_low, 'bb20_mid': bb20_mid, 'bb20_up': bb20_up}

    # Cut all data to have equal length

    smallest_len = min( [len(x) for l, x in ti_dict.items()] )
        
    df = df[len(df) - smallest_len:]

    for label, data in ti_dict.items():

        # convert smallest_len to a start_index for this data list
        start_i = (len(data) - smallest_len)

        ti_dict[label] = data[start_i:]

        # add to df
        df[label] = ti_dict[label]
        
    warnings.filterwarnings("default")
    return df

def clean_candles_df(candles_df):
    candles = candles_df.reset_index(drop=True)
    candles = candles.ffill()
    candles = candles.astype(float)
    return candles


def price_returns(df, num_rows=30, num_into_fut=5, step=10):
    """Get the return percentage of a candlestick further into the future for a list of labels"""
    df = df.reset_index(drop=True)
    labels = []
    
    for row_i in range(0, df.shape[0] - num_rows - num_into_fut, step):
        # skip all iterations while row_i < num_rows since nothing yet to create a label for
        if row_i <= num_rows: continue
        
        vf, vi = df['close'][row_i+num_into_fut], df['close'][row_i]
        price_return = (vf - vi) / vi
        labels.append(price_return)
    return labels

def split_candles(df, num_rows=30, step=10):
    """Split a DataFrame of candlestick data into a list of smaller DataFrames each with num_rows rows"""
    
    slices = []
    
    for row_i in range(0, df.shape[0] - num_rows, step):
        small_df = df.iloc[row_i:row_i+num_rows, :]
        slices.append(small_df)
        
    return slices