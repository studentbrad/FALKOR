from .technical_indicators import sma, macd, obv, bollinger_bands, ema
import warnings

# Normalization Functions
def minmaxnorm(ser):
    """Takes a Pandas Series and returns the series with minmax normalization applied"""
    return (ser-ser.min())/(ser.max()-ser.min())


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

def price_return(current_df, future_df, return_period):
    """Calculates the price return from the last entry in current_df['close'] and future_df['close'][return_period]"""
    curr_price = current_df['close'].iloc[-1]
    fut_price = future_df['close'].iloc[return_period - 1]

    print(curr_price, fut_price)

    return (fut_price - curr_price) / curr_price

def candles_to_inputs_and_labels(candles_df, num_rows=30, step=10, return_period=5):
    """Split a DataFrame of candlestick data into a list of smaller DataFrames each with num_rows rows.
    Returns (split_candles, labels), where labels are calculated with label_func """
    candles_df = candles_df.reset_index(drop=True)

    slices, labels = [], []
    
    # loop through all avaliable rows with step 
    for row_i in range(0, candles_df.shape[0] - num_rows - return_period, step):
        small_df = candles_df.iloc[row_i : row_i+num_rows, :]
        future_df = candles_df.iloc[row_i+num_rows:, :]
        label = price_return(small_df, future_df, return_period)

        slices.append(small_df)
        labels.append(label)
        
    return slices, labels