# Download training data and organize it into folders

from BookWorm import BookWorm
from api_wrappers.BinanceWrapper import BinanceWrapper

bw_wrapper = BinanceWrapper('7UsOxTtXWYOJIqDaHtwBymd6wCBqhIMFCzLTl24YBhwHm9upVl0QkJ3krYBTR0sn', 'HG4ShjgPFUPw8FpBkyjCFlb8pR2ycqVEbTtMbwI5H1C79MSe3aSMPGmrOfuvL91u')
x = bw_wrapper.client.get_historical_klines('ETHBTC', '1m', 'January 1 2017', 'February 1 2017')

worm = BookWorm()

start_date = 'Jan 1 2017'
end_date = 'Feb 1 2017'

symbols = ['ETHBTC', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'EOSUSDT', 'LTCUSDT', 'BTCUSDC', 'XRPUSDT', 'BCHABCUSDT', 'ETCUSDT', 'LINKUSDT', 'TRXUSDT', 'BTTUSDT']

path_name = "train_data/"


def download_multiple_symbols(path, symbol_list, start_time, end_time, api_wrapper, interval):
    
    for symbol in symbol_list:
        data = bw_wrapper.historical_candles(symbol, interval, start_time=start_time, end_time=end_time)
        data.to_csv(path + symbol + ".csv", index=False)
        print("saved {} with shape {}".format(path+symbol+".csv", data.shape))

download_multiple_symbols(path_name, symbols, start_date, end_date, bw_wrapper, '1m')