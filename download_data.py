# Download training data and organize it into folders

from BookWorm import BookWorm
from api_wrappers.BinanceWrapper import BinanceWrapper

bw_wrapper = BinanceWrapper('TCXVekUUrxkz6lvXqPsSXLiCgsGXI2lxs2O9QBge1jvkBYSZ762Mw64HRTDvXwZD', 
'D3ExC4xLz5mInwfNojeVdi5pHyATPsbdQcSr5ld0z98FsqmmNF5AVlQ7Fr4Z5vYi')
worm = BookWorm()

start_date = 'Jan 1 2017'
end_date = 'Feb 1 2017'

symbols = ['ETHBTC', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'EOSUSDT', 'LTCUSDT', 'BTCUSDC', 'XRPUSDT', 'BCHABCUSDT', 'ETCUSDT', 'LINKUSDT', 'TRXUSDT', 'BTTUSDT']

path_name = "train_data/"


def download_multiple_symbols(path, symbol_list, start_time, end_time, api_wrapper):
    worm = BookWorm()
    
    for symbol in symbol_list:
        data = worm.historical_candles(start_time=start_time, end_time=end_time, api_wrapper=api_wrapper, symbol=symbol, interval='1m')
        data.to_csv(path + symbol + ".csv", index=False)
        print("saved {} with shape {}".format(path+symbol+".csv", data.shape))

download_multiple_symbols(path_name, symbols,start_date, end_date, bw_wrapper)