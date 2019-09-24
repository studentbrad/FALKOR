
from ..code.apiwrappers import BinanceWrapper

def test_binance_wrapper():
    b = BinanceWrapper(
        client_id='nBjgb83VMNvqq45b3JdWUIsJDalWlXxHI2bvDz9oLdW7KgOLPvJCp30CHnthjfNJ',
        client_secret='5bBN7s7h37kUvmGIpF9FTAtspBY93WirwhTh39PV7AlKSlUE2S4EEe9b3OZVYIqd'
    )
    candles = b.historical_candles(symbol='ETHBTC', interval='1m', start_time='15 minutes ago')

    assert b.last_candles(num=5, symbol='ETHBTC', interval='1m').shape == (5, 6)
    assert candles.shape == (60, 6)
