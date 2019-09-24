from ..code.engine import Engine, BudFox
from ..code.strategies import CNNStrat

def test_engine():
    model_strat = CNNStrat
    e = Engine(mode='backtest', strategy=model_strat, candles_path='tests/600_candles.csv')
    e.run()

def test_budfox():
    pass