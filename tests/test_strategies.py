from ..code.strategies import CNNStrat, RNNStrat
from ..code.models import CNN, RNN

import pandas as pd
from torch.utils.data import DataLoader

def test_cnn_strat():
    model = CNN()
    strat = CNNStrat(model)

    input_df = pd.read_csv('tests/sample_candles.csv')
    
    strat.feed_data(input_df)
    prediction = strat.predict()

    assert prediction == 'buy' or prediction == 'sell'

def test_rnn_strat():
    model = RNN()
    strat = RNNStrat(model)

    input_df = pd.read_csv('tests/sample_candles.csv')
    
    strat.feed_data(input_df)
    prediction = strat.predict()

    assert prediction == 'buy' or prediction == 'sell'