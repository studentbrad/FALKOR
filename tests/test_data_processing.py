from ..code.data_processing import candles_to_inputs_and_labels, price_return
import pandas as pd

candles = pd.read_csv('tests/600_candles.csv')

def test_candles_to_inputs_outputs():

    inputs, labels = candles_to_inputs_and_labels(candles, num_rows=30, step=10, return_period=5)
    
    assert len(inputs) == len(labels)
    assert inputs[0].shape[0] == 30
    
def test_price_return():
    small_df = candles.iloc[100:130, :]
    rest_df = candles.iloc[130:, :]
    label = price_return(small_df, rest_df, 5)

    assert round(label, 6) == 0.000141
