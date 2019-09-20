from Strategy import Strategy
from helpers.datasets import DFTimeSeriesDataset, OCHLVDataset, df_to_array_transform, df_to_chartarray_transform
from models.GRU.GRU import GRUnet
from models.CNN.CNN import CNN

class GRUStrat(Strategy):
    def __init__(self):
        self.model = GRUnet(11, 30, 1, 500, 3)
        self.input = None
    
    def feed_data(self, last_candles):
        check_correct_size(last_candles, 30, 11)
        self.input = df_to_array_transform(last_candles)
        
    def predict(self):
        return self.model(self.input)

    def update(self):
        pass
    
class CNNStrat(Strategy):

    def __init__(self):
        self.model = CNN()
        self.input = None
    
    def feed_data(self, last_candles):
        check_correct_size(last_candles, 30, 11)
        self.input = df_to_chartarray_transform(last_candles)
        
    def predict(self):
        return self.model(self.input)

    def update(self):
        pass
    

def check_correct_size(candles_df, num_rows, num_cols):
    assert candles_df.shape[0] == num_rows
    assert candles_df.shape[1] == num_cols