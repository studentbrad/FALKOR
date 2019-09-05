import Strategy

class ModelStrat(Strategy):
    """Basic Strategy for a PyTorch Model trained to output a signal 
    which is the predicted return on investment. i.e. ratio of 
    fut_price/curr_price"""

    def __init__(self, model):
        """Initialize ModelStrat with model"""
        self.model = model

    def generate_signals(self, input_df):
        """Returns a list of trading signals"""
        return self.model(input_df)
        

