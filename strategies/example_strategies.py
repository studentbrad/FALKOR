
class RNNStrat():
    def __init__(self, model):
        """Initialize ModelStrat with model"""
        self.model = model.cuda()

    def generate_signals(self, input_df):
        """Returns a list of trading signals"""
        return self.model(input_df.cuda())
        
class CNNStrat():
    def __init__(self, model):
        """Initialize ModelStrat with model"""
        self.model = model.cuda()

    def generate_signals(self, input_df):
        """Returns a list of trading signals"""
        return self.model(input_df.cuda())
