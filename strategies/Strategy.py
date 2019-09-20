
class Strategy:
    """Abstract class representing a Strategy used by Gekko. The child class must create all NotImplemented methods"""

    def feed_data(self, last_candles):
        """ Feed last_candles to strategy """
        raise NotImplementedError    
    def predict(self):
        """ Make a prediction based on last_candles"""
        raise NotImplementedError
    def update(self):
        """Update model however specified"""
        raise NotImplementedError