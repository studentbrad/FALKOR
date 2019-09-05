
class Strategy:
    """Abstract class representing a Strategy used by Gekko. The child class must create all NotImplemented methods"""

    def generate_signals(self):
        """Returns a list of trading signals"""

        raise NotImplementedError

