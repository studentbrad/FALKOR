from BudFox import BudFox
from BookWorm import BookWorm
from Portfolio import Portfolio

from strategies.Strategy import Strategy
from helpers.datasets import DFTimeSeriesDataset
from helpers.data_processing import clean_candles_df, add_ti, price_returns, 
from torch.utils.data import *

class Gekko:
    """
    The engine room of Falkor. Every iteration, Gekko receives an input 
    of live ochl data. It sends this data to the selected Strategy, which 
    generates buy/sell signals and real-time model updating. Gekko takes 
    these signals and sends them to BudFox who will realize them. 
    Gekko then updates portfolio with any profits made
    
    Attributes:
        
        bud_fox: BudFox
            - An instance of BudFox class used to send buy/sell signals to 
            an APIWrapper for realization

        portfolio: Portfolio
            - An instance of Portfolio class that stores all trades and 
            securities  

        book_worm: BookWorm
            - An instance of BookWorm class used for pulling live data 
            from APIWrappers

    Methods:

        _trade(self, s)
            - helper method used to trade one Security, s.

        trade_portfolio(self)
            - calls _trade on every Security inside portfolio

    """

    def __init__(self, portfolio: Portfolio):
        """Initialize Gekko"""
        self.portfolio = portfolio

        self.bud_fox = BudFox()
        self.book_worm = BookWorm()

    def _trade(self, sec):
        """
        Helper method for self.trade_portfolio(). Runs strategy for security 
        and sends signals to api_wrapper
        """

        # If this security is waiting for a further time period to be traded,
        # don't trade it but update waiting periods
        if sec.status == "waiting":
            # subtract one from _waiting_periods_left
            sec.update_waiting_status(amount=-1)
            return "Gekko: This security has {} waiting periods left".format(
                sec._waiting_periods_left)
        
        # trade this security if its ready
        elif sec.status == "ready":
            
            # Get 100 most recent candles of data
            # NOTE: Creating technical indicators cuts 100 candles down to ~ 50-80 depending on the indicator. We want 30 
            # periods, so we take 100 to leave us with plenty of room to spare       
            last_candles = self.book_worm.last_candles(100, sec.api_wrapper, sec.symbol, sec.interval)

            # Get trading signals from strategy

            # Feed last_candles to strategy
            sec.strategy.feed_data(last_candles)
            # Make a prediction based on last_candles
            signal = sec.strategy.predict()
            # Update model however specified
            sec.strategy.update()


            # If strategy predicts price growth in 5 periods, we set 
            # s.status == waiting until 5 periods from now

            if signal == "buy":
                sec.set_status(status="waiting", periods=5)

            # Send signal to BudFox for realization

            # tell BudFox to paper trade instead of real trade
            self.bud_fox.paper_trade = True

            # send trading signal to BudFox
            trade_info = self.bud_fox.send_trading_signal(sec.symbol, signal, amount=20, api_wrapper=sec.api_wrapper, price="market")

            # return information pertaining to this trade
            return trade_info

        else:
            raise SyntaxError("sec.status is set to something other than ready or waiting")

    def trade_portfolio(self):
        """Iterates through every security in self.portfolio, using their specified Strategy and APIWrapper"""
        for security in self.portfolio.securities_trading:
            trade_info = self._trade(security)
            print(trade_info)

    def make_predictions(self, model, dl):
        """Make all predictions for batches in dl with model, returning a list of tuples in the format (output, truth)"""
        predictions = []

        for batch, labels in dl:
            output = model(batch)

            output_list = output.squeeze().tolist()
            label_list = labels.squeeze().tolist()

            for i in range(len(output_list)):
                predictions.append( (output_list[i], label_list[i]) )

        return predictions

    # TODO create a func for candles_to_input()

    def model_predictions(self, model, candles):
        """Convert candles into input data and use model to predict the output for each input,
        returning a list of tuples in the form (output, truth)"""
        
        # simple data cleaning 
        candles = clean_candles_df(candles)
        
        print('adding technical indicators')
        candles = add_ti(candles)
        
        # remove time column
        candles = candles.drop('time', axis=1).reset_index(drop=True)
        
        print('creating input and label lists')
        labels = price_returns(candles)
        inputs = split_candles(candles)
        # remove all inputs without a label
        inputs = inputs[len(inputs)-len(labels):]

        # calculate s - index of train/valid split
        s = int(len(inputs) * 0.7)
        
        print('creating Datasets and DataLoaders')
        if needs_image:
            ds = OCHLVDataset(inputs, labels)
        else:
            ds = DFTimeSeriesDataset(inputs, labels)

        dl = DataLoader(ds, drop_last=True, batch_size=64)
        predictions = make_predictions(model, dl)

        return predictions


from models.GRU.GRU import GRUnet
model = GRUnet(11, 30, 64, 500, 3)
candles = pd.read_csv('bitcoin1m.csv')

g = Gekko()
preds = g.model_predictions(model, candles)

print(preds[0])