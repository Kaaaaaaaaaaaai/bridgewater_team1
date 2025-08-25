import yfinance as yf

class PriceHistory:

    def __init__(self):

        self.symbol:str
        
    def get_price_history(self, start = None, end = None, period = "max"):

        data = yf.download(self.symbol, start, end, period = period, multi_level_index=False, auto_adjust=True)
        return data
    
    def get_retruns(self, start = None, end = None, period = "max"):

        data = self.get_price_history(start, end, period)
        returns = data['Close'].pct_change()
        return returns