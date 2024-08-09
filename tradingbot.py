# 
# This program creates a trading bot that uses sentiment analysis on financial news to inform trading decisions. Made using Python, Lumibot, Alpaca API, and FinBERT.
# Inspired by Nicholas Renotte's tutorial: https://www.youtube.com/watch?v=c9OjEThuJjY
#

from lumibot.brokers import Alpaca  # Interface for executing trades through the Alpaca broker
from lumibot.backtesting import YahooDataBacktesting  # Module for backtesting trading strategies using Yahoo Finance data
from lumibot.strategies.strategy import Strategy  # Base class for creating a trading strategy
from lumibot.traders import Trader  # Framework for running and managing trading strategies
from datetime import datetime  # Module for handling date and time operations
from alpaca_trade_api import REST  # Interface for interacting with Alpaca's REST API
from timedelta import Timedelta  # Function used for calculating date differences
from finbert_utils import estimate_sentiment  # Function for estimating sentiment from news headlines using FinBERT
from math import floor  # Function used for rounding down and working with 0's as opposed to round()

# Define constants for thresholds and commonly used values (for magic numbers)
TAKE_PROFIT_MULTIPLIER_BUY = 1.20  # Multiplier for take profit price when buying
STOP_LOSS_MULTIPLIER_BUY = 0.95  # Multiplier for stop loss price when buying
TAKE_PROFIT_MULTIPLIER_SELL = 0.80  # Multiplier for take profit price when selling (short sell)
STOP_LOSS_MULTIPLIER_SELL = 1.05  # Multiplier for stop loss price when selling (short sell)
SENTIMENT_THRESHOLD = 0.999  # Threshold for sentiment probability to trigger a trade

# Define API credentials
API_KEY = "key" 
API_SECRET = "secret" 
BASE_URL = "https://paper-api.alpaca.markets/v2"

# Create a dictionary to store Alpaca credentials
ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}

# Define the MLTrader class inheriting from Strategy
class MLTrader(Strategy):
    def initialize(self, symbol: str = "SPY", cash_at_risk: float = .5, sleeptime: str = "24H"):  # Added sleeptime as a parameter
        """
        Initialize the trading strategy with default parameters.

        symbol (str): The stock symbol to trade. Defaults to 'SPY', which represents the SPDR S&P 500 ETF.
        cash_at_risk (float): The fraction of total cash to be risked on each trade. Defaults to 0.5 (or 50%). Higher value means higher risk taken.
        sleeptime (str): Time between trading iterations. Defaults to '24H'.
        """
        self.symbol = symbol  # Set the trading symbol
        self.sleeptime = sleeptime  # Set sleep time between iterations
        self.last_trade = None  # Track the last trade action
        self.cash_at_risk = cash_at_risk  # Set the cash at risk per trade
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)  # Initialize Alpaca API connection

    def position_sizing(self):
        """Calculate the position size for the trade."""
        cash = self.get_cash()  # Get available cash
        last_price = self.get_last_price(self.symbol)  # Get the last price of the symbol
        quantity = floor(cash * self.cash_at_risk / last_price)  # Calculate the number of shares to buy/sell
        return cash, last_price, quantity

    def get_dates(self):
        """Get today's date and the date three days prior."""
        today = self.get_datetime()  # Get current date
        three_days_prior = today - Timedelta(days=3)  # Calculate the date three days prior
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')  # Return formatted dates

    def get_sentiment(self):
        """Fetch and estimate sentiment from news headlines."""
        today, three_days_prior = self.get_dates()  # Get date range for news
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)  # Fetch news for the symbol
        news = [ev.__dict__["_raw"]["headline"] for ev in news]  # Extract headlines from news data
        probability, sentiment = estimate_sentiment(news)  # Estimate sentiment from headlines
        return probability, sentiment

    def execute_trade(self, trade_type, last_price, quantity):
        """Helper function to execute buy or sell trades."""
        if trade_type == "buy":
            take_profit_price = last_price * TAKE_PROFIT_MULTIPLIER_BUY
            stop_loss_price = last_price * STOP_LOSS_MULTIPLIER_BUY
        elif trade_type == "sell":
            take_profit_price = last_price * TAKE_PROFIT_MULTIPLIER_SELL
            stop_loss_price = last_price * STOP_LOSS_MULTIPLIER_SELL

        order = self.create_order(
            self.symbol,
            quantity,
            trade_type,
            type="bracket",
            take_profit_price=take_profit_price,  # Set take profit price
            stop_loss_price=stop_loss_price  # Set stop loss price
        )
        self.submit_order(order)  # Submit the order
        self.last_trade = trade_type  # Update the last trade action

    def on_trading_iteration(self):
        """Execute trading logic on each iteration."""
        cash, last_price, quantity = self.position_sizing()  # Get position sizing
        probability, sentiment = self.get_sentiment()  # Get sentiment

        if cash > last_price:  # Ensure there is enough cash to trade
            if sentiment == "positive" and probability > SENTIMENT_THRESHOLD:
                if self.last_trade == "sell":  # Close any open short positions
                    self.sell_all()
                self.execute_trade("buy", last_price, quantity)  # Execute buy order
            elif sentiment == "negative" and probability > SENTIMENT_THRESHOLD:
                if self.last_trade == "buy":  # Close any open long positions
                    self.sell_all()
                self.execute_trade("sell", last_price, quantity)  # Execute sell order

# Define the start and end dates for backtesting
start_date = datetime(2020, 3, 30)
end_date = datetime(2021, 12, 31)

# Initialize the broker with Alpaca credentials
broker = Alpaca(ALPACA_CREDS)

# Create an instance of the MLTrader strategy
strategy = MLTrader(name='mlstrat', broker=broker, parameters={"symbol": "SPY", "cash_at_risk": .5})

# Run backtesting for the strategy
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": "SPY", "cash_at_risk": .75}
)

# Uncomment the following lines to deploy the strategy for live trading
# trader = Trader()
# trader.add_strategy(strategy)
# trader.run_all()