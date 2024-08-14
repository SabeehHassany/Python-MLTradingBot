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
import numpy as np  # For numerical calculations
import talib # Library for technical analysis indicators

# Define constants for ETF trading strategy 
#TAKE_PROFIT_MULTIPLIER_BUY = 1.20  # Multiplier for take profit price when buying
#STOP_LOSS_MULTIPLIER_BUY = 0.95  # Multiplier for stop loss price when buying
#TAKE_PROFIT_MULTIPLIER_SELL = 0.80  # Multiplier for take profit price when selling (short sell)
#STOP_LOSS_MULTIPLIER_SELL = 1.05  # Multiplier for stop loss price when selling (short sell)
#SENTIMENT_THRESHOLD = 0.90  # Threshold for sentiment probability to trigger a trade
#SYMBOL = "SPY"
#CASH_AT_RISK = .5 # Fraction of cash to be risked per trade
#SLEEP_TIME = "24H"  # Time between trading iterations

# Define constants for day trading strategy
#TAKE_PROFIT_MULTIPLIER_BUY = 1.01  # Narrow take profit for day trading
#STOP_LOSS_MULTIPLIER_BUY = 0.99  # Narrow stop loss for day trading
#TAKE_PROFIT_MULTIPLIER_SELL = 0.99  # Narrow take profit for short selling in day trading
#STOP_LOSS_MULTIPLIER_SELL = 1.01  # Narrow stop loss for short selling in day trading
#SENTIMENT_THRESHOLD = 0.75  # Lower threshold for day trading
#SYMBOL = "TSLA"
#CASH_AT_RISK = .15 # Fraction of cash to be risked per trade
#SLEEP_TIME = "15M"  # More frequent trading iterations, every 15 minutes

# Define constants for multi-indicator strategy
SENTIMENT_THRESHOLD_BUY = 0.85  # Sentiment threshold for buying
SENTIMENT_THRESHOLD_SELL = 0.95  # Sentiment threshold for selling
MA_PERIOD_SHORT = 20  # Period for short moving average
MA_PERIOD_LONG = 50  # Period for long moving average
RSI_PERIOD = 14  # Period for RSI calculation
RSI_OVERBOUGHT = 70  # RSI overbought threshold
RSI_OVERSOLD = 30  # RSI oversold threshold
ATR_PERIOD = 14  # Period for ATR calculation
VOLATILITY_THRESHOLD = 1.5  # Multiplier for adjusting position size based on volatility
SYMBOL = "TSLA"
CASH_AT_RISK = .5 # Fraction of cash to be risked per trade
SLEEP_TIME = "24H"  # Time between trading iterations


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

# Define the ParameterTrader class inheriting from Strategy
class ParameterTrader(Strategy):
    """
    The ParameterTrader Strategy is a flexible algorithm that adapts to both long-term investing and high-frequency 
    day trading. It uses sentiment analysis of financial news to guide trading decisions. Depending on the parameters, 
    it can operate as a long-term strategy, reacting to significant sentiment shifts, or as a day trading strategy, 
    executing frequent trades based on short-term sentiment changes. 
    """
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

    def get_sentiment(self):
        """Fetch dates and estimate sentiment from news headlines."""
        today = (self.get_datetime()).strftime('%Y-%m-%d')  # Get current date
        three_days_prior = (self.get_datetime() - Timedelta(days=3)).strftime('%Y-%m-%d')  # Get news from the past 3 days
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)  # Fetch news
        news = [ev.__dict__["_raw"]["headline"] for ev in news]  # Extract headlines
        probability, sentiment = estimate_sentiment(news)  # Estimate sentiment
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

# Define the IndicatorTrader class inheriting from Strategy
class IndicatorTrader(Strategy):
    """
    IndicatorTrader Strategy combines sentiment analysis with technical indicators to make trading decisions. 
    It uses moving averages and RSI (Relative Strength Index) in addition to sentiment analysis to identify 
    potential buy and sell signals. This strategy is designed for day trading and aims to capture medium-term 
    price movements.
    """
    def initialize(self, symbol: str = "SPY", cash_at_risk: float = .5, sleeptime: str = "24H"):
        """
        Initialize the trading strategy with default parameters.
        
        symbol (str): The stock symbol to trade. Defaults to 'SPY', which represents the S&P 500.
        cash_at_risk (float): The fraction of total cash to be risked on each trade. Defaults to 0.5 (or 50%). Higher value means higher risk taken.
        sleeptime (str): Time between trading iterations. Defaults to '24H'.

        """
        self.symbol = symbol  # Set the trading symbol
        self.sleeptime = sleeptime  # Set sleep time between iterations
        self.last_trade = None  # Track the last trade action
        self.cash_at_risk = cash_at_risk  # Set the cash at risk per trade
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET, api_version="v2")  # Initialize Alpaca API connection
    
    def get_technical_indicators(self):
        """Calculate technical indicators used for trading decisions."""
        # Get start and end dates
        end_date = self.get_datetime().strftime('%Y-%m-%d')
        start_date = (self.get_datetime() - Timedelta(days=100)).strftime('%Y-%m-%d')

        # Get historical price data
        bars = self.api.get_bars(self.symbol, '1D', start=start_date, end=end_date, limit=100).df
        
        # Check if the DataFrame is empty
        if bars.empty:
            print("No data returned from the API for symbol:", self.symbol)
            return None, None, None, None  # Returning None or handle appropriately
        
        # Calculate short and long moving averages for the closing prices
        short_ma = talib.SMA(bars['close'], timeperiod=MA_PERIOD_SHORT)
        long_ma = talib.SMA(bars['close'], timeperiod=MA_PERIOD_LONG)
        
        # Calculate the Relative Strength Index (RSI), that measures the speed and change of price movements relative to a momentum
        rsi = talib.RSI(bars['close'], timeperiod=RSI_PERIOD)
        
        # Calculate the Average True Range (ATR) which measures market volatility for understanding how much the price typically moves.
        atr = talib.ATR(bars['high'], bars['low'], bars['close'], timeperiod=ATR_PERIOD)
                
        return short_ma, long_ma, rsi, atr

    def get_sentiment(self):
        """Fetch dates and estimate sentiment from news headlines."""
        today = (self.get_datetime()).strftime('%Y-%m-%d')  # Get current date
        three_days_prior = (self.get_datetime() - Timedelta(days=3)).strftime('%Y-%m-%d')  # Get news from the past 3 days
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)  # Fetch news
        news = [ev.__dict__["_raw"]["headline"] for ev in news]  # Extract headlines
        probability, sentiment = estimate_sentiment(news)  # Estimate sentiment
        
        return probability, sentiment
    
    def dynamic_position_sizing(self, atr):
        """Dynamically adjust position size based on volatility (ATR)."""
        cash = self.get_cash()  # Get available cash
        last_price = self.get_last_price(self.symbol)  # Get the last price
        risk_per_share = atr[-1] * VOLATILITY_THRESHOLD  # Adjust risk per share based on volatility
        position_size = np.floor(cash / (risk_per_share + last_price))  # Calculate position size
        #print(f"Calculated Position Size: {position_size}")
        
        return position_size

    def execute_trade(self, trade_type, quantity):
        """Helper function to execute buy or sell trades."""
        order = self.create_order(
            self.symbol,
            quantity,
            trade_type,
            type="market"
        )
        self.submit_order(order)  # Submit the order
        self.last_trade = trade_type  # Update last trade

    def on_trading_iteration(self):
        """Execute trading logic on each iteration."""
        short_ma, long_ma, rsi, atr = self.get_technical_indicators()  # Get technical indicators
        probability, sentiment = self.get_sentiment()  # Get sentiment analysis
        
        print(f"Short MA: {short_ma[-1]}, Long MA: {long_ma[-1]}, RSI: {rsi[-1]}, ATR: {atr[-1]}, Sentiment: {sentiment}, Probability: {probability}")

        # Ensure that all technical indicators and sentiment align for a trade
        if short_ma[-1] > long_ma[-1] and rsi[-1] < RSI_OVERSOLD and sentiment == "positive" and probability > SENTIMENT_THRESHOLD_BUY:
            # Buy if short MA is above long MA, RSI indicates oversold, and sentiment is positive
            print("Conditions met for buying")
            if self.last_trade != "buy":
                self.sell_all()  # Close any open short positions
            quantity = self.dynamic_position_sizing(atr)  # Adjust position size based on volatility
            self.execute_trade("buy", quantity)  # Execute buy order
                
        elif short_ma[-1] < long_ma[-1] and rsi[-1] > RSI_OVERBOUGHT and sentiment == "negative" and probability > SENTIMENT_THRESHOLD_SELL:
            # Sell if short MA is below long MA, RSI indicates overbought, and sentiment is negative
            print("Conditions met for selling")
            if self.last_trade != "sell":
                self.sell_all()  # Close any open long positions
            quantity = self.dynamic_position_sizing(atr)  # Adjust position size based on volatility
            self.execute_trade("sell", quantity)  # Execute sell order

# Define the start and end dates for backtesting
start_date = datetime(2021, 1, 1)
end_date = datetime(2021, 4, 1)

# Initialize the broker with Alpaca credentials
broker = Alpaca(ALPACA_CREDS)

# Create an instance of the strategy
#strategy = ParameterTrader(name='parametertrader', broker=broker, parameters={"symbol": SYMBOL, "cash_at_risk": CASH_AT_RISK, "sleeptime": SLEEP_TIME})
strategy = IndicatorTrader(name='indicatortrader', broker=broker, parameters={"symbol": SYMBOL, "cash_at_risk": CASH_AT_RISK, "sleeptime": SLEEP_TIME})


# Run backtesting for the strategy
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": SYMBOL, "cash_at_risk": CASH_AT_RISK, "sleeptime": SLEEP_TIME}
)

# Uncomment the following lines to deploy the strategy for live trading
# trader = Trader()
# trader.add_strategy(strategy)
# trader.run_all()