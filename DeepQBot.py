"""
Disclaimer

All investment strategies and investments involve risk of loss.
Nothing contained in this program, scripts, code or repositoy should be
construed as investment advice.Any reference to an investment's past or
potential performance is not, and should not be construed as, a recommendation
or as a guarantee of any specific outcome or profit.

By using this program you accept all liabilities,
and that no claims can be made against the developers,
or others connected with the program.
"""

# trading view library
from tradingview_ta import TA_Handler, Interval, Exchange

# use for environment variables
import os

# use if needed to pass args to external modules
import sys

# used to create threads & dynamic loading of modules
import threading
import importlib

# used for directory handling
import glob

# Needed for colorful console output Install with: python3 -m pip install colorama (Mac/Linux) or pip install colorama (PC)
from colorama import init
init()

# needed for the binance API / websockets / Exception handling
from binance.client import Client
from binance.exceptions import BinanceAPIException

# used for dates
from datetime import date, datetime, timedelta
import time

# used to repeatedly execute the code
from itertools import count

# used to store trades and sell assets
import json

# Load creds modules
from helpers.handle_creds import (
    load_correct_creds, test_api_key
)

# for colourful logging to the console
class txcolors:
    BUY = '\033[92m'
    WARNING = '\033[93m'
    SELL_LOSS = '\033[91m'
    SELL_PROFIT = '\033[32m'
    DIM = '\033[2m\033[35m'
    DEFAULT = '\033[39m'

# print with timestamps
old_out = sys.stdout
class St_ampe_dOut:
    """Stamped stdout."""
    nl = True
    def write(self, x):
        """Write function overloaded."""
        if x == '\n':
            old_out.write(x)
            self.nl = True
        elif self.nl:
            old_out.write(f'{txcolors.DIM}[{str(datetime.now().replace(microsecond=0))}]{txcolors.DEFAULT} {x}')
            self.nl = False
        else:
            old_out.write(x)

    def flush(self):
        pass

sys.stdout = St_ampe_dOut()

class DeepQBot(object):
    
    def __init__(self, 
    access_key,
    secret_key,
    TEST_MODE = True, 
    LOG_TRADES = True, 
    LOG_FILE = "trades.txt",
    DEBUG_SETTING = False,
    AMERICAN_USER = False,
    PAIR_WITH = 'USDT',
    QUANTITY = 15,
    MAX_COINS = 20,
    FIATS = ['EURUSDT', 'GBPUSDT', 'JPYUSDT', 'USDUSDT', 'DOWN', 'UP'],
    TIME_DIFFERENCE = 5,
    RECHECK_INTERVAL = 6,
    CHANGE_IN_PRICE = 0.5,
    STOP_LOSS = 5,
    TAKE_PROFIT = 0.65,
    CUSTOM_LIST = False,
    TICKERS_LIST = 'tickers.txt',
    USE_TRAILING_STOP_LOSS = True,
    TRAILING_STOP_LOSS = .2,
    TRAILING_TAKE_PROFIT = .1,
    TRADING_FEE= .0075
    ):
        
        # tracks profit/loss each session
        self.session_profit = 0

        # set to false at Start
        self.bot_paused = False
        
        # Default no debugging
        self.DEBUG = False

        # Load the trading vars
        self.TEST_MODE = TEST_MODE
        self.LOG_TRADES = LOG_TRADES
        self.LOG_FILE = LOG_FILE
        self.DEBUG_SETTING = DEBUG_SETTING
        self.AMERICAN_USER = AMERICAN_USER
        self.PAIR_WITH = PAIR_WITH
        self.QUANTITY = QUANTITY
        self.MAX_COINS = MAX_COINS
        self.FIATS = FIATS
        self.TIME_DIFFERENCE = TIME_DIFFERENCE
        self.RECHECK_INTERVAL = RECHECK_INTERVAL
        self.CHANGE_IN_PRICE = CHANGE_IN_PRICE
        self.STOP_LOSS = STOP_LOSS
        self.CUSTOM_LIST = CUSTOM_LIST
        self.TAKE_PROFIT = TAKE_PROFIT
        self.TICKERS_LIST = TICKERS_LIST
        self.USE_TRAILING_STOP_LOSS = USE_TRAILING_STOP_LOSS
        self.TRAILING_STOP_LOSS = TRAILING_STOP_LOSS
        self.TRAILING_TAKE_PROFIT = TRAILING_TAKE_PROFIT
        self.TRADING_FEE = TRADING_FEE
        
        # Load creds for correct environment
        self.access_key = access_key
        self.secret_key = secret_key

        # Authenticate with the client, Ensure API key is good before continuing
        if self.AMERICAN_USER:
            self.client = Client(self.access_key, self.secret_key, tld='us')
        else:
            self.client = Client(self.access_key, self.secret_key)        

        # If the users has a bad / incorrect API key.
        # this will stop the script from starting, and display a helpful error.
        api_ready, msg = test_api_key(self.client, BinanceAPIException)
        if api_ready is not True:
            exit(f'{txcolors.SELL_LOSS}{msg}{txcolors.DEFAULT}') 

        # Use CUSTOM_LIST symbols if CUSTOM_LIST is set to True
        if CUSTOM_LIST: self.tickers=[line.strip() for line in open(TICKERS_LIST)]

        # try to load all the coins bought by the bot if the file exists and is not empty
        self.coins_bought = {}

        # path to the saved coins_bought file
        self.coins_bought_file_path = 'coins_bought.json'        
    
        # rolling window of prices; cyclical queue
        self.historical_prices = [None] * (TIME_DIFFERENCE * RECHECK_INTERVAL)
        self.hsp_head = -1

        # prevent including a coin in volatile_coins if it has already appeared there less than TIME_DIFFERENCE minutes ago
        self.volatility_cooloff = {}

        # use separate files for testing and live trading
        if TEST_MODE:
            self.coins_bought_file_path = 'test_' + self.coins_bought_file_path    

        # if saved coins_bought json file exists and it's not empty then load it
        if os.path.isfile(self.coins_bought_file_path) and os.stat(self.coins_bought_file_path).st_size!= 0:
            with open(self.coins_bought_file_path) as file:
                    self.coins_bought = json.load(file)       

        print('Press Ctrl-Z to stop the script')

        # seed initial prices
        self.get_price()

    def run_bot(self):
        
        while True:
            try:
                time.sleep((self.TIME_DIFFERENCE*60))
                orders, last_price, volume = self.buy()
                self.update_portfolio(orders, last_price, volume)
                coins_sold = self.sell_coins()
                self.remove_from_portfolio(coins_sold)
            except:
                pass

    def get_price(self, add_to_historical=True):
        '''Return the current price for all coins on binance'''

        initial_price = {}
        prices = self.client.get_all_tickers()

        for coin in prices:

            if self.CUSTOM_LIST:
                if any(item + self.PAIR_WITH == coin['symbol'] for item in self.tickers) and all(item not in coin['symbol'] for item in self.FIATS):
                    initial_price[coin['symbol']] = { 'price': coin['price'], 'time': datetime.now()}
            else:
                if self.PAIR_WITH in coin['symbol'] and all(item not in coin['symbol'] for item in self.FIATS):
                    initial_price[coin['symbol']] = { 'price': coin['price'], 'time': datetime.now()}

        if add_to_historical:
            self.hsp_head += 1

            if self.hsp_head == self.RECHECK_INTERVAL:
                self.hsp_head = 0

            self.historical_prices[self.hsp_head] = initial_price

        return initial_price

    def wait_for_price(self):
        '''calls the initial price and ensures the correct amount of time has passed
        before reading the current price again'''

        volatile_coins = {}
        externals = {}

        coins_up = 0
        coins_down = 0
        coins_unchanged = 0

        self.pause_bot()

        if self.historical_prices[self.hsp_head]['BNB' + self.PAIR_WITH]['time'] > datetime.now() - timedelta(minutes=float(self.TIME_DIFFERENCE / self.RECHECK_INTERVAL)):

            # sleep for exactly the amount of time required
            time.sleep((timedelta(minutes=float(self.TIME_DIFFERENCE / self.RECHECK_INTERVAL)) - (datetime.now() - self.historical_prices[self.hsp_head]['BNB' + self.PAIR_WITH]['time'])).total_seconds())

        print(f'Working...Session profit:{self.session_profit:.2f}% Est:${(self.QUANTITY * self.session_profit)/100:.2f}')

        # retreive latest prices
        self.get_price()

        # calculate the difference in prices
        for coin in self.historical_prices[self.hsp_head]:

            # minimum and maximum prices over time period
            min_price = min(self.historical_prices, key = lambda x: float("inf") if x is None else float(x[coin]['price']))
            max_price = max(self.historical_prices, key = lambda x: -1 if x is None else float(x[coin]['price']))

            threshold_check = (-1.0 if min_price[coin]['time'] > max_price[coin]['time'] else 1.0) * (float(max_price[coin]['price']) - float(min_price[coin]['price'])) / float(min_price[coin]['price']) * 100

            # each coin with higher gains than our CHANGE_IN_PRICE is added to the volatile_coins dict if less than MAX_COINS is not reached.
            if threshold_check > self.CHANGE_IN_PRICE:
                coins_up +=1

                if coin not in self.volatility_cooloff:
                    self.volatility_cooloff[coin] = datetime.now() - timedelta(minutes=self.TIME_DIFFERENCE)

                # only include coin as volatile if it hasn't been picked up in the last TIME_DIFFERENCE minutes already
                if datetime.now() >= self.volatility_cooloff[coin] + timedelta(minutes=self.TIME_DIFFERENCE):
                    self.volatility_cooloff[coin] = datetime.now()

                    if len(self.coins_bought) + len(volatile_coins) < self.MAX_COINS or self.MAX_COINS == 0:
                        #volatile_coins[coin] = round(threshold_check, 3)
                        print(f'{coin} has gained {round(threshold_check, 3)}% within the last {self.TIME_DIFFERENCE} minutes.')
                        
                        # Here goes new code for external signalling
                        externals, _ = self.external_signals()
                        exnumber = 0

                        for excoin in externals:
                            if excoin not in volatile_coins and excoin not in self.coins_bought and (len(self.coins_bought) + exnumber) < self.MAX_COINS:
                                volatile_coins[excoin] = 1
                                exnumber +=1
                                print(f'External signal received on {excoin}, calculating volume in {self.PAIR_WITH}')
                                
                    else:
                        #print(f'{txcolors.WARNING}{coin} has gained {round(threshold_check, 3)}% within the last {TIME_DIFFERENCE} minutes, but you are holding max number of coins{txcolors.DEFAULT}')
                        pass
            elif threshold_check < self.CHANGE_IN_PRICE:
                coins_down +=1

            else:
                coins_unchanged +=1

        # Disabled until fix
        #print(f'Up: {coins_up} Down: {coins_down} Unchanged: {coins_unchanged}')

        return volatile_coins, len(volatile_coins), self.historical_prices[self.hsp_head]        

    def external_signals(self):
        '''TA module to identify which coins to buy and sell'''

        MY_EXCHANGE = 'BINANCE'
        MY_SCREENER = 'CRYPTO'
        MY_FIRST_INTERVAL = Interval.INTERVAL_5_MINUTES
        MY_SECOND_INTERVAL = Interval.INTERVAL_15_MINUTES
        PAIR_WITH = 'USDT'
        TIME_TO_WAIT = 5 # Minutes to wait between analysis
        FULL_LOG = False # List anylysis result to console

        buy_coins = {}
        sell_coins = {}
        first_analysis = {}
        second_analysis = {}
        first_handler = {}
        second_handler = {}
        pairs = {}

        pairs=[line.strip() for line in open(self.TICKERS_LIST)]
        for line in open(self.TICKERS_LIST):
            pairs=[line.strip() + PAIR_WITH for line in open(self.TICKERS_LIST)] 

        for pair in pairs:
        
            first_handler[pair] = TA_Handler(
                symbol=pair,
                exchange=MY_EXCHANGE,
                screener=MY_SCREENER,
                interval=MY_FIRST_INTERVAL,
                timeout= 10
            )
            
            second_handler[pair] = TA_Handler(
                symbol=pair,
                exchange=MY_EXCHANGE,
                screener=MY_SCREENER,
                interval=MY_SECOND_INTERVAL,
                timeout= 10
            )
        
        try:
            first_analysis = first_handler[pair].get_analysis()
            second_analysis = second_handler[pair].get_analysis()
        except Exception as e:
            print("Signalsample:")
            print("Exception:")
            print(e)
            print (f'Coin: {pair}')
            print (f'First handler: {first_handler[pair]}')
            print (f'Second handler: {second_handler[pair]}')

        first_tacheck = first_analysis.summary['BUY']
        first_recommendation = first_analysis.summary['RECOMMENDATION']
        first_RSI = float(first_analysis.indicators['RSI'])

        second_tacheck = second_analysis.summary['BUY']
        second_recommendation = second_analysis.summary['RECOMMENDATION']
        second_RSI = float(second_analysis.indicators['RSI'])

        if (first_recommendation == "BUY" or first_recommendation == "STRONG_BUY") and (second_recommendation == "BUY" or second_recommendation == "STRONG_BUY"):
                if first_RSI <= 67 and second_RSI <= 67 :
                    buy_coins[pair] = pair
                    print(f'Signalsample: Buy Signal detected on {pair}')
     
        elif (first_recommendation == "SELL" or first_recommendation == "STRONG_SELL") and (second_recommendation == "SELL" or second_recommendation == "STRONG_SELL"):
                sell_coins[pair] = pair
                print(f'Signalsample: Sell Signal detected on {pair}')

        return buy_coins, sell_coins

    def pause_bot(self):
        '''Pause the script when exeternal indicators detect a bearish trend in the market'''

        MY_FIRST_INTERVAL = Interval.INTERVAL_5_MINUTES
        MY_SECOND_INTERVAL = Interval.INTERVAL_15_MINUTES

        EXCHANGE = 'BINANCE'
        SCREENER = 'CRYPTO'
        SYMBOL = 'BTCUSDT'

        analysis = {}
        first_handler = {}
        second_handler = {}
        
        first_handler = TA_Handler(
                symbol=SYMBOL,
                exchange=EXCHANGE,
                screener=SCREENER,
                interval=MY_FIRST_INTERVAL,
                timeout= 10)
        
        second_handler = TA_Handler(
                symbol=SYMBOL,
                exchange=EXCHANGE,
                screener=SCREENER,
                interval=MY_SECOND_INTERVAL,
                timeout= 10)

        # start counting for how long the bot's been paused
        start_time = time.perf_counter()

        self.bot_paused == True
        while self.bot_paused == True:
            
            try:
                first_analysis = first_handler.get_analysis()
                second_analysis = second_handler.get_analysis()
            except Exception as e:
                print("pausebotmod:")
                print("Exception:")
                print(e)
            
            first_market_summary = first_analysis.summary['RECOMMENDATION']
            second_market_summary = second_analysis.summary['RECOMMENDATION']

            if first_market_summary == "SELL" or first_market_summary == "STRONG_SELL" or second_market_summary == "SELL" or second_market_summary == "STRONG_SELL":
                self.bot_paused = True
                print(f'pausebotmod: Market not looking too good, bot paused from buying {first_analysis.summary} {second_analysis.summary} .Waiting {TIME_TO_WAIT} minutes for next market checkup')

            else:
                print(f'pausebotmod: Market looks ok, bot is running {first_analysis.summary} {second_analysis.summary} .Waiting {TIME_TO_WAIT} minutes for next market checkup ')
                self.bot_paused = False

            print(f'{txcolors.WARNING}Pausing buying due to change in market conditions, stop loss and take profit will continue to work...{txcolors.DEFAULT}')

            # Sell function needs to work even while paused
            coins_sold = self.sell_coins()
            self.remove_from_portfolio(coins_sold)
            self.get_price(True)

            # pausing here
            if self.hsp_head == 1: print(f'Paused...Session profit:{self.session_profit:.2f}% Est:${(self.QUANTITY * self.session_profit)/100:.2f}')
            time.sleep((self.TIME_DIFFERENCE * 60) / self.RECHECK_INTERVAL)

        else:
            # stop counting the pause time
            stop_time = time.perf_counter()
            time_elapsed = timedelta(seconds=int(stop_time-start_time))

            print(f'{txcolors.WARNING}Resuming buying due to change in market conditions, total sleep time: {time_elapsed}{txcolors.DEFAULT}')

        return

    def convert_volume(self):
        '''Converts the volume given in QUANTITY from USDT to the each coin's volume'''

        volatile_coins, number_of_coins, last_price = self.wait_for_price()
        lot_size = {}
        volume = {}

        for coin in volatile_coins:

            # Find the correct step size for each coin
            # max accuracy for BTC for example is 6 decimal points
            # while XRP is only 1
            try:
                info = self.client.get_symbol_info(coin)
                step_size = info['filters'][2]['stepSize']
                lot_size[coin] = step_size.index('1') - 1

                if lot_size[coin] < 0:
                    lot_size[coin] = 0

            except:
                pass

            # calculate the volume in coin from QUANTITY in USDT (default)
            volume[coin] = float(self.QUANTITY / float(last_price[coin]['price']))

            # define the volume with the correct step size
            if coin not in lot_size:
                volume[coin] = float('{:.1f}'.format(volume[coin]))

            else:
                # if lot size has 0 decimal points, make the volume an integer
                if lot_size[coin] == 0:
                    volume[coin] = int(volume[coin])
                else:
                    volume[coin] = float('{:.{}f}'.format(volume[coin], lot_size[coin]))

        return volume, last_price  

    def buy(self):
        '''Place Buy market orders for each volatile coin found'''
        volume, last_price = self.convert_volume()
        orders = {}

        for coin in volume:

            # only buy if the there are no active trades on the coin
            if coin not in self.coins_bought:
                print(f"{txcolors.BUY}Preparing to buy {volume[coin]} {coin}{txcolors.DEFAULT}")

                if self.TEST_MODE:
                    orders[coin] = [{
                        'symbol': coin,
                        'orderId': 0,
                        'time': datetime.now().timestamp()
                    }]

                    # Log trade
                    if self.LOG_TRADES:
                        self.write_log(f"Buy : {volume[coin]} {coin} - {last_price[coin]['price']}")

                    continue

                # try to create a real order if the test orders did not raise an exception
                try:
                    buy_limit = self.client.create_order(
                        symbol = coin,
                        side = 'BUY',
                        type = 'MARKET',
                        quantity = volume[coin]
                    )

                # error handling here in case position cannot be placed
                except Exception as e:
                    print(e)

                # run the else block if the position has been placed and return order info
                else:
                    orders[coin] = self.client.get_all_orders(symbol=coin, limit=1)

                    # binance sometimes returns an empty list, the code will wait here until binance returns the order
                    while orders[coin] == []:
                        print('Binance is being slow in returning the order, calling the API again...')

                        orders[coin] = self.client.get_all_orders(symbol=coin, limit=1)
                        time.sleep(1)

                    else:
                        print('Order returned, saving order to file')

                        # Log trade
                        if self.LOG_TRADES:
                            self.write_log(f"Buy : {volume[coin]} {coin} - {last_price[coin]['price']}")


            else:
                print(f'Signal detected, but there is already an active trade on {coin}')

        return orders, last_price, volume

    def sell_coins(self):
        '''sell coins that have reached the STOP LOSS or TAKE PROFIT threshold'''

        # Here goes new code for external signalling
        _, externals = self.external_signals()
        #print(externals)
        last_price = self.get_price(False) # don't populate rolling window
        #last_price = get_price(add_to_historical=True) # don't populate rolling window
        coins_sold = {}

        for coin in list(self.coins_bought):
            # define stop loss and take profit
            TP = float(self.coins_bought[coin]['bought_at']) + (float(self.coins_bought[coin]['bought_at']) * self.coins_bought[coin]['take_profit']) / 100
            SL = float(self.coins_bought[coin]['bought_at']) + (float(self.coins_bought[coin]['bought_at']) * self.coins_bought[coin]['stop_loss']) / 100


            LastPrice = float(last_price[coin]['price'])
            BuyPrice = float(self.coins_bought[coin]['bought_at'])
            PriceChange = float((LastPrice - BuyPrice) / BuyPrice * 100)

            # check that the price is above the take profit and readjust SL and TP accordingly if trialing stop loss used
            if LastPrice > TP and self.USE_TRAILING_STOP_LOSS:

                # increasing TP by TRAILING_TAKE_PROFIT (essentially next time to readjust SL)
                self.coins_bought[coin]['take_profit'] = PriceChange + self.TRAILING_TAKE_PROFIT
                self.coins_bought[coin]['stop_loss'] = self.coins_bought[coin]['take_profit'] - self.TRAILING_STOP_LOSS
                print(f"{coin} TP reached, adjusting TP {self.coins_bought[coin]['take_profit']:.2f}  and SL {self.coins_bought[coin]['stop_loss']:.2f} accordingly to lock-in profit")
                continue

            # check that the price is below the stop loss or above take profit (if trailing stop loss not used) and sell if this is the case
            if coin in externals or LastPrice < SL or LastPrice > TP and not self.USE_TRAILING_STOP_LOSS:
                print(f"{txcolors.SELL_PROFIT if PriceChange >= 0. else txcolors.SELL_LOSS}TP or SL reached, selling {self.coins_bought[coin]['volume']} {coin} - {BuyPrice} - {LastPrice} : {PriceChange-(self.TRADING_FEE*2):.2f}% Est:${(self.QUANTITY*(PriceChange-(self.TRADING_FEE*2)))/100:.2f}{txcolors.DEFAULT}")

                # try to create a real order
                try:

                    if not self.TEST_MODE:
                        sell_coins_limit = self.client.create_order(
                            symbol = coin,
                            side = 'SELL',
                            type = 'MARKET',
                            quantity = self.coins_bought[coin]['volume']

                        )

                # error handling here in case position cannot be placed
                except Exception as e:
                    print(e)

                # run the else block if coin has been sold and create a dict for each coin sold
                else:
                    coins_sold[coin] = self.coins_bought[coin]

                    # prevent system from buying this coin for the next TIME_DIFFERENCE minutes
                    self.volatility_cooloff[coin] = datetime.now()

                    # Log trade
                    if self.LOG_TRADES:
                        profit = ((LastPrice - BuyPrice) * coins_sold[coin]['volume'])* (1-(self.TRADING_FEE*2)) # adjust for trading fee here
                        self.write_log(f"Sell: {coins_sold[coin]['volume']} {coin} - {BuyPrice} - {LastPrice} Profit: {profit:.2f} {PriceChange-(self.TRADING_FEE*2):.2f}%")
                        session_profit=session_profit + (PriceChange-(self.TRADING_FEE*2))
                continue

            # no action; print once every TIME_DIFFERENCE
            if self.hsp_head == 1:
                if len(self.coins_bought) > 0:
                    print(f'TP or SL not yet reached, not selling {coin} for now {BuyPrice} - {LastPrice} : {txcolors.SELL_PROFIT if PriceChange >= 0. else txcolors.SELL_LOSS}{PriceChange-(self.TRADING_FEE*2):.2f}% Est:${(self.QUANTITY*(PriceChange-(self.TRADING_FEE*2)))/100:.2f}{txcolors.DEFAULT}')

        if self.hsp_head == 1 and len(self.coins_bought) == 0: print(f'Not holding any coins')
    
        return coins_sold

    def update_portfolio(self, orders, last_price, volume):
        
        '''add every coin bought to our portfolio for tracking/selling later'''
        if self.DEBUG: print(orders)
        for coin in orders:

            self.coins_bought[coin] = {
                'symbol': orders[coin][0]['symbol'],
                'orderid': orders[coin][0]['orderId'],
                'timestamp': orders[coin][0]['time'],
                'bought_at': last_price[coin]['price'],
                'volume': volume[coin],
                'stop_loss': -1 * self.STOP_LOSS,
                'take_profit': self.TAKE_PROFIT,
                }

            # save the coins in a json file in the same directory
            with open(self.coins_bought_file_path, 'w') as file:
                json.dump(self.coins_bought, file, indent=4)

            print(f'Order with id {orders[coin][0]["orderId"]} placed and saved to file')

    def remove_from_portfolio(self, coins_sold):
        '''Remove coins sold due to SL or TP from portfolio'''
        
        for coin in coins_sold:
            self.coins_bought.pop(coin)

        with open(self.coins_bought_file_path, 'w') as file:
            json.dump(self.coins_bought, file, indent=4)

    def write_log(self, logline):
        timestamp = datetime.now().strftime("%d/%m %H:%M:%S")
        with open(self.LOG_FILE,'a+') as f:
            f.write(timestamp + ' ' + logline + '\n')





