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

# Keras library
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# trading view library
from tradingview_ta import TA_Handler, Interval, Exchange

# use for environment variables
import os
import datetime

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
    test_api_key
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
    QUANTITY = 1000,
    MAX_COINS = 1,
    FIATS = ['EURUSDT', 'GBPUSDT', 'JPYUSDT', 'USDUSDT', 'DOWN', 'UP'],
    TIME_DIFFERENCE = 5,
    RECHECK_INTERVAL = 6,
    CHANGE_IN_PRICE = 0,
    STOP_LOSS = 5,
    TAKE_PROFIT = 5,
    CUSTOM_LIST = False,
    TICKERS_LIST = 'tickers.txt',
    TICKER = "BTC",
    USE_TRAILING_STOP_LOSS = True,
    TRAILING_STOP_LOSS = .2,
    TRAILING_TAKE_PROFIT = .1,
    TRADING_FEE= .0075,
    LOAD_MODEL = None,
    SAVE_MODEL = 'models/my_model'
    ):
        
        # tracks profit/loss each session
        self.session_profit = 0

        # set to false at Start
        self.bot_paused = False
        self.action = None
        
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
        self.TICKER = TICKER
        
        # Model Parameters
        self.LOAD_MODEL = LOAD_MODEL
        self.SAVE_MODEL = SAVE_MODEL

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
            #try:
            #self.pause_bot()
            orders, last_price, volume = self.buy()
            self.update_portfolio(orders, last_price, volume)
            coins_sold = self.sell_coins()
            self.remove_from_portfolio(coins_sold)
            #except:
                #pass

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

        #self.pause_bot()

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
                        #print(f'{coin} has gained {round(threshold_check, 3)}% within the last {self.TIME_DIFFERENCE} minutes.')
                        
                        # Here goes new code for external signalling
                        externals, _= self.external_signals()
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

        return volatile_coins, len(volatile_coins), self.historical_prices[self.hsp_head]        

    def external_signals(self):
        '''TA module to identify which coins to buy and sell'''

        buy_coins = {}
        sell_coins = {}
        analysis = {}
        handler = {}
        pairs = {}

        return buy_coins, sell_coins

    def pause_bot(self):
        '''Pause the script when exeternal indicators detect a bearish trend in the market'''

        MY_FIRST_INTERVAL = Interval.INTERVAL_5_MINUTES
        MY_SECOND_INTERVAL = Interval.INTERVAL_15_MINUTES

        EXCHANGE = 'BINANCE'
        SCREENER = 'CRYPTO'
        SYMBOL = 'BTCUSDT'
        PAUSE_INTERVAL = 1 # Minutes

        analysis = {}
        first_handler = {}
        second_handler = {}
        
        # start counting for how long the bot's been paused
        start_time = time.perf_counter()

        self.bot_paused = True
        while self.bot_paused == True:
            
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
                print(f'Market not looking too good, bot paused from buying {first_analysis.summary} {second_analysis.summary} .Waiting {PAUSE_INTERVAL} minutes for next market checkup')

            else:
                print(f'Market looks ok, bot is running {first_analysis.summary} {second_analysis.summary}')
                self.bot_paused = False
                return

            print(f'{txcolors.WARNING}Pausing buying due to change in market conditions, stop loss and take profit will continue to work...{txcolors.DEFAULT}')

            # Sell function needs to work even while paused
            coins_sold = self.sell_coins()
            self.remove_from_portfolio(coins_sold)
            self.get_price(True)

            # pausing here
            if self.hsp_head == 1: print(f'Paused...Session profit:{self.session_profit:.2f}% Est:${(self.QUANTITY * self.session_profit)/100:.2f}')
            time.sleep(PAUSE_INTERVAL)

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
                        self.session_profit=self.session_profit + (PriceChange-(self.TRADING_FEE*2))
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

    def market_state(self):
        '''Get the current market state'''
        
        EXCHANGE = 'BINANCE'
        SCREENER = 'CRYPTO'
        INTERVAL_LIST = [Interval.INTERVAL_1_MINUTE, Interval.INTERVAL_5_MINUTES, Interval.INTERVAL_15_MINUTES, 
                 Interval.INTERVAL_1_HOUR, Interval.INTERVAL_4_HOURS, Interval.INTERVAL_1_DAY, Interval.INTERVAL_1_WEEK,
                 Interval.INTERVAL_1_MONTH] 
        
        state = []
        for INTERVAL in INTERVAL_LIST:
            
            try:
                handler = TA_Handler(symbol=self.TICKER + self.PAIR_WITH,
                                exchange=EXCHANGE,
                                screener=SCREENER,
                                interval=INTERVAL,
                                timeout= 10)
            except Exception as e:
                print("market_state:")
                print("Exception:")
                print(e)
                print (f'Coin: {self.TICKER }')
            
            analysis = handler.get_analysis()
            data = list(analysis.indicators.values())
            state.append(data)
        
        state = np.asarray(state).astype('float32')
        state = np.nan_to_num(state)
        state = state.flatten()
        return state

    def deepQModel(self, num_inputs = 696):
        '''Build a neural network model'''

        num_actions = 3 # Buy, Sell, Hold
        num_hidden = 10

        # Minute Input
        inp_time_minute = layers.Input(shape=(1, ))
        emb_t1 = layers.Embedding(61, 4)(inp_time_minute)
        emb_t1 = layers.Flatten()(emb_t1)

        # Hour Input
        inp_time_hour = layers.Input(shape=(1, ))
        emb_t2 = layers.Embedding(25, 4)(inp_time_hour)
        emb_t2 = layers.Flatten()(emb_t2)

        # Day Input
        inp_time_day = layers.Input(shape=(1, ))
        emb_t3 = layers.Embedding(32, 4)(inp_time_day)
        emb_t3 = layers.Flatten()(emb_t3)

        # Month Input
        inp_time_month = layers.Input(shape=(1, ))
        emb_t4 = layers.Embedding(13, 4)(inp_time_month)
        emb_t4 = layers.Flatten()(emb_t4)

        # TA Input
        inputs = layers.Input(shape=(num_inputs,))

        # Concatenate Inputs
        conct1 = layers.Concatenate(axis=-1)([inputs, emb_t1, emb_t2, emb_t3, emb_t4])

        common_1 = layers.Dense(num_hidden, activation="relu")(conct1)
        common_2 = layers.Dense(num_hidden, activation="relu")(common_1)
        common_3 = layers.Dense(num_hidden, activation="relu")(common_2)
        action = layers.Dense(num_actions, activation="softmax")(common_3)
        critic = layers.Dense(1)(common_3)

        model = keras.Model(inputs=[inp_time_minute, inp_time_hour, inp_time_day, inp_time_month, inputs], outputs=[action, critic])

        return model

    def trainNetwork(self):
        '''Train the RL Network'''

        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        huber_loss = keras.losses.Huber()
        action_probs_history = []
        critic_value_history = []
        rewards_history = []
        running_reward = 0
        episode_count = 0
        max_steps_per_episode = 10
        gamma = 0.99
        eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
        num_actions = 3 # Buy, Sell, Hold
        action = None
        
        if self.LOAD_MODEL != None:
            # Load an existing model
            model = keras.models.load_model(self.LOAD_MODEL)
        else:
            # Load a New Model
            model = self.deepQModel()

        while True:  # Run until solved
            state = self.market_state()
            episode_reward = 0
            
            with tf.GradientTape() as tape:
                for timestep in range(1, max_steps_per_episode):
                    
                    iteration_profit = 0
                    print(f'DeepQBot: Training timestep {timestep} of {max_steps_per_episode}')

                    # Get the current price of coin
                    price_list = self.get_price(add_to_historical=False)
                    price_current = float(price_list[self.TICKER + self.PAIR_WITH]['price'])
                    #print(f"DeepQBot: Price of {self.TICKER} is {price_current}")

                    #iteration_profit = self.session_profit
                    state = tf.convert_to_tensor(state)
                    state = tf.expand_dims(state, 0)

                    # Get Time
                    now = datetime.now()
                    
                    minute = tf.convert_to_tensor(float(now.minute))
                    minute = tf.expand_dims(minute, 0)
                   
                    hour = tf.convert_to_tensor(float(now.hour))
                    hour = tf.expand_dims(hour, 0)
                    
                    day = tf.convert_to_tensor(float(now.day))
                    day = tf.expand_dims(day, 0)
                    
                    month = tf.convert_to_tensor(float(now.month))
                    month = tf.expand_dims(month, 0)

                    # Predict action probabilities and estimated future rewards
                    # from environment state
                    action_probs, critic_value = model([minute, hour, day, month, state])
                    critic_value_history.append(critic_value[0, 0])

                    # Sample action from action probability distribution
                    action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                    action_probs_history.append(tf.math.log(action_probs[0, action]))

                    # ACTION LIST:
                    # 0: BUY
                    # 1: SELL
                    # 2: HOLD

                    print(f'DeepQBot: Action Recieved: {action}')

                    # Apply the sampled action in our environment
                    #self.step()
                    # Wait before getting a new price list
                    print(f"DeepQBot: Waiting {self.TIME_DIFFERENCE} minutes for next analysis.")
                    time.sleep(self.TIME_DIFFERENCE * 60)

                    # Check the new price
                    price_list = self.get_price(add_to_historical=False)
                    price_new = float(price_list[self.TICKER + self.PAIR_WITH]['price'])
                    #print(f"DeepQBot: Price of {self.TICKER} is {price_new}")
                    
                    # REWARD BLOCK
                    if (action == 0 or action == 2) and price_new > price_current:
                        iteration_profit = price_new - price_current # Positive Reward
                    elif (action == 0 or action == 2) and price_new < price_current:
                        iteration_profit = price_new - price_current # Negative Reward   
                    elif action == 1 and price_new < price_current:
                        iteration_profit = price_current - price_new # Positive Reward
                    elif action == 1 and price_new > price_current:
                        iteration_profit = price_current - price_new # Negative Reward
                    else:
                        iteration_profit = 0

                    print(f'DeepQBot: Timestep Profit: {iteration_profit}')

                    rewards_history.append(iteration_profit)
                    episode_reward += iteration_profit

                    # Get new state
                    state = self.market_state()

                # Update running reward to check condition for solving
                running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
                print(f'DeepQBot: Total Profit: {running_reward}')

                # Calculate expected value from rewards
                # - At each timestep what was the total reward received after that timestep
                # - Rewards in the past are discounted by multiplying them with gamma
                # - These are the labels for our critic
                returns = []
                discounted_sum = 0
                for r in rewards_history[::-1]:
                    discounted_sum = r + gamma * discounted_sum
                    returns.insert(0, discounted_sum)

                # Normalize
                returns = np.array(returns)
                returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
                returns = returns.tolist()

                # Calculating loss values to update our network
                history = zip(action_probs_history, critic_value_history, returns)
                actor_losses = []
                critic_losses = []
                for log_prob, value, ret in history:
                    # At this point in history, the critic estimated that we would get a
                    # total reward = `value` in the future. We took an action with log probability
                    # of `log_prob` and ended up recieving a total reward = `ret`.
                    # The actor must be updated so that it predicts an action that leads to
                    # high rewards (compared to critic's estimate) with high probability.
                    diff = ret - value
                    actor_losses.append(-log_prob * diff)  # actor loss

                    # The critic must be updated so that it predicts a better estimate of
                    # the future rewards.
                    critic_losses.append(
                        huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                    )

                # Backpropagation
                loss_value = sum(actor_losses) + sum(critic_losses)
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # Clear the loss and reward history
                action_probs_history.clear()
                critic_value_history.clear()
                rewards_history.clear()

            # Log details
            episode_count += 1
            if episode_count % 10 == 0:
                template = "running reward: {:.2f} at episode {}"
                print(template.format(running_reward, episode_count))

            # Save the Model
            model.save(self.SAVE_MODEL)



    



