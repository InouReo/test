from __future__ import (absolute_import, division, print_function,
						unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt

import math


# Create a Stratey
class TestStrategy(bt.Strategy):
	params = (
		('maperiod', 15), #moving average period
		('period', 15), #
	)

	def log(self, txt, dt=None):
		''' Logging function fot this strategy'''
		dt = dt or self.datas[0].datetime.date(0)
		print('%s, %s' % (dt.isoformat(), txt))

	def __init__(self):
		# Keep a reference to the "close" line in the data[0] dataseries

		self.dataclose = self.datas[0].close

		# To keep track of pending orders and buy price/commission
		self.order = None
		self.buyprice = None
		self.buycomm = None

##		# Add a MovingAverageSimple indicator
		self.sma = bt.indicators.SimpleMovingAverage( 
			self.datas[0], period=self.params.maperiod)

		#ta-lib のexample

#		self.sma = bt.talib.SMA(self.data, timeperiod=self.p.period)
		self.RSI = bt.talib.RSI(self.data, timeperiod=self.p.period)


		# Indicators for the plotting show
		bt.indicators.ExponentialMovingAverage(self.datas[0], period=25) #exponetial移動平均
		bt.indicators.WeightedMovingAverage(self.datas[0], period=25, #重み付け移動平均
											subplot=True)
		bt.indicators.StochasticSlow(self.datas[0]) #ストキャスティクス
		bt.indicators.MACDHisto(self.datas[0]) #「Moving Average Convergence/Divergence Trading Method」の略で、日本語では移動平均・収束・拡散手法と言います。
		rsi = bt.indicators.RSI(self.datas[0]) #RSI
		bt.indicators.SmoothedMovingAverage(rsi, period=10) 
		bt.indicators.ATR(self.datas[0], plot=False)

	def notify_order(self, order):
		if order.status in [order.Submitted, order.Accepted]:
			# Buy/Sell order submitted/accepted to/by broker - Nothing to do
			return

		# Check if an order has been completed
		# Attention: broker could reject order if not enougth cash
		if order.status in [order.Completed, order.Canceled, order.Margin]:

			if order.isbuy():
				self.log(
					'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
					(order.executed.price,
					 order.executed.value,
					 order.executed.comm))

				self.buyprice = order.executed.price
				self.buycomm = order.executed.comm
			else:  # Sell
				self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
						 (order.executed.price,
						  order.executed.value,
						  order.executed.comm))

			self.bar_executed = len(self)

		# Write down: no pending order
		self.order = None

	def notify_trade(self, trade):
		if not trade.isclosed:
			return

		self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
				 (trade.pnl, trade.pnlcomm))

	def next(self):
		# Simply log the closing price of the series from the reference
#		self.log('Close, %.2f' % self.dataclose[0])

		# Check if an order is pending ... if yes, we cannot send a 2nd one
		if self.order:
			return
		# Check if we are in the market
		if not self.position:

#			import pdb; pdb.set_trace()
			# Not yet ... we MIGHT BUY if ...
#			if self.dataclose[0] > self.sma[0]:
			if self.RSI[0] < 30:
				# BUY, BUY, BUY!!! (with all possible default parameters)
				self.log('BUY CREATE, %.2f' % self.dataclose[0])
				# Keep track of the created order to avoid a 2nd order
				self.order = self.buy()
		else:
#			if self.dataclose[0] < self.sma[0]:
			if self.RSI[0] > 70:
				# SELL, SELL, SELL!!! (with all possible default parameters)
				self.log('SELL CREATE, %.2f' % self.dataclose[0])

				# Keep track of the created order to avoid a 2nd order
				self.order = self.sell()

class CloseSMA(bt.Strategy):
	params = (('period', 15),)

	def __init__(self):
		sma = bt.indicators.SMA(self.data, period=self.p.period)
		self.crossover = bt.indicators.CrossOver(self.data, sma)

	def next(self):
		if self.crossover > 0:
			self.buy()

		elif self.crossover < 0:
			self.sell()

class LongOnly(bt.Sizer):
#	params = (('stake', 1),)

	def _getsizing(self, comminfo, cash, data, isbuy):
		if isbuy:
			divide = math.floor(cash/data.close[0])

#			import pdb; pdb.set_trace()
			self.p.stake = divide


			return divide

		# Sell situation
		position = self.broker.getposition(data)
		if not position.size:
			return 0  # do not sell if nothing is open

		return self.p.stake


if __name__ == '__main__':
	# Create a cerebro entity
	cerebro = bt.Cerebro()

	# Add a strategy
	cerebro.addstrategy(TestStrategy)
	cerebro.addsizer(LongOnly)

	# Datas are in a subfolder of the samples. Need to find where the script is
	# because it could have been called from anywhere
	modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
#	datapath = os.path.join('datas/orcl-1995-2014.txt')
#datapath = os.path.join('tensorflow-stock-index-9/data/YH_EN_^N225.csv')
#	datapath = os.path.join('datas/YH_EN_^N225.csv')
	datapath = os.path.join('datas/test2.txt')


#	# Create a Data Feed
#	data = bt.feeds.YahooFinanceCSVData(
#		dataname=datapath,
#		# Do not pass values before this date
#		fromdate=datetime.datetime(2000, 1, 1),
#		# Do not pass values before this date
#		todate=datetime.datetime(2000, 12, 31),
#		# Missing values to be replaced with zero (0.0)
#		nullvalue=0.0)
#	data = bt.feeds.GenericCSVData(
#		dataname=datapath,
#		# Do not pass values before this date
#		fromdate=datetime.datetime(2000, 1, 1),
#		# Do not pass values before this date
#		todate=datetime.datetime(2002, 12, 31),
#		# Missing values to be replaced with zero (0.0)
#		nullvalue=0.0,
#		dtformat=('%Y-%m-%d'),
#		reverse=True)

	data = bt.feeds.GenericCSVData(
		dataname=datapath,
	
		fromdate=datetime.datetime(2009, 1, 2),
		todate=datetime.datetime(2016, 12, 31),
	
#		dtformat=('%Y-%m-%d'),
		dtformat=('%Y/%m/%d'),
	
		datetime=0,
		open=1,
		high=2,
		low=3,
		close=4,
		volume=5,
		openinterest=-1
	)

#	import pdb; pdb.set_trace()

	# Add the Data Feed to Cerebro
	cerebro.adddata(data)

	# Set our desired cash start
	cerebro.broker.setcash(10000000)

	# Add a FixedSize sizer according to the stake
#	cerebro.addsizer(bt.sizers.FixedSize, stake=10)
#	cerebro.addstrategy(CloseSMA)
#	cerebro.addsizer(LongOnly)

	# Set the commission
	cerebro.broker.setcommission(commission=0.0)

	# Print out the starting conditions
	print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

	# Run over everything
	cerebro.run()

	# Print out the final result
	print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

	# Plot the result
	cerebro.plot()
