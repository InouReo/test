from __future__ import (absolute_import, division, print_function,
						unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt

import backtrader.analyzers as btanalyzers

import math

PARAMS = (
	('maperiod', 15), #moving average period
	('period', 15), #
	('willperiod', 14), #moving average period
	('sizer', None),
)

# Create a Stratey
class TestStrategy(bt.Strategy):
	params = PARAMS

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
		self.WILLR = bt.indicators.WilliamsR(self.datas[0], period=self.params.willperiod)

		#ta-lib のexample

#		self.sma = bt.talib.SMA(self.data, timeperiod=self.p.period)
#		self.RSI = bt.talib.RSI(self.data, timeperiod=self.p.period)
#		self.WILLR = bt.talib.WILLR(self.data, timeperiod=self.p.period)

#		#Sizer
#		if self.p.sizer is not None:
#			self.sizer = 5

		# Indicators for the plotting show
		bt.indicators.ExponentialMovingAverage(self.datas[0], period=25)
		bt.indicators.WeightedMovingAverage(self.datas[0], period=25, 
											subplot=True)
#		bt.indicators.StochasticSlow(self.datas[0]) 
		bt.indicators.MACDHisto(self.datas[0]) 
#		rsi = bt.indicators.RSI(self.datas[0])
#		bt.indicators.SmoothedMovingAverage(rsi, period=10) 
		bt.indicators.ATR(self.datas[0], plot=False)

	def notify_order(self, order):
		if order.status in [order.Submitted, order.Accepted]:
			# Buy/Sell order submitted/accepted to/by broker - Nothing to do
			return

		# Check if an order has been completed
		# Attention: broker could reject order if not enougth cash
#		if order.status in [order.Completed, order.Canceled, order.Margin]:
#
#			if order.isbuy():
#				self.log(
#					'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
#					(order.executed.price,
#					 order.executed.value,
#					 order.executed.comm))
#
#				self.buyprice = order.executed.price
#				self.buycomm = order.executed.comm
#			else:  # Sell
#				self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
#						 (order.executed.price,
#						  order.executed.value,
#						  order.executed.comm))

			self.bar_executed = len(self)

		# Write down: no pending order
		self.order = None

	def notify_trade(self, trade):
		if not trade.isclosed:
			return
		self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))

	def next(self):
		# Simply log the closing price of the series from the reference
#		self.log('Close, %.2f' % self.dataclose[0])

		# Check if an order is pending ... if yes, we cannot send a 2nd one
		if self.order:
			return

		# Check if we are in the market
		if not self.position:
			# Not yet ... we MIGHT BUY if ...
#			if self.dataclose[0] > self.sma[0]:
			if -80 > self.WILLR[0]:
				# BUY, BUY, BUY!!! (with all possible default parameters)
				self.log('BUY CREATE, %.2f' % self.dataclose[0])
				# Keep track of the created order to avoid a 2nd order
				self.order = self.buy()
		else:
#			if self.dataclose[0] < self.sma[0]:
			if  self.WILLR[0] > -20:
				# SELL, SELL, SELL!!! (with all possible default parameters)
				self.log('SELL CREATE, %.2f' % self.dataclose[0])

				# Keep track of the created order to avoid a 2nd order
				self.order = self.sell()


def cut_changename(datapath,datapath_temp):
	import pandas as pd
	df = pd.read_csv(datapath) #read
	row_num, col_num = df.shape
	df.iloc[:,1:col_num] = df.iloc[:,1:col_num].astype('int') #小数点切り捨て
	df.to_csv(datapath_temp, index=False) #save

#class CloseSMA(bt.Strategy):
##	params = (('period', 15),)
#	params = PARAMS
#
#	def __init__(self):
#		sma = bt.indicators.SMA(self.data, period=self.p.period)
#		self.crossover = bt.indicators.CrossOver(self.data, sma)
#
#	def next(self):
#		if self.crossover > 0:
#			self.buy()
#
#		elif self.crossover < 0:
#			self.sell()

class LongOnly(bt.Sizer):
	params = (('stake', 1),)

	def _getsizing(self, comminfo, cash, data, isbuy):
#		import pdb; pdb.set_trace()
		if isbuy:
			#最大限，買うことができる株の数の計算
			divide = math.floor(cash/data.close[0])
			self.p.stake = divide
			return self.p.stake

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

	# Datas are in a subfolder of the samples. Need to find where the script is
	# because it could have been called from anywhere
	modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
#	datapath = os.path.join('datas/orcl-1995-2014.txt')
#datapath = os.path.join('tensorflow-stock-index-9/data/YH_EN_^N225.csv')
#	datapath = os.path.join('datas/YH_EN_^N225.csv') #BAD
	datapath = os.path.join('datas/YH_JP_1321.csv') #
#	datapath = os.path.join('datas/YH_JP_1301.csv') #OK
#	datapath = os.path.join('datas/YH_JP_1795.csv') #OK

	#pandasで読み込み，
	#小数点を四捨五入し，
	#- _ を使わないファイル名で一時的に保存
	datapath_temp = os.path.join('datas/temp.csv') #このファイル名で一時的に保存
	cut_changename(datapath,datapath_temp)



#	# Create a Data Feed
#	data = bt.feeds.YahooFinanceCSVData(
#		dataname=datapath,
#		# Do not pass values before this date
#		fromdate=datetime.datetime(2014, 10, 1),
#		# Do not pass values before this date
#		todate=datetime.datetime(2014, 12, 31),
#		# Missing values to be replaced with zero (0.0)
#		reverse=True)
	data = bt.feeds.YahooFinanceCSVData(
		dataname=datapath_temp,
#		dataname='datas/temp2.csv',
		# Do not pass values before this date
		fromdate=datetime.datetime(2010, 1, 1),
		# Do not pass values before this date
		todate=datetime.datetime(2016, 12, 31),
		dtformat=('%Y/%m/%d'),
		# Missing values to be replaced with zero (0.0)
		reverse=True)


	# Add the Data Feed to Cerebro
	cerebro.adddata(data)

	# Set our desired cash start
	cerebro.broker.setcash(10000000)

	# Add a FixedSize sizer according to the stake
#	cerebro.addstrategy(CloseSMA)
	cerebro.addsizer(LongOnly)
#	cerebro.addsizer(bt.sizers.FixedSize, stake=200)

	# Set the commission
	cerebro.broker.setcommission(commission=0.0)

	# Print out the starting conditions
	print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

	# Run over everything
	cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='SharpeRatio')
	cerebro.addanalyzer(btanalyzers.DrawDown, _name='DrawDown')
	cerebro.addanalyzer(btanalyzers.AnnualReturn, _name='AnnualReturn')
	cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='TradeAnalyzer')
	thestrats = cerebro.run()
	thestrat = thestrats[0]

	# Print out the final result
	print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

	# Analyzer
	print('SharpeRatio:', thestrat.analyzers.SharpeRatio.get_analysis())
	print('\n')
	print('DrawDown:', thestrat.analyzers.DrawDown.get_analysis())
	print('\n')
	print('AnnualReturn:', thestrat.analyzers.AnnualReturn.get_analysis())
	print('\n')
	print('TradeAnalyzer:', thestrat.analyzers.TradeAnalyzer.get_analysis())

	
	# Plot the result
	cerebro.plot()
	import pdb; pdb.set_trace()

#	cerebro.savefig('test.png') 

