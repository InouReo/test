from __future__ import (absolute_import, division, print_function,
						unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt

import backtrader.analyzers as btanalyzers

import math
import pandas as pd
import talib
import numpy as np


#['1552','ETF VIX短期先物指数','NASDAQ'],
#['2049','NEXT NOTES S&P500 VIX インバースETN','NASDAQ'],

PARAMS = (
	('maperiod', 15), #moving average period
	('period', 15), #
	('willperiod', 14), #moving average period
	('sizer', None),
)

DATAPATH_TEMP = 'datas/temp.csv' #このファイル名で一時的に保存
DATEFORMAT = '%Y-%m-%d'
INDEX_COL = 'Date' #pandasでreadするときに，Dateを1列目とする

#計算に用いるbrandの名称
STOCK_NAME       = 'YH_JP_2049'
#STOCK_NAME       = 'YH_JP_1552'
VIX_NAME         = 'VIX'
VIX_FUTURE_NAME  = 'VIX_FUTURE'
DATA_PATH = 'datas/'
SAVE_NAME = DATA_PATH + STOCK_NAME + '_temp.csv' #一時的にデータを保存するファイル名
RATIO_NAME = 'Ratio' #VIX/VIX_FUTUREの計算結果を保存する列名

#BUY_RATIOを変数と定義し、
#ratio > BUY_RATIOのときXIV(2049)を購入
#ratio < YのときVXX(1552)を購入するときのプログラムは後で書く-
BUY_RATIO = 0.05 #BUY_RATIO % 以上のとき購入


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
		#ta-lib のexample
#		self.sma = bt.talib.SMA(self.data, timeperiod=self.p.period)
#		self.RSI = bt.talib.RSI(self.data, timeperiod=self.p.period)
#		self.WILLR = bt.talib.WILLR(self.data, timeperiod=self.p.period)

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
		self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))

	def next(self):
		# Simply log the closing price of the series from the reference
#		self.log('Close, %.2f' % self.dataclose[0])
		
		# 現在, 計算に用いている日付の取得
		date_now = self.datas[0].datetime.date(0)
		date_now = (date_now.strftime(DATEFORMAT)) #format の変換
		#read
		df = pd.read_csv(SAVE_NAME,index_col='Date') #read






		# Check if an order is pending ... if yes, we cannot send a 2nd one
		if self.order:
			return

		# Check if we are in the market
		if not self.position:
			# Not yet ... we MIGHT BUY if ...
#			if self.dataclose[0] > self.sma[0]:
#			if -80 > self.WILLR[0]:

			if df.loc[date_now][RATIO_NAME] > BUY_RATIO:
				# BUY, BUY, BUY!!! (with all possible default parameters)
				self.log('BUY CREATE, %.2f' % self.dataclose[0])
				# Keep track of the created order to avoid a 2nd order
				self.order = self.buy()
#				self.buy()
		else:
			if df.loc[date_now][RATIO_NAME] <= BUY_RATIO:
				self.log('SELL CREATE, %.2f' % self.dataclose[0])

				# Keep track of the created order to avoid a 2nd order
				self.order = self.sell()
#				self.sell()



class LongOnly(bt.Sizer):
#	params = (('stake', 1),)

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

def calculate_WVF(datapath_temp):
	'''dataに対応するCSVファイルをPandasで，WVFを計算
	Args:
		data: csvで読み込んだdata
	Returns:
		
	'''
	#read data
	df = pd.read_csv(datapath) #read
	''' initialize
	'''
	#Editable values to tweak backtest
#		context.PercentAllocationPerTrade = 1
#		context.StopLossPct = 0.03
	AvgLength = 20 #移動平均の計算に用いる日数
	LengthWVF = 100 #高値を求めるときの日数の範囲を決める指標
	LengthEMA1 = 10
	LengthEMA2 = 30
	
	#internal variables to store data
#		context.SellAlert = False 
	dataAvg = 0 
	SmoothedWVF1 = 0
	SmoothedWVF2 = 0

	'''ここからコメントアウト
	'''
	'''get each close, low, highest
	'''
#		#get data for calculations
#		n = 200
	data_prices = df['Close']
	data_lows = pd.DataFrame(df['Low'].values)



	#ある範囲のmaxの高値の取得
	data_highest_sort = pd.DataFrame(data_prices.sort_index(axis=0, ascending=False).values).rolling(window = LengthWVF, center=False, min_periods=1).max()
#	data_highest_sort = pd.DataFrame(data_prices.sort_index(axis=0, ascending=False).values).rolling(window = LengthWVF, center=False, min_periods=5).max()
	data_highest = pd.DataFrame(data_highest_sort.sort_index(axis=0, ascending=False).values) #順序を戻す

	'''define buy and sell rules as logits
	'''
	#William's VIX Fix indicator a.k.a. the Synthetic VIX
	WVF = ((data_highest - data_lows)/(data_highest))*100

	
#	TypeError: Argument 'real' has incorrect type (expected numpy.ndarray, got Series)
	# calculated smoothed WVF
	#.values(): transform from pandas to numpy

	#sort して，計算
	SmoothedWVF1 = talib.EMA(WVF.sort_index(axis=0, ascending=False).T.values.astype(float)[0], timeperiod=LengthEMA1)
	SmoothedWVF2 = talib.EMA(WVF.sort_index(axis=0, ascending=False).T.values.astype(float)[0], timeperiod=LengthEMA2)
	'''ここまでコメントアウト
	'''

#	#新しく書き直してる最中
#	data_highest_sort = pd.DataFrame(df.sort_index(axis=0, ascending=False).values).rolling(window = LengthWVF, center=False, min_periods=1).max()
#	data_prices = data_highest_sort[4][300:500] #close
#	import pdb; pdb.set_trace()
#	data_highest = data_prices.rolling(window = LengthWVF,center=False).max()
#	WVF = ((data_highest - data_lows)/(data_highest))*100
#	import pdb; pdb.set_trace()
#	#新しく書き直してる最中

	#移動平均の計算
	SMA = talib.SMA(df['Close'].sort_index(axis=0, ascending=False).values.astype(float), timeperiod=AvgLength)
#	data_lows = pd.DataFrame(data_highest_sort[3][300:500].values) #low


	#WVFの計算結果を加える
	df['close_highest'] = data_highest
	df['WVF'] = WVF
	df['SmoothedWVF1'] = pd.DataFrame(SmoothedWVF1).sort_index(axis=0, ascending=False).values
	df['SmoothedWVF2'] = pd.DataFrame(SmoothedWVF2).sort_index(axis=0, ascending=False).values
	df['SMA'] = pd.DataFrame(SMA).sort_index(axis=0, ascending=False).values

	#データの一時的な保存
	row_num, col_num = df.shape
	df.to_csv(datapath_temp, index=False) #save

#VIX_FUTUREの前処理，他のデータと取得先が違うので加工を行う
def preprocessing_VIX_FUTURE(file_name_vixfuture_original,file_name_vixfuture_temp):
	temp = pd.read_csv(file_name_vixfuture_original)
	#Dateの型をtimestampからobjectに変更(dtypeで確認できる)
	for i in range(temp.shape[0]):
		temp.iloc[i,0] = pd.to_datetime(temp.iloc[i,0]).strftime("%Y-%m-%d")
	#一時的に保存
	temp.to_csv(file_name_vixfuture_temp, index=False)

def delete_complement(stocks):
	'''
	1.計算対象のbrandの日付を取得し，他のbrandがその日付以外のデータを持っている場合，削除
	2.計算対象のbrandの日付を, 他のbrandがその日付のデータを持っていない場合、直近の前の日のデータをコピーして補完する

	Returns:
		{datas[n]: pd.DataFrame()}
	'''
	# 株価を読み込む
	datas = {}
	for (name, stock) in stocks.items():
		datas[name] = stock

	# 計算対象の日付に存在しないデータを削除する
	if REMOVE_NIL_DATE:
		## 1.
		target_indexes = datas[target_brand].index
		for (exchange, data) in datas.items():
			for index in data.index:
				if not index in target_indexes:
					datas[exchange] = datas[exchange].drop(index)

		## 2.
		#target_brandが日付データを持っているが，VIXがその日付データを持っていない場合，
		#前日の日付データで補完する．
		for (exchange, data) in datas.items():
			for index in target_indexes:
				if not index in data.index:
					#stringを時刻フォーマットに変換（分割）
					tdatetime_pre = datetime.datetime.strptime(index, '%Y-%m-%d')
					#直近の前の日付の検索(最短で1日)
					while 1:
						tdatetime_pre = tdatetime_pre - datetime.timedelta(days=1) #1日減算
						tdatetime_pre_str = tdatetime_pre.strftime('%Y-%m-%d') #文字列に再変換
#						print(tdatetime_pre_str)
						if tdatetime_pre_str in data.index:
							break

					#直近の前の日付データをindex日付で挿入，
					df_append = pd.DataFrame(data.loc[tdatetime_pre_str])
					df_append = df_append.T
					df_append.index = [index]
					datas[exchange] = datas[exchange].append(df_append)
			#昇順にソート
			datas[exchange] = datas[exchange].sort_index(ascending=False)
	
	return datas

def calc_ratio(save_name):
	'''
	ratio=(VIX_FUTURE/VIX)-1 計算し、save_nameに保存
	'''
	ratio_name = RATIO_NAME #計算結果の列名
	calc_target = 'Close' #closeの値を用いて計算
	stocks[STOCK_NAME][ratio_name] = stocks[VIX_FUTURE_NAME][calc_target]/stocks[VIX_NAME][calc_target]-1
	stocks[STOCK_NAME].to_csv(save_name)


def cut_changename(datapath,datapath_temp):
	import pandas as pd
	df = pd.read_csv(datapath) #read
	row_num, col_num = df.shape
	df.iloc[:,1:col_num] = df.iloc[:,1:col_num].astype('int') #小数点切り捨て
	df.to_csv(datapath_temp, index=False) #save


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
#	datapath = os.path.join('datas/YH_EN_^N225.csv')
#	datapath = os.path.join('datas/YH_EN_^VXX.csv') #
#	datapath = os.path.join('datas/YH_JP_1552.csv') #
#	datapath = os.path.join('datas/YH_JP_1301.csv')
#	datapath = os.path.join('datas/YH_JP_1321.csv') #日経225連動型上場投資信託
#	datapath = os.path.join('tensorflow-stock-index-9/data/' +  STOCK_NAME +  '.csv') #XIV S&P VIXインバース
#	datapath = os.path.join('datas/YH_JP_1578.csv') #上場インデックスファンド日経225(ミニ)

	#pandasで読み込み，
	#小数点を四捨五入し，
	#- _ を使わないファイル名で一時的に保存
	datapath_temp = os.path.join(DATAPATH_TEMP) #このファイル名で一時的に保存
	cut_changename(DATA_PATH+STOCK_NAME +'.csv',datapath_temp)

	#VIX_FUTUREだけ，日付のフォーマットが違うので加工
#	df_VIX_FUTURE = pd.read_csv('datas/YH_EN_^VIX_FUTURE.csv')#read
	data_format = '%d-%b-%y'
	my_parser = lambda date: pd.datetime.strptime(date, data_format)
#	pd.read_csv('datas/YH_EN_^VIX_FUTURE.csv', index_col='Date',date_parser=my_parser)

	#VIX_FUTUREを読み込み，日付を加工してfile_name_vixfuture_tempに保存
	file_name_vixfuture_original = 'datas/YH_EN_^VIX_FUTURE.csv'
	file_name_vixfuture_temp = 'datas/YH_EN_^VIX_FUTURE_temp.csv'
	preprocessing_VIX_FUTURE(file_name_vixfuture_original,file_name_vixfuture_temp)

	REMOVE_NIL_DATE = True  # 計算対象の日付に存在しないデータを削除する
	target_brand = STOCK_NAME

	#値の格納, matlabでいうcellみたいな感じ?
	stocks = {
			STOCK_NAME       : pd.read_csv(datapath_temp, index_col='Date')                ,
			VIX_NAME         : pd.read_csv('datas/YH_EN_^VIX.csv', index_col='Date')       ,
			VIX_FUTURE_NAME  : pd.read_csv(file_name_vixfuture_temp, index_col='Date'),
			}

	'''def delete_complement
	1.計算対象のbrandの日付を取得し，他のbrandがその日付以外のデータを持っている場合，削除
	2.計算対象のbrandの日付を, 他のbrandがその日付のデータを持っていない場合、直近の前の日のデータをコピーして補完する
	'''
	stocks = delete_complement(stocks)
	
	'''
	ratio=(VIX_FUTURE/VIX)-1 計算し、保存
	'''
	calc_ratio(SAVE_NAME)



#	# Create a Data Feed
	data = bt.feeds.YahooFinanceCSVData(
		dataname=SAVE_NAME,
#		dataname='datas/temp2.csv',
		# Do not pass values before this date
		fromdate=datetime.datetime(2015, 1, 1),
		# Do not pass values before this date
		todate=datetime.datetime(2015, 12, 31),
#		dtformat= '%Y/%m/%d',
		# Missing values to be replaced with zero (0.0)
		reverse=True)

#	import matplotlib.pyplot as plt
##	fig = plt.figure()
#	plt.figure()
#	# matplotlib で
#	# close, WVF, SmoothedWVF1, SmoothedWVF2を確認
#	#0-100の間で乱数を発生させて6要素のリストを作成
#	df = pd.read_csv(datapath_temp) #read
#
#	dt = pd.to_datetime(df['Date']) #make timestanp
#
#	plt.subplot(4,1,1)
#	plt.plot(dt, df['Close'], label='Close')
##	plt.plot(dt, df['close_highest'], label='close_highest')
#
#	plt.subplot(4,1,2)
#	plt.plot(dt, df['WVF'], label='WVF')
##	plt.plot(df['SMA'], label='SMA')
#	plt.subplot(4,1,3)
#	plt.plot(dt, df['SmoothedWVF1'], label='SmoothedWVF1')
#	plt.legend()  # 凡例をグラフにプロット
#	plt.subplot(4,1,4)
#	plt.plot(dt, df['SmoothedWVF2'], label='SmoothedWVF2')
#	plt.legend()  # 凡例をグラフにプロット
##	plt.show()
#	plt.pause(interval=0.1) #操作中断なし
#	import pdb; pdb.set_trace()

#	VXX_MAVG   : context.vxxAvg #移動平均
#	wvf_vxx    : WVF[-1] #WVF
#	SmoothWVF1 : SmoothedWVF1
#	SmoothWVF2 : SmoothedWVF2



	# Add the Data Feed to Cerebro
	cerebro.adddata(data)

	# Set our desired cash start
	cerebro.broker.setcash(10000000)

	# Add a FixedSize sizer according to the stake
#	cerebro.addstrategy(CloseSMA)
	cerebro.addsizer(LongOnly)
	#set close to open
	cerebro.broker.set_coc(True)



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
#	print('TradeAnalyzer:', thestrat.analyzers.TradeAnalyzer.get_analysis())

	
	# Plot the result
	cerebro.plot()
	import pdb; pdb.set_trace()

#	cerebro.savefig('test.png') 

