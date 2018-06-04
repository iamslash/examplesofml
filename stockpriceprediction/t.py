from pandas_datareader import data
import pandas as pd
import fix_yahoo_finance as yf
yf.pdr_override()
import pickle, re, glob

# download historical data and write to csv file.
start_date = '2018-05-06' #'1996-05-06'
stock_code = '001040.KS'
stock_data = data.get_data_yahoo(stock_code, start_date)
#print(stock_data.head())
csv_fname = './{}.csv'.format(stock_code)
stock_data.to_csv(csv_fname)
print('ok to save {}'.format(csv_fname))

# train data

# write trained model and mark to a.dat